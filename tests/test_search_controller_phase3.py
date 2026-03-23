"""Phase 3 tests for controller loop, beam control, and termination behavior."""

import unittest

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.search.controller import ControllerConfig, run_search
from mgvs.state.models import ReasoningState, create_initial_state
from mgvs.types import StateStatus


class FakeVerifier:
    """Simple verifier that rejects blocked actions and invalid states."""

    def is_action_valid(self, state: ReasoningState, action: CandidateAction) -> bool:
        _ = state
        return action.title != "blocked"

    def is_state_valid(self, state: ReasoningState) -> bool:
        return state.normalized_form != "invalid"


class FakeCanonicalizer:
    """Canonicalizer that derives normalized_form from outputs when missing."""

    def canonicalize(self, state: ReasoningState) -> ReasoningState:
        outputs = state.symbolic_objects.get("last_outputs")
        if isinstance(outputs, list) and outputs:
            base = "|".join(str(item) for item in outputs)
            if state.branch_assignments:
                base = f"{base}::{state.branch_assignments[-1]}"
            state.normalized_form = base
            return state

        state.normalized_form = (
            f"{state.status.value}:{','.join(state.derived_facts)}:{','.join(state.branch_assignments)}"
        )
        return state


class SinglePathProposer:
    """Returns one useful action and one blocked action at depth zero."""

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        _ = state
        if depth != 0:
            return []
        return [
            CandidateAction(
                action_type=ActionType.REWRITE,
                title="blocked",
                rationale="Should be filtered by verifier",
            ),
            CandidateAction(
                action_type=ActionType.REWRITE,
                title="valid_progress",
                rationale="Simple progression",
                added_facts=["f1"],
                outputs=["nf1"],
            ),
        ]


class BranchAndPruneProposer:
    """Creates a branch then prunes one side on next depth."""

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        if depth == 0:
            return [
                CandidateAction(
                    action_type=ActionType.BRANCH,
                    title="case_split",
                    rationale="Binary split",
                    branch_labels=["yes", "no"],
                    outputs=["branched"],
                )
            ]

        if depth == 1 and state.branch_assignments:
            if state.branch_assignments[-1] == "no":
                return [
                    CandidateAction(
                        action_type=ActionType.PRUNE,
                        title="prune_no_case",
                        rationale="Inconsistent branch",
                        metadata={"prune_status": "contradiction"},
                    )
                ]
            return [
                CandidateAction(
                    action_type=ActionType.REWRITE,
                    title="advance_yes_case",
                    rationale="Continue search",
                    added_facts=["yes_branch_progress"],
                    outputs=["yes_nf"],
                )
            ]

        return []


class SolvedProposer:
    """Proposes one action that marks state as solved."""

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        _ = state
        if depth != 0:
            return []
        return [
            CandidateAction(
                action_type=ActionType.REWRITE,
                title="finish",
                rationale="Reach terminal solved state",
                added_facts=["goal_satisfied"],
                metadata={"mark_solved": True},
            )
        ]


class DuplicateProposer:
    """Produces two equivalent children that should deduplicate."""

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        _ = state
        if depth != 0:
            return []
        return [
            CandidateAction(
                action_type=ActionType.REWRITE,
                title="path_a",
                rationale="Equivalent transform A",
                outputs=["same_nf"],
                added_facts=["a"],
            ),
            CandidateAction(
                action_type=ActionType.SUBSTITUTE,
                title="path_b",
                rationale="Equivalent transform B",
                outputs=["same_nf"],
                added_facts=["b"],
            ),
        ]


class LoopingProposer:
    """Always proposes one non-terminal action to test max-depth stop."""

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        _ = state, depth
        return [
            CandidateAction(
                action_type=ActionType.EXPAND,
                title="expand_once",
                rationale="Keep progressing",
                added_facts=["tick"],
                outputs=["loop_nf"],
            )
        ]


class TestSearchControllerPhase3(unittest.TestCase):
    """Integration-style tests for the Phase 3 controller architecture."""

    def test_single_path_progression(self) -> None:
        result = run_search(
            create_initial_state("p", "proof"),
            SinglePathProposer(),
            verifier=FakeVerifier(),
            canonicalizer=FakeCanonicalizer(),
            config=ControllerConfig(max_depth=4, beam_width=2),
        )

        self.assertEqual(result.termination_reason, "no_valid_next_states")
        self.assertGreaterEqual(len(result.iteration_summaries), 1)
        first = result.iteration_summaries[0]
        self.assertEqual(first.candidate_actions, 2)
        self.assertEqual(first.accepted_actions, 1)
        self.assertEqual(result.best_state.derived_facts, ["f1"])

    def test_branch_and_prune(self) -> None:
        result = run_search(
            create_initial_state("p", "proof"),
            BranchAndPruneProposer(),
            verifier=FakeVerifier(),
            canonicalizer=FakeCanonicalizer(),
            config=ControllerConfig(max_depth=4, beam_width=4),
        )

        statuses = {state.status for state in result.final_beam}
        self.assertIn(StateStatus.CONTRADICTION, statuses)
        self.assertTrue(
            any(
                state.branch_assignments and state.branch_assignments[-1] == "yes"
                for state in result.final_beam
            )
        )

    def test_solved_state_found_stops_search(self) -> None:
        result = run_search(
            create_initial_state("p", "proof"),
            SolvedProposer(),
            verifier=FakeVerifier(),
            canonicalizer=FakeCanonicalizer(),
            config=ControllerConfig(max_depth=5, beam_width=2),
        )

        self.assertEqual(result.termination_reason, "high_priority_solved")
        self.assertEqual(result.final_beam[0].status, StateStatus.SOLVED)

    def test_duplicate_states_removed(self) -> None:
        result = run_search(
            create_initial_state("p", "proof"),
            DuplicateProposer(),
            verifier=FakeVerifier(),
            canonicalizer=FakeCanonicalizer(),
            config=ControllerConfig(max_depth=2, beam_width=5),
        )

        first = result.iteration_summaries[0]
        self.assertEqual(first.next_states, 2)
        self.assertEqual(first.kept_after_beam, 1)

    def test_max_depth_stop(self) -> None:
        result = run_search(
            create_initial_state("p", "proof"),
            LoopingProposer(),
            verifier=FakeVerifier(),
            canonicalizer=FakeCanonicalizer(),
            config=ControllerConfig(max_depth=1, beam_width=2),
        )

        self.assertEqual(result.termination_reason, "max_depth_reached")
        self.assertEqual(result.depth_reached, 1)


if __name__ == "__main__":
    unittest.main()
