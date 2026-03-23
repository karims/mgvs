"""Phase 2 tests for structured action models and application behavior."""

import unittest

from mgvs.actions.apply import apply_action
from mgvs.actions.models import ActionType, CandidateAction
from mgvs.state.models import create_initial_state
from mgvs.types import StateStatus


class TestActionApplicationPhase2(unittest.TestCase):
    """Covers deterministic action transitions for Phase 2."""

    def test_apply_single_action_updates_state(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        action = CandidateAction(
            action_type=ActionType.REWRITE,
            title="Rewrite expression",
            rationale="Normalize expression form",
            inputs=["eq1"],
            outputs=["eq1_rewritten"],
            added_facts=["f1"],
            added_constraints=["c1"],
            metadata={"normalized_form": "nf(eq1)", "mark_solved": True},
        )

        results = apply_action(state, action)

        self.assertEqual(len(results), 1)
        updated = results[0]
        self.assertEqual(updated.derived_facts, ["f1"])
        self.assertEqual(updated.domain_constraints, ["c1"])
        self.assertEqual(updated.normalized_form, "nf(eq1)")
        self.assertEqual(updated.status, StateStatus.SOLVED)
        self.assertEqual(updated.accepted_steps[-1].action, ActionType.REWRITE.value)
        self.assertGreater(updated.score, 0.0)
        self.assertEqual(state.derived_facts, [])

    def test_branch_returns_multiple_children(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        action = CandidateAction(
            action_type=ActionType.BRANCH,
            title="Case split",
            rationale="Split on binary condition",
            branch_labels=["case_true", "case_false"],
            added_facts=["split_performed"],
        )

        children = apply_action(state, action)

        self.assertEqual(len(children), 2)
        labels = {child.branch_assignments[-1] for child in children}
        self.assertEqual(labels, {"case_true", "case_false"})
        for child in children:
            self.assertEqual(child.accepted_steps[-1].action, ActionType.BRANCH.value)
            self.assertIn("split_performed", child.derived_facts)
            self.assertLess(child.score, 0.25)

    def test_prune_marks_contradiction(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        action = CandidateAction(
            action_type=ActionType.PRUNE,
            title="Prune infeasible branch",
            rationale="Constraint violation found",
            metadata={"prune_status": "contradiction"},
        )

        pruned = apply_action(state, action)[0]

        self.assertEqual(pruned.status, StateStatus.CONTRADICTION)
        self.assertEqual(pruned.accepted_steps[-1].action, ActionType.PRUNE.value)
        self.assertLess(pruned.score, 0.0)

    def test_prune_defaults_to_dead_end(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        action = CandidateAction(
            action_type=ActionType.PRUNE,
            title="Prune branch",
            rationale="No useful progress",
        )

        pruned = apply_action(state, action)[0]

        self.assertEqual(pruned.status, StateStatus.DEAD_END)
        self.assertLess(pruned.score, 0.0)


if __name__ == "__main__":
    unittest.main()
