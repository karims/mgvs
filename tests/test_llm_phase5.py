"""Phase 5 tests for PT/PCT/LSS abstraction and stub pipeline."""

import unittest

from mgvs.llm.parser import parse_lss_output
from mgvs.llm.stub import StubLLMClient, StubPipelineProposer, run_pct_stage, run_pt_stage
from mgvs.search.controller import ControllerConfig, run_search
from mgvs.state.models import create_initial_state
from mgvs.types import StateStatus


class TestLLMPhase5(unittest.TestCase):
    """Covers deterministic PT/PCT/LSS behavior with stub backend."""

    def setUp(self) -> None:
        self.client = StubLLMClient()

    def test_pt_can_initialize_state(self) -> None:
        state = create_initial_state(raw_problem="Solve x + 1 = 2", target_type="equation")

        updated = run_pt_stage(state, self.client)

        self.assertIn("x", updated.symbolic_objects)
        self.assertIn("x + 1 = 2", updated.current_equations)
        self.assertIn("solve for x", updated.open_goals)

    def test_pct_adds_strategy_tags_and_goals(self) -> None:
        state = create_initial_state(raw_problem="Solve x + 1 = 2", target_type="equation")
        pt_state = run_pt_stage(state, self.client)

        updated = run_pct_stage(pt_state, self.client)

        self.assertIn("isolate_variable", updated.strategy_tags)
        self.assertIn("isolate x", updated.open_goals)
        self.assertIn("equation is linear", updated.derived_facts)

    def test_lss_actions_drive_controller(self) -> None:
        state = create_initial_state(raw_problem="Solve x + 1 = 2", target_type="equation")
        state = run_pct_stage(run_pt_stage(state, self.client), self.client)

        proposer = StubPipelineProposer(self.client, max_candidates=2)
        result = run_search(
            state,
            proposer,
            config=ControllerConfig(max_depth=3, beam_width=2),
        )

        self.assertEqual(result.termination_reason, "high_priority_solved")
        self.assertTrue(any(s.status == StateStatus.SOLVED for s in result.final_beam))

    def test_lss_parser_returns_candidate_actions(self) -> None:
        prompt_state = create_initial_state(raw_problem="Solve x + 1 = 2", target_type="equation")
        prompt_state = run_pct_stage(run_pt_stage(prompt_state, self.client), self.client)
        raw = self.client.generate_lss(
            '{"task":"local_step_synthesis","context":{"open_goals":["isolate x"]}}'
        )

        actions = parse_lss_output(raw)

        self.assertGreaterEqual(len(actions), 1)
        self.assertEqual(actions[0].title, "subtract_one_both_sides")


if __name__ == "__main__":
    unittest.main()
