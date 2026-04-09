"""Phase 6 tests for end-to-end local solve runner and CLI wiring."""

import io
import json
import unittest
from contextlib import redirect_stdout

from mgvs.cli.main import main
from mgvs.llm.base import UnifiedLLMClient
from mgvs.solve.runner import SolveConfig, solve
from mgvs.types import StateStatus


class TestSolvePhase6(unittest.TestCase):
    """Validates architecture-in-motion using deterministic stub pipeline."""

    def test_solve_happy_path(self) -> None:
        result = solve("Solve x + 1 = 2", config=SolveConfig(target_type="equation"))

        self.assertEqual(result.best_state.status, StateStatus.SOLVED)
        self.assertIn("x = 1", result.best_state.derived_facts)
        self.assertTrue(any("subtract_one_both_sides" in line for line in result.trace_summary))

    def test_solve_branch_path(self) -> None:
        result = solve("Branch demo: analyze two sign cases", config=SolveConfig())

        self.assertEqual(result.best_state.status, StateStatus.SOLVED)
        self.assertTrue(any("split_positive_negative" in line for line in result.trace_summary))

    def test_solve_contradiction_path(self) -> None:
        result = solve("Contradiction demo: inconsistent constraints", config=SolveConfig())

        self.assertEqual(result.best_state.status, StateStatus.CONTRADICTION)
        self.assertTrue(any("prune_contradiction" in line for line in result.trace_summary))

    def test_solve_parametric_path(self) -> None:
        result = solve("Parametric demo: represent family of solutions", config=SolveConfig())

        self.assertEqual(result.best_state.status, StateStatus.PARAMETRIC)
        self.assertTrue(any("introduce_parameterized_witness" in line for line in result.trace_summary))

    def test_pct_answer_candidate_can_short_circuit_before_lss(self) -> None:
        class PCTAnswerClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return json.dumps({"open_goals": ["solve"]})

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "strategy_tags": ["direct_answer"],
                        "open_goals": [],
                        "candidate_equations": [],
                        "answer_candidate": 50,
                    }
                )

            def generate_lss(self, prompt: str) -> str:
                raise AssertionError("LSS should not run when PCT answer candidate is accepted")

        result = solve("Direct answer demo", config=SolveConfig(), client=PCTAnswerClient())

        self.assertEqual(result.best_state.status, StateStatus.SOLVED)
        self.assertEqual(result.predicted_answer, 50)
        self.assertIn("pct_answer_candidate_detected", result.policy_trace)
        self.assertIn("pct_answer_candidate_accepted", result.policy_trace)
        self.assertTrue(any("pct_answer_candidate_accepted" in line for line in result.trace_summary))

    def test_pct_answer_candidate_rejection_falls_through_to_lss(self) -> None:
        lss_calls: list[str] = []

        class RejectedPCTAnswerClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return json.dumps({"current_equations": ["x = 7"], "open_goals": ["solve"]})

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "strategy_tags": ["direct_answer"],
                        "open_goals": ["confirm x"],
                        "candidate_equations": [],
                        "answer_candidate": 50,
                    }
                )

            def generate_lss(self, prompt: str) -> str:
                lss_calls.append(prompt)
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "rewrite",
                                "title": "confirm_x",
                                "added_facts": ["x = 7"],
                                "added_constraints": [],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

        result = solve("Rejected answer candidate demo", config=SolveConfig(), client=RejectedPCTAnswerClient())

        self.assertIn("pct_answer_candidate_detected", result.policy_trace)
        self.assertIn("pct_answer_candidate_rejected", result.policy_trace)
        self.assertEqual(len(lss_calls), 1)
        self.assertNotEqual(result.termination_reason, "pct_answer_candidate_accepted")

    def test_no_pct_answer_candidate_uses_normal_path(self) -> None:
        result = solve("Solve x + 1 = 2", config=SolveConfig(target_type="equation"))

        self.assertEqual(result.best_state.status, StateStatus.SOLVED)
        self.assertNotIn("pct_answer_candidate_detected", result.policy_trace)
        self.assertNotIn("pct_answer_candidate_accepted", result.policy_trace)
        self.assertNotIn("pct_answer_candidate_rejected", result.policy_trace)
        self.assertNotEqual(result.termination_reason, "pct_answer_candidate_accepted")

    def test_cli_solve_command_outputs_summary(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["solve", "--problem", "Solve x + 1 = 2"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("final status: solved", output)
        self.assertIn("accepted trace:", output)


if __name__ == "__main__":
    unittest.main()
