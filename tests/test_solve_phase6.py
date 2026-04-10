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
        self.assertEqual(result.answer_source, "pct_answer_candidate")
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

    def test_endgame_can_set_predicted_answer_after_lss(self) -> None:
        class EndgameAnswerClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "entities": ["K"],
                        "target": "find K",
                        "constraints": ["distinct rectangles have distinct perimeters"],
                    }
                )

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "strategy_tags": ["counting"],
                        "open_goals": ["derive a final count"],
                        "candidate_equations": [],
                        "answer_candidate": None,
                    }
                )

            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "record_count_bound",
                                "added_facts": ["the perimeter count is tightly bounded"],
                                "added_constraints": [],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "answer": 50,
                        "confidence": "high",
                        "justification": ["Reduced state supports a unique final count."],
                    }
                )

        result = solve("Endgame answer demo", config=SolveConfig(), client=EndgameAnswerClient())

        self.assertEqual(result.predicted_answer, 50)
        self.assertEqual(result.answer_status, "predicted")
        self.assertEqual(result.answer_source, "endgame_llm")
        self.assertIn("endgame_called", result.policy_trace)
        self.assertIn("endgame_answer_found", result.policy_trace)
        self.assertIn("answer_source=endgame_llm", result.policy_trace)
        self.assertTrue(any("endgame_answer_found" in line for line in result.trace_summary))

    def test_endgame_null_answer_keeps_existing_flow(self) -> None:
        class EndgameNoAnswerClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "entities": ["K"],
                        "target": "find K",
                        "constraints": ["distinct rectangles have distinct perimeters"],
                    }
                )

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "strategy_tags": ["counting"],
                        "open_goals": ["derive a final count"],
                        "candidate_equations": [],
                        "answer_candidate": None,
                    }
                )

            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "record_count_bound",
                                "added_facts": ["the perimeter count is tightly bounded"],
                                "added_constraints": [],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps({"answer": None, "confidence": "low", "justification": []})

        result = solve("Endgame no answer demo", config=SolveConfig(), client=EndgameNoAnswerClient())

        self.assertIsNone(result.predicted_answer)
        self.assertEqual(result.answer_status, "missing_answer")
        self.assertEqual(result.answer_source, "")
        self.assertIn("endgame_called", result.policy_trace)
        self.assertIn("endgame_no_answer", result.policy_trace)
        self.assertNotIn("endgame_answer_found", result.policy_trace)

    def test_final_llm_can_set_answer_after_lss(self) -> None:
        class FinalLLMAnswerClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "entities": ["K"],
                        "target": "find K",
                        "constraints": ["distinct rectangles have distinct perimeters"],
                    }
                )

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "strategy_tags": ["counting"],
                        "open_goals": ["derive a final count"],
                        "candidate_equations": [],
                        "answer_candidate": None,
                    }
                )

            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "record_count_bound",
                                "added_facts": ["the perimeter count is tightly bounded"],
                                "added_constraints": [],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps({"answer": None, "confidence": "low", "justification": []})

            def generate(self, prompt: str, system_prompt: str = "") -> str:
                _ = prompt, system_prompt
                return json.dumps({"answer": 73})

        result = solve("Final LLM answer demo", config=SolveConfig(), client=FinalLLMAnswerClient())

        self.assertEqual(result.predicted_answer, 73)
        self.assertEqual(result.answer_status, "solved")
        self.assertEqual(result.answer_source, "final_llm")
        self.assertIn("final_llm_called", result.policy_trace)
        self.assertIn("final_llm_answer_found", result.policy_trace)
        self.assertIn("answer_source=final_llm", result.policy_trace)

    def test_final_llm_parse_failure_keeps_existing_behavior(self) -> None:
        class FinalLLMParseFailClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "entities": ["K"],
                        "target": "find K",
                        "constraints": ["distinct rectangles have distinct perimeters"],
                    }
                )

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "strategy_tags": ["counting"],
                        "open_goals": ["derive a final count"],
                        "candidate_equations": [],
                        "answer_candidate": None,
                    }
                )

            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "record_count_bound",
                                "added_facts": ["the perimeter count is tightly bounded"],
                                "added_constraints": [],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps({"answer": None, "confidence": "low", "justification": []})

            def generate(self, prompt: str, system_prompt: str = "") -> str:
                _ = prompt, system_prompt
                return "not json"

        result = solve("Final LLM parse fail demo", config=SolveConfig(), client=FinalLLMParseFailClient())

        self.assertIsNone(result.predicted_answer)
        self.assertEqual(result.answer_status, "missing_answer")
        self.assertEqual(result.answer_source, "")
        self.assertIn("final_llm_called", result.policy_trace)
        self.assertIn("final_llm_no_answer", result.policy_trace)
        self.assertNotIn("final_llm_answer_found", result.policy_trace)

    def test_cli_output_includes_answer_source(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["solve", "--problem", "Solve x + 1 = 2"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("answer source:", output)

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
