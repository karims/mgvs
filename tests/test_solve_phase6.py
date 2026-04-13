"""Phase 6 tests for end-to-end local solve runner and CLI wiring."""

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from mgvs.cli.main import main
from mgvs.llm.base import UnifiedLLMClient
from mgvs.solve.runner import SolveConfig, is_endgame_ready, solve
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

    def test_phase1_trace_artifact_persists_pt_pct_lss_sections(self) -> None:
        class TraceClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return "Restatement:\n- Solve for x.\nWhat is given:\n- x + 1 = 2"

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return (
                    "Candidate approaches:\n- isolate variable\n"
                    "Possible intermediate lemmas:\n- x = 1\n"
                    "Useful reformulations:\n- x + 1 = 2"
                )

            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return (
                    "Candidate next steps:\n- subtract 1 from both sides to get x = 1\n"
                    "What each step would establish:\n- x = 1\n"
                    "Most promising immediate continuation:\n- verify substitution"
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(
                os.environ,
                {
                    "MGVS_PHASE1_TRACE": "1",
                    "MGVS_PHASE1_TRACE_DIR": str(Path(tmp_dir) / "traces"),
                },
                clear=False,
            ):
                solve("Trace persistence demo", config=SolveConfig(), client=TraceClient())

            files = sorted((Path(tmp_dir) / "traces").glob("*.md"))
            self.assertEqual(len(files), 1)
            text = files[0].read_text(encoding="utf-8")
            self.assertIn("## PROBLEM", text)
            self.assertIn("## PT RAW OUTPUT", text)
            self.assertIn("## PCT RAW OUTPUT", text)
            self.assertIn("## LSS RAW OUTPUT", text)
            self.assertIn("## EXTRACTED CANDIDATE MOVES", text)
            self.assertIn("## BRANCH EXPANSIONS", text)
            self.assertIn("## BRANCH SCORES", text)
            self.assertIn("## PRUNING DECISIONS", text)
            self.assertIn("## BEST BRANCH", text)
            self.assertIn("## FINAL SYNTHESIS", text)
            self.assertIn("## FINAL OUTPUT / FINAL ANSWER", text)
            self.assertIn("x + 1 = 2", text)

    def test_exploratory_search_records_branch_trace_and_synthesis_marker(self) -> None:
        class ExploratoryClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return (
                    "Restatement:\n1. Solve for x.\n"
                    "What is given:\n1. x + 1 = 2\n"
                    "What must be found or proved:\n1. Determine x"
                )

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return (
                    "Candidate approaches:\n1. Isolate x directly.\n"
                    "Why each approach might help:\n1. Removes one unknown.\n"
                    "Possible intermediate lemmas:\n1. x = 1\n"
                    "Useful reformulations:\n1. x + 1 = 2"
                )

            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return (
                    "Candidate next steps:\n1. Subtract 1 from both sides.\n"
                    "Why each step helps:\n1. Isolates x.\n"
                    "What each step would establish:\n1. x = 1\n"
                    "Most promising immediate continuation:\n1. Verify by substitution."
                )

        with patch.dict(os.environ, {"MGVS_PHASE1_TRACE": "1"}, clear=False):
            result = solve(
                "Exploratory solve demo",
                config=SolveConfig(exploratory_search=True),
                client=ExploratoryClient(),
            )

        self.assertTrue(any(item.startswith("exploratory_search ") for item in result.policy_trace))
        self.assertIn("phase6_synthesis_available", result.trace_summary)

    def test_exploratory_knobs_load_from_env_with_conservative_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MGVS_EXPLORATORY_SEARCH": "1",
                "MGVS_MAX_INITIAL_CANDIDATE_MOVES": "3",
                "MGVS_MAX_BRANCHES_TO_EXPAND": "2",
                "MGVS_MAX_SEARCH_DEPTH": "2",
                "MGVS_ENABLE_VERBOSE_TRACE": "1",
                "MGVS_ENABLE_PHASE6_SYNTHESIS": "1",
            },
            clear=False,
        ):
            cfg = SolveConfig.from_env(target_type="proof")
        self.assertTrue(cfg.exploratory_search)
        self.assertEqual(cfg.max_initial_candidate_moves, 3)
        self.assertEqual(cfg.max_branches_to_expand, 2)
        self.assertEqual(cfg.max_search_depth, 2)
        self.assertTrue(cfg.enable_verbose_trace)
        self.assertTrue(cfg.enable_phase6_synthesis)

    def test_exploratory_lss_vague_steps_are_filtered(self) -> None:
        class VagueLSSClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return "What is given:\n1. x + 1 = 2\nWhat must be found or proved:\n1. Find x."

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return "Candidate approaches:\n1. isolate variable."

            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return (
                    "Candidate next steps:\n1. derive a relation\n2. solve the equations\n"
                    "Why each step helps:\n1. proceed\n2. proceed"
                )

        result = solve(
            "Vague LSS filtering demo",
            config=SolveConfig(exploratory_search=True, enable_phase6_synthesis=False),
            client=VagueLSSClient(),
        )
        self.assertTrue(any(item.startswith("exploratory_search ") for item in result.policy_trace))

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
                                "title": "derive_explicit_perimeter_count_bound",
                                "added_facts": ["Possible rectangle perimeters range from 4 to 2000 and are even."],
                                "added_constraints": ["The number of distinct perimeters is at most 999."],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "ready": True,
                        "answer": 50,
                        "confidence": "high",
                        "justification": ["Reduced state supports a unique final count."],
                        "missing_requirements": [],
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
                                "title": "derive_explicit_perimeter_count_bound",
                                "added_facts": ["Possible rectangle perimeters range from 4 to 2000 and are even."],
                                "added_constraints": ["The number of distinct perimeters is at most 999."],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "ready": False,
                        "answer": None,
                        "confidence": "low",
                        "justification": [],
                        "missing_requirements": ["need one more reduction"],
                    }
                )

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
                                "title": "derive_explicit_perimeter_count_bound",
                                "added_facts": ["Possible rectangle perimeters range from 4 to 2000 and are even."],
                                "added_constraints": ["The number of distinct perimeters is at most 999."],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "ready": False,
                        "answer": None,
                        "confidence": "low",
                        "justification": [],
                        "missing_requirements": ["need one more reduction"],
                    }
                )

            def generate(self, prompt: str, system_prompt: str = "") -> str:
                _ = prompt, system_prompt
                return json.dumps(
                    {
                        "ready": True,
                        "answer": 73,
                        "confidence": "high",
                        "justification": [],
                        "missing_requirements": [],
                    }
                )

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
                                "title": "derive_explicit_perimeter_count_bound",
                                "added_facts": ["Possible rectangle perimeters range from 4 to 2000 and are even."],
                                "added_constraints": ["The number of distinct perimeters is at most 999."],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "ready": False,
                        "answer": None,
                        "confidence": "low",
                        "justification": [],
                        "missing_requirements": ["need one more reduction"],
                    }
                )

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

    def test_endgame_is_blocked_for_weak_qualitative_state(self) -> None:
        class WeakStateClient(UnifiedLLMClient):
            def __init__(self) -> None:
                self.endgame_calls = 0
                self.final_calls = 0

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
                                "title": "weak_bound_statement",
                                "added_facts": ["the perimeter count is tightly bounded"],
                                "added_constraints": [],
                                "metadata": {"mark_solved": True},
                            }
                        ]
                    }
                )

            def generate_endgame(self, prompt: str) -> str:
                self.endgame_calls += 1
                _ = prompt
                return json.dumps(
                    {
                        "ready": True,
                        "answer": 50,
                        "confidence": "high",
                        "justification": [],
                        "missing_requirements": [],
                    }
                )

            def generate(self, prompt: str, system_prompt: str = "") -> str:
                self.final_calls += 1
                _ = prompt, system_prompt
                return json.dumps(
                    {
                        "ready": True,
                        "answer": 50,
                        "confidence": "high",
                        "justification": [],
                        "missing_requirements": [],
                    }
                )

        client = WeakStateClient()
        result = solve("Weak endgame gate demo", config=SolveConfig(), client=client)

        self.assertEqual(client.endgame_calls, 0)
        self.assertEqual(client.final_calls, 0)
        self.assertIsNone(result.predicted_answer)
        self.assertNotIn("endgame_called", result.policy_trace)
        self.assertNotIn("final_llm_called", result.policy_trace)

    def test_endgame_ready_with_explicit_bound_even_without_steps(self) -> None:
        from mgvs.state.models import create_initial_state

        state = create_initial_state("Rectangle count problem", "proof")
        state.domain_constraints.append("The number of distinct perimeters is at most 999.")

        self.assertTrue(is_endgame_ready(state))

    def test_endgame_not_ready_for_empty_and_vague_state(self) -> None:
        from mgvs.state.models import create_initial_state

        state = create_initial_state("Rectangle count problem", "proof")
        state.derived_facts.append("the perimeter count is tightly bounded")

        self.assertFalse(is_endgame_ready(state))

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

    def test_experiment_state_first_mode_emits_stage_summaries(self) -> None:
        class _NoActionClient(UnifiedLLMClient):
            def generate_pt(self, prompt: str) -> str:
                _ = prompt
                return json.dumps({"objects": ["x"], "relations": ["x+1=2"], "constraints": ["x+1=2"], "goal": "solve x"})

            def generate_pct(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "strategy_tags": ["normalize_equations"],
                        "open_goals": ["isolate x"],
                        "candidate_equations": [],
                        "answer_candidate": None,
                    }
                )

            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps({"actions": []})

            def generate_endgame(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "ready": False,
                        "answer": None,
                        "confidence": "low",
                        "justification": [],
                        "missing_requirements": ["Need one concrete reduction step."],
                    }
                )

        result = solve(
            "Experiment mode demo",
            config=SolveConfig(experiment_state_first=True),
            client=_NoActionClient(),
        )

        self.assertFalse(result.fallback_used)
        self.assertTrue(any(item.startswith("pt_quality=") for item in result.policy_trace))
        self.assertTrue(any(item.startswith("pct_additions=") for item in result.policy_trace))
        self.assertTrue(any(item.startswith("lss.stop_reason=") for item in result.policy_trace))
        self.assertTrue(any(item.startswith("endgame.readiness_score=") for item in result.policy_trace))


if __name__ == "__main__":
    unittest.main()
