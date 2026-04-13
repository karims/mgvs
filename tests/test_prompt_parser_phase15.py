"""Phase 15 tests for compact prompting contracts and robust parser behavior."""

import json
import unittest

from mgvs.actions.models import ActionType
from mgvs.llm.parser import (
    parse_endgame_solve_output,
    parse_lss_output,
    parse_pct_output,
    parse_pt_output,
)
from mgvs.llm.prompts import (
    build_endgame_solve_prompt,
    build_lss_prompt,
    build_pct_prompt,
    build_pt_prompt,
)
from mgvs.state.models import create_initial_state


class TestPromptParserPhase15(unittest.TestCase):
    """Covers revised PT/PCT/LSS contracts and parser resilience."""

    def test_pt_structured_parse(self) -> None:
        payload = {
            "objects": ["equation"],
            "variables": ["x"],
            "domains": ["x is an integer"],
            "relations": ["x+1=2"],
            "constraints": ["x+1=2", "x>=0"],
            "goal": "solve x",
            "unknowns_remaining": ["x"],
        }
        parsed = parse_pt_output(json.dumps(payload))

        self.assertIn("x", parsed.symbolic_objects)
        self.assertIn("equation", parsed.symbolic_objects)
        self.assertEqual(parsed.current_equations, ["x+1=2"])
        self.assertEqual(parsed.domain_constraints, ["x is an integer", "x+1=2", "x>=0"])
        self.assertEqual(parsed.global_constraints, [])
        self.assertEqual(parsed.open_goals, ["solve x"])
        self.assertEqual(parsed.derived_facts, [])

    def test_pt_defaults_missing_fields(self) -> None:
        parsed = parse_pt_output("{}")

        self.assertEqual(parsed.symbolic_objects, {})
        self.assertEqual(parsed.current_equations, [])
        self.assertEqual(parsed.domain_constraints, [])
        self.assertEqual(parsed.global_constraints, [])
        self.assertEqual(parsed.open_goals, [])

    def test_pt_text_extractor_recovers_fields_from_sectioned_trace(self) -> None:
        parsed = parse_pt_output(
            "\n".join(
                [
                    "Restatement:",
                    "1. Determine x.",
                    "What is given:",
                    "1. x + 1 = 2",
                    "2. x is an integer.",
                    "What must be found or proved:",
                    "1. Find x.",
                    "Key mathematical structure:",
                    "1. Linear equation in one variable.",
                    "Plausible first directions:",
                    "1. Subtract 1 from both sides.",
                ]
            )
        )

        self.assertIn("x + 1 = 2", parsed.current_equations)
        self.assertTrue(parsed.open_goals)
        self.assertTrue(parsed.symbolic_objects)

    def test_pct_tactic_extraction(self) -> None:
        payload = {
            "strategy_tags": ["eliminate", "substitute"],
            "open_goals": ["remove y", "finish solve"],
            "candidate_equations": ["x+y=10"],
            "answer_candidate": 50,
        }
        parsed = parse_pct_output(json.dumps(payload))

        self.assertIn("eliminate", parsed.strategy_tags)
        self.assertIn("substitute", parsed.strategy_tags)
        self.assertIn("remove y", parsed.open_goals)
        self.assertIn("finish solve", parsed.open_goals)
        self.assertEqual(parsed.candidate_equations, ["x+y=10"])
        self.assertEqual(parsed.answer_candidate, 50)

    def test_pct_defaults_missing_optional_fields(self) -> None:
        parsed = parse_pct_output(json.dumps({"strategy_tags": ["counting"]}))

        self.assertEqual(parsed.strategy_tags, ["counting"])
        self.assertEqual(parsed.open_goals, [])
        self.assertEqual(parsed.candidate_equations, [])
        self.assertIsNone(parsed.answer_candidate)

    def test_pct_ignores_extra_keys(self) -> None:
        parsed = parse_pct_output(
            json.dumps(
                {
                    "strategy_tags": ["modular"],
                    "open_goals": ["check residues"],
                    "candidate_equations": [],
                    "answer_candidate": "17",
                    "essay": "long derivation that should be ignored",
                    "proof": ["ignore this too"],
                }
            )
        )

        self.assertEqual(parsed.strategy_tags, ["modular"])
        self.assertEqual(parsed.open_goals, ["check residues"])
        self.assertEqual(parsed.candidate_equations, [])
        self.assertEqual(parsed.answer_candidate, 17)

    def test_pct_answer_fallback_from_non_json_text(self) -> None:
        parsed = parse_pct_output("We can solve the system directly. **Answer: 50**")

        self.assertEqual(parsed.strategy_tags, [])
        self.assertEqual(parsed.open_goals, [])
        self.assertEqual(parsed.candidate_equations, [])
        self.assertIsNone(parsed.answer_candidate)

    def test_lss_candidate_action_parsing(self) -> None:
        payload = {
            "actions": [
                {
                    "action_type": "tighten_bound",
                    "title": "add_numeric_bound",
                    "added_facts": ["x <= 10"],
                    "added_constraints": ["x>=0"],
                }
            ]
        }
        parsed = parse_lss_output(json.dumps(payload))

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].action_type, ActionType.TIGHTEN_BOUND)
        self.assertEqual(parsed[0].title, "add_numeric_bound")
        self.assertEqual(parsed[0].added_constraints, ["x>=0"])
        self.assertEqual(parsed[0].rationale, "unspecified rationale")

    def test_lss_missing_optional_fields_default_safely(self) -> None:
        parsed = parse_lss_output(
            json.dumps({"actions": [{"action_type": "rewrite", "title": "restate_constraint"}]})
        )

        self.assertEqual(parsed, [])

    def test_lss_rejects_empty_transition(self) -> None:
        parsed = parse_lss_output(
            json.dumps({"actions": [{"action_type": "rewrite", "title": "empty_step", "added_facts": [], "added_constraints": []}]})
        )

        self.assertEqual(parsed, [])

    def test_lss_rejects_generic_advice(self) -> None:
        parsed = parse_lss_output(
            json.dumps(
                {
                    "actions": [
                        {
                            "action_type": "derive_relation",
                            "title": "consider symmetry",
                            "added_facts": ["Try simplifying the expression."],
                            "added_constraints": [],
                        }
                    ]
                }
            )
        )

        self.assertEqual(parsed, [])

    def test_malformed_response_handling(self) -> None:
        payload = {
            "actions": [
                {"action_type": "unknown", "title": "bad", "rationale": "bad"},
                {"action_type": "rewrite", "rationale": "missing title"},
            ]
        }
        parsed = parse_lss_output(json.dumps(payload))
        self.assertEqual(parsed, [])

    def test_partial_parse_fallback(self) -> None:
        wrapped = (
            "prefix text "
            "{\"actions\":["
            "{\"action_type\":\"rewrite\",\"title\":\"t\",\"added_facts\":[\"x=1\"]},"
            "{\"action_type\":\"substitute\",\"title\":\"s\",\"added_constraints\":[\"x>0\"]}"
            "]} suffix"
        )
        parsed = parse_lss_output(wrapped)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].action_type, ActionType.REWRITE)
        self.assertEqual(parsed[1].rationale, "unspecified rationale")

    def test_lss_parser_caps_to_two_actions(self) -> None:
        payload = {
            "actions": [
                {"action_type": "rewrite", "title": "a", "added_facts": ["x=1"]},
                {"action_type": "substitute", "title": "b", "added_constraints": ["x>0"]},
                {"action_type": "eliminate", "title": "c", "added_facts": ["y=2"]},
            ]
        }
        parsed = parse_lss_output(json.dumps(payload))

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].title, "a")
        self.assertEqual(parsed[1].title, "b")

    def test_endgame_valid_json_answer(self) -> None:
        parsed = parse_endgame_solve_output(
            json.dumps(
                {
                    "ready": True,
                    "answer": 84,
                    "confidence": "high",
                    "justification": ["Reduced state forces a unique count."],
                    "missing_requirements": [],
                }
            )
        )

        self.assertEqual(parsed.answer, 84)
        self.assertTrue(parsed.ready)
        self.assertEqual(parsed.confidence, "high")
        self.assertEqual(parsed.justification, ["Reduced state forces a unique count."])

    def test_endgame_null_answer(self) -> None:
        parsed = parse_endgame_solve_output(
            json.dumps(
                {
                    "ready": False,
                    "answer": None,
                    "confidence": "medium",
                    "justification": [],
                    "missing_requirements": ["need one more bound"],
                }
            )
        )

        self.assertIsNone(parsed.answer)
        self.assertFalse(parsed.ready)
        self.assertEqual(parsed.confidence, "medium")
        self.assertEqual(parsed.justification, [])

    def test_endgame_missing_ready_rejected(self) -> None:
        parsed = parse_endgame_solve_output(json.dumps({"answer": 17}))

        self.assertIsNone(parsed.answer)
        self.assertFalse(parsed.ready)
        self.assertEqual(parsed.confidence, "low")
        self.assertEqual(parsed.justification, [])

    def test_endgame_ignores_extra_keys(self) -> None:
        parsed = parse_endgame_solve_output(
            json.dumps(
                {
                    "ready": True,
                    "answer": "91",
                    "confidence": "high",
                    "justification": ["Use the reduced bound."],
                    "missing_requirements": [],
                    "essay": "ignore this",
                    "scratchwork": ["ignore this too"],
                }
            )
        )

        self.assertEqual(parsed.answer, 91)
        self.assertTrue(parsed.ready)
        self.assertEqual(parsed.confidence, "high")
        self.assertEqual(parsed.justification, ["Use the reduced bound."])

    def test_endgame_answer_rejected_when_not_ready_or_missing_requirements(self) -> None:
        parsed = parse_endgame_solve_output(
            json.dumps(
                {
                    "ready": False,
                    "answer": 12,
                    "confidence": "high",
                    "justification": ["bad"],
                    "missing_requirements": ["need reduction"],
                }
            )
        )

        self.assertIsNone(parsed.answer)
        self.assertFalse(parsed.ready)
        self.assertEqual(parsed.missing_requirements, ["need reduction"])

    def test_prompts_are_compact_structured_contracts(self) -> None:
        state = create_initial_state("Solve x+1=2", "equation")
        state.symbolic_objects["x"] = {"kind": "entity"}
        state.domain_constraints.append("x+1=2")
        state.current_equations.append("x+1=2")
        state.open_goals.append("solve x")
        state.strategy_tags.append("isolate_variable")
        state.derived_facts.append("x is scalar")
        pt = json.loads(build_pt_prompt(state.raw_problem, state.target_type))
        pct = json.loads(build_pct_prompt(state, max_tactics=3))
        lss = json.loads(build_lss_prompt(state, max_candidates=2))
        endgame = json.loads(
            build_endgame_solve_prompt(
                raw_problem=state.raw_problem,
                pt_target="solve x",
                pt_constraints=["x+1=2"],
                current_equations=["x+1=2"],
                derived_facts=["x is scalar"],
                open_goals=["solve x"],
                strategy_tags=["isolate_variable"],
                trace_summary=["rewrite: subtract 1"],
            )
        )

        self.assertEqual(pt["contract"], "pt_v2")
        self.assertEqual(pct["contract"], "pct_v2")
        self.assertEqual(lss["contract"], "lss_v2")
        self.assertEqual(endgame["contract"], "endgame_v1")
        self.assertEqual(
            sorted(pt["output_schema"].keys()),
            ["constraints", "domains", "goal", "objects", "relations", "unknowns_remaining", "variables"],
        )
        self.assertEqual(pt["example_output"]["goal"], "determine x")
        self.assertIn("Do not solve the problem.", pt["instructions"])
        self.assertIn("Extract only machine-usable base state.", pt["instructions"])
        self.assertEqual(pct["constraints"]["max_tactics"], 3)
        self.assertEqual(
            sorted(pct["output_schema"].keys()),
            ["answer_candidate", "candidate_equations", "open_goals", "strategy_tags"],
        )
        self.assertEqual(len(pct["example_outputs"]), 1)
        self.assertEqual(
            pct["example_outputs"][0]["candidate_equations"],
            ["p = 2(a + b)", "4 <= p <= 2000"],
        )
        self.assertEqual(
            pct["example_outputs"][0]["strategy_tags"],
            ["canonical_variables", "bound_search_space"],
        )
        self.assertIn("Do not solve the full problem in this stage.", pct["instructions"])
        self.assertIn(
            "Strengthen the state, do not narrate strategy.",
            pct["instructions"],
        )
        self.assertIn(
            "strategy_tags must be concrete and operational.",
            pct["instructions"],
        )
        self.assertIn(
            "open_goals must describe specific state improvements, not vague plans.",
            pct["instructions"],
        )
        self.assertIn(
            "candidate_equations may contain canonical variable definitions, justified bounds, invariants, or exact case relations supported by current state.",
            pct["instructions"],
        )
        self.assertIn(
            "Do not invent unsupported transformed equations or hidden assumptions.",
            pct["instructions"],
        )
        self.assertEqual(pct["context"]["pt_entities"], ["x"])
        self.assertEqual(pct["context"]["pt_constraints"], ["x+1=2"])
        self.assertEqual(pct["context"]["pt_target"], "solve x")
        self.assertEqual(pct["context"]["current_equations"], ["x+1=2"])
        self.assertEqual(lss["constraints"]["max_candidates"], 1)
        self.assertEqual(lss["constraints"]["preferred_candidates"], 1)
        self.assertEqual(
            sorted(lss["output_schema"]["actions"][0].keys()),
            ["action_type", "added_constraints", "added_facts", "title"],
        )
        self.assertEqual(len(lss["example_outputs"]), 3)
        self.assertEqual(lss["example_outputs"][0]["actions"][0]["action_type"], "derive_relation")
        self.assertEqual(
            lss["example_outputs"][0]["actions"][0]["added_facts"],
            ["p = 2(a + b)"],
        )
        self.assertEqual(
            lss["example_outputs"][1]["actions"][0]["title"],
            "derive_explicit_perimeter_count_bound",
        )
        self.assertEqual(
            lss["example_outputs"][1]["actions"][0]["added_constraints"],
            ["The number of distinct perimeters is at most 999."],
        )
        self.assertEqual(lss["example_outputs"][2]["label"], "bad_example")
        self.assertIn(
            "The action must be atomic and machine-usable.",
            lss["instructions"],
        )
        self.assertIn(
            "Choose exactly one action_type from the allowed set.",
            lss["instructions"],
        )
        self.assertIn(
            "added_facts and added_constraints must contain only new items, not explanations.",
            lss["instructions"],
        )
        self.assertIn(
            "Do not restate the current state, target, or existing equations.",
            lss["instructions"],
        )
        self.assertIn(
            "Do not add vague prose, unjustified assumptions, or weak observations.",
            lss["instructions"],
        )
        self.assertIn(
            "Prefer explicit new relations, explicit bounds, explicit invariants, or explicit finite case splits.",
            lss["instructions"],
        )
        self.assertIn(
            'If no meaningful step exists, return {"actions": []}.',
            lss["instructions"],
        )
        self.assertEqual(lss["context"]["pt_entities"], ["x"])
        self.assertEqual(lss["context"]["pt_constraints"], ["x+1=2"])
        self.assertEqual(lss["context"]["pt_target"], "solve x")
        self.assertEqual(lss["context"]["strategy_tags"], ["isolate_variable"])
        self.assertEqual(lss["context"]["derived_facts"], ["x is scalar"])
        self.assertEqual(
            sorted(endgame["output_schema"].keys()),
            ["answer", "confidence", "justification", "missing_requirements", "ready"],
        )
        self.assertEqual(len(endgame["example_outputs"]), 2)
        self.assertIsNone(endgame["example_outputs"][1]["answer"])
        self.assertFalse(endgame["example_outputs"][1]["ready"])
        self.assertEqual(endgame["example_outputs"][1]["confidence"], "low")
        self.assertEqual(
            endgame["example_outputs"][1]["justification"],
            ["The reduced state does not yet determine a unique integer."],
        )
        self.assertEqual(endgame["context"]["pt_target"], "solve x")
        self.assertEqual(endgame["context"]["pt_constraints"], ["x+1=2"])
        self.assertEqual(endgame["context"]["trace_summary"], ["rewrite: subtract 1"])
        self.assertIn("Use the reduced state as the primary input.", endgame["instructions"])
        self.assertIn(
            'If the state is insufficient, return {"ready": false, "answer": null, ...}.',
            endgame["instructions"],
        )
        self.assertIn("Answer only from accumulated state.", endgame["instructions"])
        self.assertIn("Do not guess.", endgame["instructions"])
        self.assertIn(
            "Do not output a confident integer answer from vague qualitative bounds alone.",
            endgame["instructions"],
        )
        self.assertIn(
            "Only return a numeric answer when the equations, facts, and constraints are sufficient to justify a unique integer.",
            endgame["instructions"],
        )

    def test_lss_context_filters_strong_domain_tags_when_fallback_only(self) -> None:
        state = create_initial_state("Find n", "integer")
        state.strategy_tags = ["llm_fallback", "modular_casework", "parity_check", "safe_tag"]
        payload = json.loads(build_lss_prompt(state, max_candidates=2))
        tags = payload["context"]["strategy_tags"]
        self.assertIn("llm_fallback", tags)
        self.assertIn("safe_tag", tags)
        self.assertNotIn("modular_casework", tags)
        self.assertNotIn("parity_check", tags)


if __name__ == "__main__":
    unittest.main()
