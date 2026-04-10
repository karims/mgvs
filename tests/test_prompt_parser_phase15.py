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
            "entities": ["x"],
            "target": "solve x",
            "constraints": ["x+1=2", "x>=0"],
        }
        parsed = parse_pt_output(json.dumps(payload))

        self.assertIn("x", parsed.symbolic_objects)
        self.assertEqual(parsed.current_equations, [])
        self.assertEqual(parsed.domain_constraints, ["x+1=2", "x>=0"])
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
        self.assertEqual(parsed.answer_candidate, 50)

    def test_lss_candidate_action_parsing(self) -> None:
        payload = {
            "actions": [
                {
                    "action_type": "derive_constraint",
                    "title": "add_bound",
                    "added_facts": ["x in R"],
                    "added_constraints": ["x>=0"],
                }
            ]
        }
        parsed = parse_lss_output(json.dumps(payload))

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].action_type, ActionType.DERIVE_CONSTRAINT)
        self.assertEqual(parsed[0].title, "add_bound")
        self.assertEqual(parsed[0].added_constraints, ["x>=0"])
        self.assertEqual(parsed[0].rationale, "unspecified rationale")

    def test_lss_missing_optional_fields_default_safely(self) -> None:
        parsed = parse_lss_output(
            json.dumps({"actions": [{"action_type": "rewrite", "title": "restate_constraint"}]})
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].added_facts, [])
        self.assertEqual(parsed[0].added_constraints, [])
        self.assertEqual(parsed[0].rationale, "unspecified rationale")

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
        wrapped = "prefix text {\"actions\":[{\"action_type\":\"rewrite\",\"title\":\"t\"},{\"action_type\":\"substitute\",\"title\":\"s\"}]} suffix"
        parsed = parse_lss_output(wrapped)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].action_type, ActionType.REWRITE)
        self.assertEqual(parsed[1].rationale, "unspecified rationale")

    def test_lss_parser_caps_to_two_actions(self) -> None:
        payload = {
            "actions": [
                {"action_type": "rewrite", "title": "a"},
                {"action_type": "substitute", "title": "b"},
                {"action_type": "eliminate", "title": "c"},
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
                    "answer": 84,
                    "confidence": "high",
                    "justification": ["Reduced state forces a unique count."],
                }
            )
        )

        self.assertEqual(parsed.answer, 84)
        self.assertEqual(parsed.confidence, "high")
        self.assertEqual(parsed.justification, ["Reduced state forces a unique count."])

    def test_endgame_null_answer(self) -> None:
        parsed = parse_endgame_solve_output(
            json.dumps({"answer": None, "confidence": "medium", "justification": []})
        )

        self.assertIsNone(parsed.answer)
        self.assertEqual(parsed.confidence, "medium")
        self.assertEqual(parsed.justification, [])

    def test_endgame_defaults_missing_confidence_and_justification(self) -> None:
        parsed = parse_endgame_solve_output(json.dumps({"answer": 17}))

        self.assertEqual(parsed.answer, 17)
        self.assertEqual(parsed.confidence, "low")
        self.assertEqual(parsed.justification, [])

    def test_endgame_ignores_extra_keys(self) -> None:
        parsed = parse_endgame_solve_output(
            json.dumps(
                {
                    "answer": "91",
                    "confidence": "high",
                    "justification": ["Use the reduced bound."],
                    "essay": "ignore this",
                    "scratchwork": ["ignore this too"],
                }
            )
        )

        self.assertEqual(parsed.answer, 91)
        self.assertEqual(parsed.confidence, "high")
        self.assertEqual(parsed.justification, ["Use the reduced bound."])

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
        self.assertEqual(pct["constraints"]["max_tactics"], 3)
        self.assertEqual(
            sorted(pct["output_schema"].keys()),
            ["answer_candidate", "candidate_equations", "open_goals", "strategy_tags"],
        )
        self.assertEqual(len(pct["example_outputs"]), 2)
        self.assertIn("Do not solve the full problem in this stage.", pct["instructions"])
        self.assertIn("Do not provide bullet lists.", pct["instructions"])
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
        self.assertEqual(len(lss["example_outputs"]), 2)
        self.assertEqual(lss["example_outputs"][0]["actions"][0]["action_type"], "substitute")
        self.assertEqual(
            lss["example_outputs"][0]["actions"][0]["added_facts"],
            ["a + s_a = b + s_b + 10"],
        )
        self.assertEqual(lss["example_outputs"][1]["label"], "bad_example_do_not_copy")
        self.assertIn(
            "The action must be grounded in the current problem context, PT constraints, PT target, current equations, or known facts.",
            lss["instructions"],
        )
        self.assertIn(
            "Do not invent generic placeholder variables unless they already appear in the context.",
            lss["instructions"],
        )
        self.assertIn(
            "Do not output high-level tactics like eliminate variable or simplify without showing the result.",
            lss["instructions"],
        )
        self.assertIn(
            "Every action must introduce NEW information not already present in current_equations.",
            lss["instructions"],
        )
        self.assertIn(
            "If action_type is substitute or eliminate, the resulting simplified or derived relation MUST appear in added_facts.",
            lss["instructions"],
        )
        self.assertIn(
            "Do not copy or restate existing equations into added_facts or added_constraints.",
            lss["instructions"],
        )
        self.assertIn(
            "Prefer explicit derived equations such as simplified equations, substituted expressions, reduced forms, or new equalities between variables.",
            lss["instructions"],
        )
        self.assertIn("Do not repeat already-known equations or constraints.", lss["instructions"])
        self.assertIn(
            "Do not restate target-only facts such as the final modulus target without adding new information.",
            lss["instructions"],
        )
        self.assertIn("Do not propose actions that merely restate current equations.", lss["instructions"])
        self.assertIn("Do not propose empty eliminations with no downstream consequence.", lss["instructions"])
        self.assertIn(
            "Propose one action that introduces a genuinely new constraint, bound, counting relation, or case distinction tied to the current problem.",
            lss["instructions"],
        )
        self.assertIn('If no materially advancing action is available, return {"actions": []}.', lss["instructions"])
        self.assertEqual(lss["context"]["pt_entities"], ["x"])
        self.assertEqual(lss["context"]["pt_constraints"], ["x+1=2"])
        self.assertEqual(lss["context"]["pt_target"], "solve x")
        self.assertEqual(lss["context"]["strategy_tags"], ["isolate_variable"])
        self.assertEqual(lss["context"]["derived_facts"], ["x is scalar"])
        self.assertEqual(
            sorted(endgame["output_schema"].keys()),
            ["answer", "confidence", "justification"],
        )
        self.assertEqual(endgame["context"]["pt_target"], "solve x")
        self.assertEqual(endgame["context"]["pt_constraints"], ["x+1=2"])
        self.assertEqual(endgame["context"]["trace_summary"], ["rewrite: subtract 1"])
        self.assertIn("Use the reduced state as the primary input.", endgame["instructions"])
        self.assertIn(
            "Do not restart the whole problem from scratch unless necessary.",
            endgame["instructions"],
        )
        self.assertIn("Return 1 to 3 short justification strings at most.", endgame["instructions"])


if __name__ == "__main__":
    unittest.main()
