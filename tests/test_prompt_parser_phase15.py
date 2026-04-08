"""Phase 15 tests for compact prompting contracts and robust parser behavior."""

import json
import unittest

from mgvs.actions.models import ActionType
from mgvs.llm.parser import parse_lss_output, parse_pct_output, parse_pt_output
from mgvs.llm.prompts import build_lss_prompt, build_pct_prompt, build_pt_prompt
from mgvs.state.models import create_initial_state


class TestPromptParserPhase15(unittest.TestCase):
    """Covers revised PT/PCT/LSS contracts and parser resilience."""

    def test_pt_structured_parse(self) -> None:
        payload = {
            "unknowns": ["x"],
            "targets": ["solve x"],
            "facts": ["x is integer"],
            "constraints": {"domain": ["x>=0"], "global": ["sum fixed"]},
            "equations": ["x+1=2"],
            "symbolic_objects": {"x": {"kind": "scalar"}},
        }
        parsed = parse_pt_output(json.dumps(payload))

        self.assertEqual(parsed.current_equations, ["x+1=2"])
        self.assertEqual(parsed.domain_constraints, ["x>=0"])
        self.assertEqual(parsed.global_constraints, ["sum fixed"])
        self.assertEqual(parsed.open_goals, ["solve x"])
        self.assertEqual(parsed.derived_facts, ["x is integer"])

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
                    "type": "derive_constraint",
                    "name": "add bound",
                    "why": "keep domain consistent",
                    "facts": ["x in R"],
                    "constraints": ["x>=0"],
                }
            ]
        }
        parsed = parse_lss_output(json.dumps(payload))

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].action_type, ActionType.DERIVE_CONSTRAINT)
        self.assertEqual(parsed[0].title, "add bound")
        self.assertEqual(parsed[0].added_constraints, ["x>=0"])

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
        wrapped = "prefix text {\"actions\":[{\"action_type\":\"rewrite\",\"title\":\"t\",\"rationale\":\"r\"},{\"action_type\":\"substitute\",\"title\":\"s\"}]} suffix"
        parsed = parse_lss_output(wrapped)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].action_type, ActionType.REWRITE)
        self.assertEqual(parsed[1].rationale, "unspecified rationale")

    def test_prompts_are_compact_structured_contracts(self) -> None:
        state = create_initial_state("Solve x+1=2", "equation")
        pt = json.loads(build_pt_prompt(state.raw_problem, state.target_type))
        pct = json.loads(build_pct_prompt(state, max_tactics=3))
        lss = json.loads(build_lss_prompt(state, max_candidates=2))

        self.assertEqual(pt["contract"], "pt_v2")
        self.assertEqual(pct["contract"], "pct_v2")
        self.assertEqual(lss["contract"], "lss_v2")
        self.assertEqual(pct["constraints"]["max_tactics"], 3)
        self.assertEqual(
            sorted(pct["output_schema"].keys()),
            ["answer_candidate", "candidate_equations", "open_goals", "strategy_tags"],
        )
        self.assertIn("example_output", pct)
        self.assertEqual(lss["constraints"]["max_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
