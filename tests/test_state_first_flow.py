"""Focused integration-style tests for the state-first sequential flow."""

from __future__ import annotations

import json
import unittest

from mgvs.llm.base import UnifiedLLMClient
from mgvs.solve.runner import SolveConfig, solve
from mgvs.types import StateStatus


class _StateFirstClient(UnifiedLLMClient):
    """Deterministic stage fixture for PT->PCT->LSS->endgame tests."""

    def __init__(self, *, endgame_payload: dict[str, object]) -> None:
        self._endgame_payload = endgame_payload

    def generate_pt(self, prompt: str) -> str:
        _ = prompt
        return json.dumps(
            {
                "objects": ["x"],
                "variables": ["x"],
                "relations": ["x + 1 = 2"],
                "constraints": ["x is integer"],
                "goal": "solve x",
                "unknowns_remaining": ["x"],
            }
        )

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
        return json.dumps(
            {
                "actions": [
                    {
                        "action_type": "derive_relation",
                        "title": "isolate_variable_symbolically",
                        "added_facts": ["0 <= x <= 2"],
                        "added_constraints": ["x is integer"],
                        "metadata": {"mark_solved": True},
                    }
                ]
            }
        )

    def generate_endgame(self, prompt: str) -> str:
        _ = prompt
        return json.dumps(self._endgame_payload)


class TestStateFirstFlow(unittest.TestCase):
    """Non-heavy state-first flow checks with deterministic stage fixtures."""

    def test_state_first_flow_accepts_transition_and_endgame_answer_when_ready(self) -> None:
        client = _StateFirstClient(
            endgame_payload={
                "ready": True,
                "answer": 1,
                "confidence": "high",
                "justification": ["State is reduced enough."],
                "missing_requirements": [],
            }
        )

        result = solve(
            "Solve x + 1 = 2",
            config=SolveConfig(experiment_state_first=True),
            client=client,
        )

        self.assertEqual(result.best_state.status, StateStatus.SOLVED)
        self.assertEqual(result.predicted_answer, 1)
        self.assertEqual(result.answer_source, "endgame_llm")
        self.assertIn("0 <= x <= 2", result.best_state.derived_facts)
        self.assertEqual(len(result.best_state.accepted_steps), 1)
        self.assertIn("endgame_called", result.policy_trace)
        self.assertIn("endgame_answer_found", result.policy_trace)

    def test_state_first_flow_drops_endgame_answer_when_not_ready(self) -> None:
        client = _StateFirstClient(
            endgame_payload={
                "ready": False,
                "answer": 99,
                "confidence": "high",
                "justification": ["Not enough reduction yet."],
                "missing_requirements": ["Need one more concrete derived relation."],
            }
        )

        result = solve(
            "Solve x + 1 = 2",
            config=SolveConfig(experiment_state_first=True),
            client=client,
        )

        self.assertEqual(result.best_state.status, StateStatus.SOLVED)
        self.assertIsNone(result.predicted_answer)
        self.assertEqual(result.answer_source, "")
        self.assertIn("endgame_called", result.policy_trace)
        self.assertIn("endgame_no_answer", result.policy_trace)


if __name__ == "__main__":
    unittest.main()
