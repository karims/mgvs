"""Phase 12 tests for runtime budgets, caching, and hardened inference behavior."""

import json
import unittest
from unittest.mock import patch

from mgvs.config import VLLMRuntimeConfig
from mgvs.llm.prompts import build_lss_prompt
from mgvs.llm.vllm_client import VLLMClient
from mgvs.search.controller import ControllerConfig, run_search
from mgvs.solve.runner import SolveConfig, reset_runtime_state, solve
from mgvs.state.models import create_initial_state
from mgvs.types import StateStatus


class _CountingClient:
    """Deterministic client with call counters for PT/PCT/LSS."""

    def __init__(self) -> None:
        self.pt_calls = 0
        self.pct_calls = 0
        self.lss_calls = 0

    def generate_pt(self, prompt: str) -> str:
        _ = prompt
        self.pt_calls += 1
        return json.dumps(
            {
                "symbolic_objects": {"x": {"kind": "scalar"}},
                "current_equations": ["x + 1 = 2"],
                "domain_constraints": [],
                "global_constraints": [],
                "witness_parameters": {},
                "open_goals": ["isolate x"],
            }
        )

    def generate_pct(self, prompt: str) -> str:
        _ = prompt
        self.pct_calls += 1
        return json.dumps(
            {
                "strategy_tags": ["linear"],
                "open_goals": ["isolate x"],
                "added_facts": [],
                "added_constraints": [],
            }
        )

    def generate_lss(self, prompt: str) -> str:
        _ = prompt
        self.lss_calls += 1
        return json.dumps(
            {
                "actions": [
                    {
                        "action_type": "rewrite",
                        "title": "solve_once",
                        "rationale": "direct solve",
                        "inputs": ["x + 1 = 2"],
                        "outputs": ["x = 1"],
                        "added_facts": ["x = 1"],
                        "added_constraints": [],
                        "branch_labels": [],
                        "metadata": {"mark_solved": True, "normalized_form": "x=1"},
                    }
                ]
            }
        )


class _LoopProposer:
    def propose(self, state, depth):
        _ = state, depth
        from mgvs.actions.models import ActionType, CandidateAction

        return [
            CandidateAction(
                action_type=ActionType.EXPAND,
                title="tick",
                rationale="keep looping",
                outputs=["nf"],
            )
        ]


class TestRuntimePhase12(unittest.TestCase):
    """Covers budget enforcement, stage caching, and vLLM retry behavior."""

    def tearDown(self) -> None:
        reset_runtime_state()

    def test_controller_budget_exhaustion_marks_states(self) -> None:
        ticks = [0.0, 0.0, 0.0, 2.0]

        def time_fn() -> float:
            return ticks.pop(0) if ticks else 2.0

        state = create_initial_state("p", "proof")
        result = run_search(
            state,
            _LoopProposer(),
            config=ControllerConfig(max_depth=10, beam_width=2, max_wall_time_s=1.0, time_fn=time_fn),
        )

        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertTrue(any(s.status == StateStatus.DEAD_END for s in result.final_beam))

    def test_stage_caching_reuses_pt_pct_lss(self) -> None:
        client = _CountingClient()
        cfg = SolveConfig(target_type="equation", max_depth=2, beam_width=2, max_candidates=2)

        first = solve("Solve x + 1 = 2", config=cfg, client=client)
        second = solve("Solve x + 1 = 2", config=cfg, client=client)

        self.assertEqual(first.best_state.status, StateStatus.SOLVED)
        self.assertEqual(second.best_state.status, StateStatus.SOLVED)
        self.assertEqual(client.pt_calls, 1)
        self.assertEqual(client.pct_calls, 1)
        self.assertEqual(client.lss_calls, 1)

    def test_session_budget_exhaustion(self) -> None:
        client = _CountingClient()
        cfg = SolveConfig(target_type="equation", session_max_wall_time_s=1.0)

        with patch("mgvs.solve.runner.time.monotonic", side_effect=[0.0, 2.0]):
            first = solve("Solve x + 1 = 2", config=cfg, client=client)
            second = solve("Solve x + 1 = 2", config=cfg, client=client)

        self.assertEqual(first.termination_reason, "high_priority_solved")
        self.assertEqual(second.termination_reason, "session_budget_exhausted")

    def test_vllm_retry_and_lss_candidate_decay(self) -> None:
        calls: list[dict[str, object]] = []

        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, headers, timeout
            calls.append(payload)
            if len(calls) == 1:
                return {"choices": [{"message": {"content": "not json"}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "actions": [
                                        {
                                            "action_type": "rewrite",
                                            "title": "ok",
                                            "rationale": "ok",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        client = VLLMClient(
            runtime=VLLMRuntimeConfig(model_name="m", retries=1, lss_retry_candidate_decay=0.5),
            transport=transport,
        )
        prompt = build_lss_prompt(create_initial_state("p", "proof"), max_candidates=8)
        raw = client.generate_lss(prompt)

        self.assertEqual(len(calls), 2)
        first_prompt = json.loads(calls[0]["messages"][1]["content"])  # type: ignore[index]
        second_prompt = json.loads(calls[1]["messages"][1]["content"])  # type: ignore[index]
        self.assertEqual(first_prompt["constraints"]["max_candidates"], 1)
        self.assertEqual(second_prompt["constraints"]["max_candidates"], 1)
        self.assertIn("actions", json.loads(raw))


if __name__ == "__main__":
    unittest.main()
