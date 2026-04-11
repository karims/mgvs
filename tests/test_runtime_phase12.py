"""Phase 12 tests for runtime budgets, caching, hardened inference, and debug flow."""

import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.config import VLLMRuntimeConfig
from mgvs.llm.prompts import build_lss_prompt
from mgvs.llm.vllm_client import VLLMClient
from mgvs.search.controller import ControllerConfig, run_search
from mgvs.solve.runner import (
    DomainAwareProposer,
    RunAttemptContext,
    SolveConfig,
    TrackingCompositeVerifier,
    build_default_verifier,
    reset_runtime_state,
    solve,
)
from mgvs.state.models import create_initial_state
from mgvs.state.trace import TraceStep
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


class _NoActionClient:
    def generate_pt(self, prompt: str) -> str:
        _ = prompt
        return '{"symbolic_objects":{},"current_equations":[],"open_goals":["g"],"domain_constraints":[],"global_constraints":[],"witness_parameters":{}}'

    def generate_pct(self, prompt: str) -> str:
        _ = prompt
        return '{"strategy_tags":[],"open_goals":["g"],"added_facts":[],"added_constraints":[]}'

    def generate_lss(self, prompt: str) -> str:
        _ = prompt
        return '{"actions":[]}'


class _OneActionProposer:
    def propose(self, state, depth):
        _ = state, depth
        return [
            CandidateAction(
                action_type=ActionType.DERIVE_CONSTRAINT,
                title="candidate_bound",
                rationale="test candidate",
                added_facts=["n <= 10"],
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

    def test_controller_logs_when_accepted_candidates_produce_no_next_states(self) -> None:
        class _RejectingVerifier:
            def __init__(self) -> None:
                self._last_action_rejection = None
                self._last_state_rejection = None

            def is_action_valid(self, state, action):
                _ = state, action
                self._last_action_rejection = None
                return True

            def is_state_valid(self, state):
                _ = state
                self._last_state_rejection = {
                    "layer": "state_transition_reject",
                    "reason": "synthetic_invalid_state",
                    "details": {"field": "current_equations"},
                }
                return False

            def consume_last_action_rejection(self):
                rejection = self._last_action_rejection
                self._last_action_rejection = None
                return rejection

            def consume_last_state_rejection(self):
                rejection = self._last_state_rejection
                self._last_state_rejection = None
                return rejection

        state = create_initial_state("p", "proof")
        buffer = StringIO()
        with patch.dict(os.environ, {"MGVS_DEBUG_RUNTIME": "1"}, clear=False):
            with redirect_stdout(buffer):
                run_search(
                    state,
                    _OneActionProposer(),
                    verifier=_RejectingVerifier(),
                    config=ControllerConfig(max_depth=1, beam_width=1, candidate_cap_per_state=1),
                )

        output = buffer.getvalue()
        self.assertIn("state_transition_reject", output)
        self.assertIn("synthetic_invalid_state", output)
        self.assertIn("accepted candidate(s) produced no next states", output)

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
            runtime=VLLMRuntimeConfig(
                model_name="m",
                retries=1,
                lss_retries=2,
                lss_max_tokens=192,
                lss_retry_candidate_decay=0.5,
            ),
            transport=transport,
        )
        prompt = build_lss_prompt(create_initial_state("p", "proof"), max_candidates=8)
        raw = client.generate_lss(prompt)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["max_tokens"], 192)
        self.assertEqual(calls[1]["max_tokens"], 192)
        first_prompt = json.loads(calls[0]["messages"][1]["content"])  # type: ignore[index]
        second_prompt = json.loads(calls[1]["messages"][1]["content"])  # type: ignore[index]
        self.assertEqual(first_prompt["constraints"]["max_candidates"], 1)
        self.assertEqual(second_prompt["constraints"]["max_candidates"], 1)
        self.assertIn("actions", json.loads(raw))

    def test_vllm_runtime_config_reads_debug_single_path_from_env(self) -> None:
        with patch.dict(os.environ, {"MGVS_DEBUG_SINGLE_PATH": "1"}, clear=False):
            cfg = VLLMRuntimeConfig.from_env()
        self.assertTrue(cfg.debug_single_path)

    def test_vllm_with_overrides_updates_stage_retry_and_token_caps(self) -> None:
        client = VLLMClient(runtime=VLLMRuntimeConfig())
        overridden = client.with_overrides(
            retries=0,
            pt_retries=0,
            pct_retries=0,
            lss_retries=0,
            pt_max_tokens=512,
            pct_max_tokens=512,
            lss_max_tokens=256,
            debug_single_path=True,
        )

        self.assertEqual(overridden._runtime.retries, 0)
        self.assertEqual(overridden._runtime.pt_retries, 0)
        self.assertEqual(overridden._runtime.pct_retries, 0)
        self.assertEqual(overridden._runtime.lss_retries, 0)
        self.assertEqual(overridden._runtime.pt_max_tokens, 512)
        self.assertEqual(overridden._runtime.pct_max_tokens, 512)
        self.assertEqual(overridden._runtime.lss_max_tokens, 256)
        self.assertTrue(overridden._runtime.debug_single_path)

    def test_duplicate_lss_action_is_rejected(self) -> None:
        state = create_initial_state("p", "proof")
        state.add_trace_step(
            TraceStep(
                action="eliminate",
                rationale="prior step",
                updates={
                    "title": "eliminate_n",
                    "added_facts": ["n is bounded"],
                    "added_constraints": [],
                },
            )
        )

        class _Client:
            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "eliminate",
                                "title": "eliminate_n",
                                "added_facts": ["n is bounded"],
                                "added_constraints": [],
                            }
                        ]
                    }
                )

        proposer = DomainAwareProposer(
            client=_Client(),
            plugins=[],
            max_candidates=1,
            cache_prefix="test",
            allow_expensive_branching=False,
            attempt_context=RunAttemptContext(),
        )

        self.assertEqual(proposer.propose(state, 0), [])

    def test_semantic_duplicate_eliminate_action_is_rejected_even_with_new_title(self) -> None:
        state = create_initial_state("p", "proof")
        state.add_trace_step(
            TraceStep(
                action="eliminate",
                rationale="prior step",
                updates={
                    "title": "eliminate_n",
                    "added_facts": ["n is bounded"],
                    "added_constraints": ["n <= 10"],
                },
            )
        )

        class _Client:
            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "eliminate",
                                "title": "rename_same_elimination",
                                "added_facts": ["n is bounded"],
                                "added_constraints": ["n <= 10"],
                            }
                        ]
                    }
                )

        proposer = DomainAwareProposer(
            client=_Client(),
            plugins=[],
            max_candidates=1,
            cache_prefix="test",
            allow_expensive_branching=False,
            attempt_context=RunAttemptContext(),
        )

        self.assertEqual(proposer.propose(state, 0), [])

    def test_no_new_information_action_is_rejected(self) -> None:
        state = create_initial_state("p", "proof")
        state.derived_facts.append("n is bounded")
        state.domain_constraints.append("perimeters are distinct")

        class _Client:
            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "repeat_known_info",
                                "added_facts": ["n is bounded"],
                                "added_constraints": ["perimeters are distinct"],
                            }
                        ]
                    }
                )

        proposer = DomainAwareProposer(
            client=_Client(),
            plugins=[],
            max_candidates=1,
            cache_prefix="test",
            allow_expensive_branching=False,
            attempt_context=RunAttemptContext(),
        )

        self.assertEqual(proposer.propose(state, 0), [])

    def test_equation_restatement_action_is_rejected(self) -> None:
        state = create_initial_state("p", "proof")
        state.current_equations.append("x + y = 10")

        class _Client:
            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "copy_equation_back",
                                "added_facts": ["x + y = 10"],
                                "added_constraints": ["x + y = 10"],
                            }
                        ]
                    }
                )

        proposer = DomainAwareProposer(
            client=_Client(),
            plugins=[],
            max_candidates=1,
            cache_prefix="test",
            allow_expensive_branching=False,
            attempt_context=RunAttemptContext(),
        )

        self.assertEqual(proposer.propose(state, 0), [])

    def test_genuinely_new_lss_action_is_accepted(self) -> None:
        state = create_initial_state("p", "proof")
        state.derived_facts.append("n is bounded")

        class _Client:
            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "add_unique_perimeter_constraint",
                                "added_facts": [],
                                "added_constraints": ["distinct rectangles must have distinct perimeters"],
                            }
                        ]
                    }
                )

        proposer = DomainAwareProposer(
            client=_Client(),
            plugins=[],
            max_candidates=1,
            cache_prefix="test",
            allow_expensive_branching=False,
            attempt_context=RunAttemptContext(),
        )

        actions = proposer.propose(state, 0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, ActionType.DERIVE_CONSTRAINT)

    def test_grounded_new_action_scores_higher_than_generic_restatement(self) -> None:
        state = create_initial_state("rectangle problem", "proof")
        state.symbolic_objects["K"] = {"kind": "entity"}
        state.domain_constraints.append("distinct rectangles must have distinct perimeters")
        state.open_goals.append("find K mod 10^5")

        class _Client:
            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "restate_target",
                                "added_facts": ["K mod 10^5 is required"],
                                "added_constraints": [],
                            },
                            {
                                "action_type": "derive_constraint",
                                "title": "use_unique_perimeters",
                                "added_facts": [],
                                "added_constraints": ["distinct rectangles imply distinct perimeter counts"],
                            },
                        ]
                    }
                )

        proposer = DomainAwareProposer(
            client=_Client(),
            plugins=[],
            max_candidates=2,
            cache_prefix="test",
            allow_expensive_branching=False,
            attempt_context=RunAttemptContext(),
        )

        actions = proposer.propose(state, 0)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].title, "use_unique_perimeters")

    def test_repeated_action_scores_lower_than_fresh_action(self) -> None:
        state = create_initial_state("counting problem", "proof")
        state.add_trace_step(
            TraceStep(
                action="derive_constraint",
                rationale="prior step",
                updates={
                    "title": "repeat_counting_relation",
                    "added_facts": ["perimeter count is bounded"],
                    "added_constraints": [],
                },
            )
        )

        class _Client:
            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "derive_constraint",
                                "title": "fresh_bound",
                                "added_facts": ["number of distinct perimeters is at most 998"],
                                "added_constraints": [],
                            },
                            {
                                "action_type": "derive_constraint",
                                "title": "repeat_variant",
                                "added_facts": ["perimeter count is bounded"],
                                "added_constraints": [],
                            },
                        ]
                    }
                )

        proposer = DomainAwareProposer(
            client=_Client(),
            plugins=[],
            max_candidates=2,
            cache_prefix="test",
            allow_expensive_branching=False,
            attempt_context=RunAttemptContext(),
        )

        actions = proposer.propose(state, 0)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].title, "fresh_bound")

    def test_debug_single_path_disables_fallback_attempt(self) -> None:
        result = solve(
            "custom unsolved",
            config=SolveConfig(requested_mode="balanced", debug_single_path=True),
            client=_NoActionClient(),
        )

        self.assertFalse(result.fallback_used)
        self.assertEqual(result.solve_mode, "balanced")
        self.assertEqual(result.fallback_reason, "")

    def test_debug_single_path_prints_effective_debug_settings(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            solve(
                "custom unsolved",
                config=SolveConfig(requested_mode="balanced", debug_single_path=True),
                client=_NoActionClient(),
            )

        output = buffer.getvalue()
        self.assertIn("debug_single_path enabled:", output)
        self.assertIn("token_caps={pt:512,pct:512,lss:256}", output)
        self.assertIn("beam_width=1", output)
        self.assertIn("candidate_cap_per_state=1", output)

    def test_local_verifier_rejection_logs_title_type_and_reason(self) -> None:
        state = create_initial_state("Solve x + 1 = 2", "equation")
        state.open_goals.append("isolate x")

        class _InvalidLSSClient:
            def generate_lss(self, prompt: str) -> str:
                _ = prompt
                return json.dumps(
                    {
                        "actions": [
                            {
                                "action_type": "rewrite",
                                "title": "bad_rewrite",
                                "added_facts": ["x = 1"],
                                "added_constraints": [],
                            }
                        ]
                    }
                )

        proposer = DomainAwareProposer(
            client=_InvalidLSSClient(),
            plugins=[],
            max_candidates=1,
            cache_prefix="verify",
            allow_expensive_branching=False,
            attempt_context=RunAttemptContext(),
        )
        verifier = TrackingCompositeVerifier(build_default_verifier())
        action = proposer.propose(state, 0)[0]

        with patch.dict(os.environ, {"MGVS_DEBUG_RUNTIME": "1"}, clear=False):
            buffer = StringIO()
            with redirect_stdout(buffer):
                accepted = verifier.is_action_valid(state, action)

        output = buffer.getvalue()
        self.assertFalse(accepted)
        self.assertIn("local_reject", output)
        self.assertIn("bad_rewrite", output)
        self.assertIn("rewrite", output)
        self.assertIn("malformed_action", output)


if __name__ == "__main__":
    unittest.main()
