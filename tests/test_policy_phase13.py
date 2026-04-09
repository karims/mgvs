"""Phase 13 tests for solve-mode policy routing and fallback behavior."""

import unittest
from unittest.mock import patch

from mgvs.config import SolveModeSettings, SolvePolicyConfig
from mgvs.solve.policy import SolveMode, select_solve_mode
from mgvs.solve.runner import SolveConfig, solve
from mgvs.state.models import create_initial_state


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


class TestPolicyPhase13(unittest.TestCase):
    def test_policy_config_serializable(self) -> None:
        cfg = SolvePolicyConfig.default()
        payload = cfg.to_dict()
        self.assertIn("fast", payload)
        self.assertIn("balanced", payload)
        self.assertIn("deep", payload)

    def test_mode_selection_simple_problem(self) -> None:
        cfg = SolvePolicyConfig.default()
        state = create_initial_state("Solve x + 1 = 2", "equation")
        selected = select_solve_mode("Solve x + 1 = 2", state, cfg)
        self.assertEqual(selected.mode, SolveMode.FAST)

    def test_mode_selection_harder_problem(self) -> None:
        cfg = SolvePolicyConfig.default()
        long_problem = "Given polynomial and modular structure " + ("x^5 + x^3 + " * 80)
        state = create_initial_state(long_problem, "number_theory")
        selected = select_solve_mode(long_problem, state, cfg)
        self.assertEqual(selected.mode, SolveMode.DEEP)

    def test_config_propagation_into_runtime_settings(self) -> None:
        policy = SolvePolicyConfig.default()
        custom_deep = SolveModeSettings(
            beam_width=2,
            max_depth=1,
            max_candidates_per_state=1,
            llm_retries=0,
            allow_expensive_branching=False,
            use_pt=True,
            use_pct=True,
            use_lss=True,
        )
        policy = SolvePolicyConfig(
            fast=policy.fast,
            balanced=policy.balanced,
            deep=custom_deep,
            easy_problem_max_chars=policy.easy_problem_max_chars,
            hard_problem_min_chars=policy.hard_problem_min_chars,
            budget_pressure_fast_threshold=policy.budget_pressure_fast_threshold,
            budget_pressure_fallback_threshold=policy.budget_pressure_fallback_threshold,
            malformed_retry_fallback_threshold=policy.malformed_retry_fallback_threshold,
        )
        result = solve(
            "A very hard number theory question about divisibility and congruences" + (" x^2" * 100),
            config=SolveConfig(requested_mode="deep", policy_config=policy, max_depth=9),
        )
        self.assertLessEqual(result.depth_reached, 1)
        self.assertEqual(result.solve_mode, "deep")

    def test_fallback_trigger_when_search_fails(self) -> None:
        result = solve(
            "custom unsolved",
            config=SolveConfig(requested_mode="balanced"),
            client=_NoActionClient(),
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.solve_mode, "fast")
        self.assertEqual(result.fallback_reason, "no_valid_branches_survived")

    def test_fallback_trigger_under_budget_pressure(self) -> None:
        with patch("mgvs.search.controller.time.monotonic", side_effect=[0.0, 2.0, 2.0, 2.0]):
            result = solve(
                "budget pressure demo",
                config=SolveConfig(requested_mode="deep", max_wall_time_s=0.5),
            )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "no_valid_branches_survived")

    def test_metadata_shows_mode_used(self) -> None:
        result = solve("Solve x + 1 = 2", config=SolveConfig(requested_mode="auto"))
        self.assertIn(result.solve_mode, {"fast", "balanced", "deep"})
        self.assertTrue(any(item.startswith("mode=") for item in result.policy_trace))


if __name__ == "__main__":
    unittest.main()
