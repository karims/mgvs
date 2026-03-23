"""Phase 10 tests for domain plugin detection and guidance behavior."""

import unittest

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.domains import active_domain_plugins
from mgvs.domains.algebra import AlgebraDomainPlugin
from mgvs.domains.number_theory import NumberTheoryDomainPlugin
from mgvs.domains.polynomial import PolynomialDomainPlugin
from mgvs.solve.runner import SolveConfig, solve
from mgvs.state.models import create_initial_state


class TestDomainsPhase10(unittest.TestCase):
    """Covers v1 domain plugin heuristics and state/action enrichment."""

    def test_algebra_plugin_detection_and_tags(self) -> None:
        state = create_initial_state("Solve system: x + y = 2 and x - y = 0", "equation")
        plugin = AlgebraDomainPlugin()

        self.assertTrue(plugin.matches(state))
        plugin.annotate_state(state)

        self.assertIn("domain:algebra", state.strategy_tags)
        self.assertIn("strategy:eliminate", state.strategy_tags)
        self.assertIn("isolate primary unknown", state.open_goals)

    def test_polynomial_plugin_detection_and_goals(self) -> None:
        state = create_initial_state("Find roots of polynomial x^3 - 1", "polynomial")
        plugin = PolynomialDomainPlugin()

        self.assertTrue(plugin.matches(state))
        plugin.annotate_state(state)

        self.assertIn("domain:polynomial", state.strategy_tags)
        self.assertIn("strategy:factorization", state.strategy_tags)
        self.assertIn("choose polynomial representation", state.open_goals)

    def test_number_theory_plugin_adds_constraint_and_validation(self) -> None:
        state = create_initial_state("Given n divisible by 3, show property", "number_theory")
        plugin = NumberTheoryDomainPlugin()

        self.assertTrue(plugin.matches(state))
        plugin.annotate_state(state)

        self.assertIn("domain:number_theory", state.strategy_tags)
        self.assertIn("variables interpreted over integers", state.domain_constraints)

        bad_action = CandidateAction(
            action_type=ActionType.BIND_WITNESS,
            title="bind witness",
            rationale="candidate",
            inputs=["x = k"],
        )
        check = plugin.validate_action(state, bad_action)
        self.assertFalse(check.passed)

    def test_active_domain_plugins_selection(self) -> None:
        state = create_initial_state("Use modulo arithmetic and polynomial identity", "mixed")
        plugins = active_domain_plugins(state)
        names = {plugin.name for plugin in plugins}
        self.assertIn("number_theory", names)
        self.assertIn("polynomial", names)

    def test_solve_integration_adds_domain_tags(self) -> None:
        result = solve("Solve x + 1 = 2", config=SolveConfig(target_type="equation"))
        self.assertIn("domain:algebra", result.best_state.strategy_tags)


if __name__ == "__main__":
    unittest.main()
