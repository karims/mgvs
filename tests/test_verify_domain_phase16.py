"""Phase 16 tests for strengthened verifier and domain plugin behavior."""

import unittest

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.domains.algebra import AlgebraDomainPlugin
from mgvs.domains.number_theory import NumberTheoryDomainPlugin
from mgvs.domains.polynomial import PolynomialDomainPlugin
from mgvs.state.models import create_initial_state
from mgvs.verify.consistency import V0StateConsistencyVerifier
from mgvs.verify.global_ import V0GlobalCompatibilityVerifier
from mgvs.verify.local import V0LocalValidityVerifier


class TestVerifyDomainPhase16(unittest.TestCase):
    """Checks narrow high-value verifier/domain improvements."""

    def test_local_accepts_valid_substitution(self) -> None:
        verifier = V0LocalValidityVerifier()
        state = create_initial_state("Solve x + 1 = 2", "equation")
        action = CandidateAction(
            action_type=ActionType.SUBSTITUTE,
            title="substitute x",
            rationale="use equality",
            inputs=["x = 1"],
            outputs=["x + 1 = 2"],
        )
        result = verifier.verify_action(state, action)
        self.assertTrue(result.passed)

    def test_local_rejects_bad_branch_labels(self) -> None:
        verifier = V0LocalValidityVerifier()
        state = create_initial_state("Branch case", "proof")
        action = CandidateAction(
            action_type=ActionType.BRANCH,
            title="split",
            rationale="cases",
            branch_labels=["case_a", "case_a"],
        )
        result = verifier.verify_action(state, action)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "malformed_action")

    def test_local_rejects_malformed_state_equations(self) -> None:
        verifier = V0LocalValidityVerifier()
        state = create_initial_state("p", "proof")
        state.current_equations = ["", "hello"]
        result = verifier.verify_state(state)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "malformed_equation")

    def test_consistency_rejects_constraint_contradiction(self) -> None:
        verifier = V0StateConsistencyVerifier()
        state = create_initial_state("p", "proof")
        state.domain_constraints = ["x > 5", "x < 3"]
        result = verifier.verify_state(state)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "inconsistent_constraints")

    def test_consistency_rejects_invalid_witness_binding(self) -> None:
        verifier = V0StateConsistencyVerifier()
        state = create_initial_state("p", "proof")
        state.witness_parameters["w"] = "1"
        action = CandidateAction(
            action_type=ActionType.BIND_WITNESS,
            title="bind",
            rationale="bind witness",
            metadata={"witness_key": "w"},
            outputs=["2"],
        )
        result = verifier.verify_action(state, action)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "invalid_witness_binding")

    def test_global_witness_compatibility_failures(self) -> None:
        verifier = V0GlobalCompatibilityVerifier()
        state = create_initial_state("p", "proof")
        state.global_constraints = ["witness:alpha must satisfy condition"]
        action = CandidateAction(
            action_type=ActionType.BIND_WITNESS,
            title="bind",
            rationale="bind witness",
            metadata={"required_global_constraints": ["must_exist"], "witness_key": "alpha"},
        )

        action_result = verifier.verify_action(state, action)
        state_result = verifier.verify_state(state)

        self.assertFalse(action_result.passed)
        self.assertEqual(action_result.reason, "missing_required_global_constraint")
        self.assertFalse(state_result.passed)
        self.assertEqual(state_result.reason, "missing_required_witness")

    def test_domain_plugins_add_sharper_guidance(self) -> None:
        algebra = AlgebraDomainPlugin()
        poly = PolynomialDomainPlugin()
        nt = NumberTheoryDomainPlugin()

        s1 = create_initial_state("Solve x + y = 2", "equation")
        algebra.annotate_state(s1)
        self.assertIn("strategy:normalize_equations", s1.strategy_tags)
        self.assertIn("simplify equation form", s1.open_goals)

        s2 = create_initial_state("Polynomial x^6 + x^2 + 1", "polynomial")
        poly.annotate_state(s2)
        self.assertIn("strategy:degree_analysis", s2.strategy_tags)

        bad_poly_action = CandidateAction(
            action_type=ActionType.EXPAND,
            title="expand",
            rationale="expand polynomial",
        )
        self.assertFalse(poly.validate_action(s2, bad_poly_action).passed)

        s3 = create_initial_state("n is divisible by 3, work mod 3", "number_theory")
        nt.annotate_state(s3)
        self.assertIn("strategy:parity", s3.strategy_tags)

        bad_nt_action = CandidateAction(
            action_type=ActionType.EXPAND,
            title="expand",
            rationale="expand",
        )
        self.assertFalse(nt.validate_action(s3, bad_nt_action).passed)


if __name__ == "__main__":
    unittest.main()
