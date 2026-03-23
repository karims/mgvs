"""Phase 4 tests for multi-level verification framework."""

import unittest

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.state.models import create_initial_state
from mgvs.types import StateStatus
from mgvs.verify.base import CompositeVerifier
from mgvs.verify.consistency import V0StateConsistencyVerifier
from mgvs.verify.global_ import V0GlobalCompatibilityVerifier
from mgvs.verify.local import V0LocalValidityVerifier


class TestVerificationPhase4(unittest.TestCase):
    """Covers v0 local/consistency/global verification behavior."""

    def setUp(self) -> None:
        self.verifier = CompositeVerifier(
            local_verifier=V0LocalValidityVerifier(),
            consistency_verifier=V0StateConsistencyVerifier(),
            global_verifier=V0GlobalCompatibilityVerifier(),
        )

    def test_pass_case(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        action = CandidateAction(
            action_type=ActionType.REWRITE,
            title="normalize",
            rationale="canonical rewrite",
            outputs=["nf"],
            added_facts=["f1"],
        )

        action_result = self.verifier.verify_action(state, action)
        state_result = self.verifier.verify_state(state)

        self.assertTrue(action_result.passed)
        self.assertTrue(state_result.passed)

    def test_malformed_action_rejection(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        action = CandidateAction(
            action_type=ActionType.REWRITE,
            title="   ",
            rationale="bad",
        )

        result = self.verifier.verify_action(state, action)

        self.assertFalse(result.passed)
        self.assertIsNotNone(result.first_failure)
        self.assertEqual(result.first_failure.level, "local")
        self.assertEqual(result.first_failure.reason, "malformed_action")

    def test_missing_witness_and_invalid_global_constraint_rejection(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        state.global_constraints.append("   ")
        action = CandidateAction(
            action_type=ActionType.BIND_WITNESS,
            title="bind witness",
            rationale="requires witness",
            metadata={"required_witness_keys": ["w_key"]},
        )

        action_result = self.verifier.verify_action(state, action)
        state_result = self.verifier.verify_state(state)

        self.assertFalse(action_result.passed)
        self.assertIsNotNone(action_result.first_failure)
        self.assertEqual(action_result.first_failure.level, "global")
        self.assertEqual(action_result.first_failure.reason, "missing_required_witness")

        self.assertFalse(state_result.passed)
        self.assertIsNotNone(state_result.first_failure)
        self.assertEqual(state_result.first_failure.level, "global")
        self.assertEqual(state_result.first_failure.reason, "invalid_global_constraints")

    def test_contradiction_style_rejection(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        state.status = StateStatus.CONTRADICTION
        action = CandidateAction(
            action_type=ActionType.REWRITE,
            title="try rewrite",
            rationale="should be blocked",
            outputs=["x"],
        )

        result = self.verifier.verify_action(state, action)

        self.assertFalse(result.passed)
        self.assertIsNotNone(result.first_failure)
        self.assertEqual(result.first_failure.level, "consistency")
        self.assertEqual(result.first_failure.reason, "terminal_state_action")


if __name__ == "__main__":
    unittest.main()
