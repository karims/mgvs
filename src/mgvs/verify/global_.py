"""Global compatibility checks for shared witness and constraints."""

from __future__ import annotations

from mgvs.actions.models import CandidateAction
from mgvs.state.models import ReasoningState
from mgvs.verify.base import VerificationResult


class V0GlobalCompatibilityVerifier:
    """v0 global verifier with witness-key compatibility checks."""

    level = "global"

    def verify_action(self, state: ReasoningState, action: CandidateAction) -> VerificationResult:
        """Validate required witness keys declared by action metadata."""

        required = action.metadata.get("required_witness_keys", [])
        if not isinstance(required, list):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="invalid_global_metadata",
                details={"field": "required_witness_keys", "error": "must_be_list"},
            )

        missing = [key for key in required if key not in state.witness_parameters]
        if missing:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="missing_required_witness",
                details={"missing_keys": missing},
            )

        return VerificationResult(
            passed=True,
            level=self.level,
            reason="ok",
            details={},
        )

    def verify_state(self, state: ReasoningState) -> VerificationResult:
        """Validate minimal quality of declared global constraints."""

        empty_constraints = [c for c in state.global_constraints if not c.strip()]
        if empty_constraints:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="invalid_global_constraints",
                details={"error": "contains_empty_constraint"},
            )

        return VerificationResult(
            passed=True,
            level=self.level,
            reason="ok",
            details={},
        )


def verify_global(state: ReasoningState, action: CandidateAction) -> VerificationResult:
    """Convenience function for global action checks."""

    return V0GlobalCompatibilityVerifier().verify_action(state, action)
