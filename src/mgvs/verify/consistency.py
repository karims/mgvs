"""Consistency checks for state/action combinations and transitions."""

from __future__ import annotations

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.state.models import ReasoningState
from mgvs.types import StateStatus
from mgvs.verify.base import VerificationResult


TERMINAL_STATUSES: set[StateStatus] = {
    StateStatus.SOLVED,
    StateStatus.CONTRADICTION,
    StateStatus.DEAD_END,
    StateStatus.PARAMETRIC,
}


class V0StateConsistencyVerifier:
    """v0 consistency verifier for obvious transition contradictions."""

    level = "consistency"

    def verify_action(self, state: ReasoningState, action: CandidateAction) -> VerificationResult:
        """Reject contradictory status/action combinations and misuse patterns."""

        if state.status in TERMINAL_STATUSES:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="terminal_state_action",
                details={"status": state.status.value, "action_type": action.action_type.value},
            )

        if action.action_type == ActionType.PRUNE and not state.branch_assignments:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="invalid_prune_usage",
                details={"error": "prune_requires_branch_context"},
            )

        if action.action_type == ActionType.PRUNE and action.branch_labels:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="invalid_prune_usage",
                details={"error": "prune_cannot_create_branches"},
            )

        if action.action_type == ActionType.BRANCH and action.metadata.get("prune_status") is not None:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="invalid_branch_usage",
                details={"error": "branch_cannot_include_prune_status"},
            )

        return VerificationResult(
            passed=True,
            level=self.level,
            reason="ok",
            details={},
        )

    def verify_state(self, state: ReasoningState) -> VerificationResult:
        """Reject clearly inconsistent state/status combinations."""

        if state.status == StateStatus.SOLVED and state.open_goals:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="inconsistent_state",
                details={"error": "solved_state_has_open_goals"},
            )

        if state.status == StateStatus.CONTRADICTION and "high_priority_solved" in state.strategy_tags:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="inconsistent_state",
                details={"error": "contradiction_tagged_solved"},
            )

        return VerificationResult(
            passed=True,
            level=self.level,
            reason="ok",
            details={},
        )


def verify_consistency(state: ReasoningState, action: CandidateAction) -> VerificationResult:
    """Convenience function for consistency action checks."""

    return V0StateConsistencyVerifier().verify_action(state, action)
