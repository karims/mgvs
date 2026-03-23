"""Consistency checks for state/action combinations and transitions."""

from __future__ import annotations

import re

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

        if action.action_type == ActionType.BIND_WITNESS:
            witness_key = action.metadata.get("witness_key")
            if not isinstance(witness_key, str) or not witness_key.strip():
                return VerificationResult(
                    passed=True,
                    level=self.level,
                    reason="ok",
                    details={},
                )
            if witness_key in state.witness_parameters and action.outputs:
                existing = str(state.witness_parameters[witness_key])
                candidate = action.outputs[0].strip()
                if existing and candidate and existing != candidate:
                    return VerificationResult(
                        passed=False,
                        level=self.level,
                        reason="invalid_witness_binding",
                        details={"error": "conflicting_witness_assignment", "witness_key": witness_key},
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

        contradiction_pair = self._find_direct_constraint_contradiction(state)
        if contradiction_pair is not None:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="inconsistent_constraints",
                details={"error": "direct_sign_contradiction", "constraints": list(contradiction_pair)},
            )

        if (
            len(state.branch_assignments) >= 2
            and state.branch_assignments[-1] == state.branch_assignments[-2]
            and state.status == StateStatus.ACTIVE
        ):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="inconsistent_state",
                details={"error": "repeated_branch_assignment", "branch": state.branch_assignments[-1]},
            )

        return VerificationResult(
            passed=True,
            level=self.level,
            reason="ok",
            details={},
        )

    @staticmethod
    def _find_direct_constraint_contradiction(state: ReasoningState) -> tuple[str, str] | None:
        """Detect simple x>k / x<k sign contradictions in constraints."""

        constraints = [*state.domain_constraints, *state.global_constraints]
        gt_map: dict[str, float] = {}
        lt_map: dict[str, float] = {}

        for constraint in constraints:
            text = constraint.replace(" ", "")
            gt_match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)>(-?\d+(?:\.\d+)?)", text)
            lt_match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)<(-?\d+(?:\.\d+)?)", text)
            if gt_match:
                gt_map[gt_match.group(1)] = float(gt_match.group(2))
            if lt_match:
                lt_map[lt_match.group(1)] = float(lt_match.group(2))

        for var, gt in gt_map.items():
            if var not in lt_map:
                continue
            lt = lt_map[var]
            if gt >= lt:
                return (f"{var}>{gt}", f"{var}<{lt}")
        return None


def verify_consistency(state: ReasoningState, action: CandidateAction) -> VerificationResult:
    """Convenience function for consistency action checks."""

    return V0StateConsistencyVerifier().verify_action(state, action)
