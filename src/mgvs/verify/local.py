"""Local validity checks for structural action/state correctness."""

from __future__ import annotations

from typing import Any

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.state.models import ReasoningState
from mgvs.verify.base import VerificationResult


class V0LocalValidityVerifier:
    """v0 local verifier for lightweight structural checks."""

    level = "local"

    def verify_action(self, state: ReasoningState, action: CandidateAction) -> VerificationResult:
        """Reject malformed or effectively empty action proposals."""

        _ = state
        if not action.title.strip():
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "title", "error": "empty"},
            )
        if not action.rationale.strip():
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "rationale", "error": "empty"},
            )
        if not isinstance(action.action_type, ActionType):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "action_type", "error": "invalid_enum"},
            )
        if action.action_type == ActionType.BRANCH and not action.branch_labels:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "branch_labels", "error": "missing"},
            )
        if action.action_type != ActionType.BRANCH and action.branch_labels:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "branch_labels", "error": "unexpected"},
            )
        if self._is_effectively_empty_action(action):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="empty_action",
                details={"action_type": action.action_type.value},
            )

        return VerificationResult(
            passed=True,
            level=self.level,
            reason="ok",
            details={},
        )

    def verify_state(self, state: ReasoningState) -> VerificationResult:
        """Check minimal structural integrity of state payload."""

        if not state.raw_problem.strip():
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_state",
                details={"field": "raw_problem", "error": "empty"},
            )
        if not state.target_type.strip():
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_state",
                details={"field": "target_type", "error": "empty"},
            )
        if not isinstance(state.symbolic_objects, dict):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_state",
                details={"field": "symbolic_objects", "error": "not_dict"},
            )
        return VerificationResult(
            passed=True,
            level=self.level,
            reason="ok",
            details={},
        )

    @staticmethod
    def _is_effectively_empty_action(action: CandidateAction) -> bool:
        """Detect action proposals that do not carry actionable updates."""

        if action.action_type == ActionType.PRUNE:
            return False

        payloads: list[Any] = [
            action.inputs,
            action.outputs,
            action.added_facts,
            action.added_constraints,
            action.branch_labels,
        ]
        if any(payload for payload in payloads):
            return False

        metadata_keys = {
            "mark_solved",
            "mark_parametric",
            "mark_needs_reinterpretation",
            "normalized_form",
            "required_witness_keys",
        }
        return not any(key in action.metadata for key in metadata_keys)


def verify_local(state: ReasoningState, action: CandidateAction) -> VerificationResult:
    """Convenience function for local action validity checks."""

    return V0LocalValidityVerifier().verify_action(state, action)
