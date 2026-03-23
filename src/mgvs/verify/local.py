"""Local validity checks for structural action/state correctness."""

from __future__ import annotations

import re
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
        branch_check = self._validate_branch_action(action)
        if branch_check is not None:
            return branch_check

        transform_check = self._validate_transform_action(action)
        if transform_check is not None:
            return transform_check

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
        malformed_equations = [eq for eq in state.current_equations if not self._looks_equation_like(eq)]
        if malformed_equations:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_equation",
                details={"examples": malformed_equations[:2]},
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

    def _validate_branch_action(self, action: CandidateAction) -> VerificationResult | None:
        """Validate branch label structure and safety bounds."""

        if action.action_type != ActionType.BRANCH:
            return None

        labels = [label.strip() for label in action.branch_labels]
        if any(not label for label in labels):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "branch_labels", "error": "contains_empty_label"},
            )

        if len(set(labels)) != len(labels):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "branch_labels", "error": "duplicate_labels"},
            )

        if len(labels) > 6:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "branch_labels", "error": "fanout_too_large", "fanout": len(labels)},
            )
        return None

    def _validate_transform_action(self, action: CandidateAction) -> VerificationResult | None:
        """Validate structure for rewrite/substitute family actions."""

        transform_types = {
            ActionType.REWRITE,
            ActionType.SUBSTITUTE,
            ActionType.ELIMINATE,
            ActionType.FACTOR,
            ActionType.EXPAND,
        }
        if action.action_type not in transform_types:
            return None

        if action.action_type == ActionType.SUBSTITUTE and not action.inputs:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "inputs", "error": "substitute_requires_input"},
            )

        if action.action_type in {ActionType.REWRITE, ActionType.SUBSTITUTE} and not action.outputs:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="malformed_action",
                details={"field": "outputs", "error": "transform_requires_output"},
            )

        if action.action_type == ActionType.SUBSTITUTE:
            samples = list(action.inputs) + list(action.outputs)
            if samples and not any(self._looks_equation_like(sample) for sample in samples):
                return VerificationResult(
                    passed=False,
                    level=self.level,
                    reason="malformed_action",
                    details={"field": "inputs/outputs", "error": "not_equation_like_for_substitute"},
                )
        return None

    @staticmethod
    def _looks_equation_like(text: str) -> bool:
        """Heuristic equation/constraint detector for lightweight validation."""

        value = text.strip()
        if not value:
            return False
        if any(token in value for token in ("=", "<", ">", "mod", "congruent", "|")):
            return True
        return bool(re.search(r"[a-zA-Z]", value) and re.search(r"\d|\^|\+|-|\*", value))


def verify_local(state: ReasoningState, action: CandidateAction) -> VerificationResult:
    """Convenience function for local action validity checks."""

    return V0LocalValidityVerifier().verify_action(state, action)
