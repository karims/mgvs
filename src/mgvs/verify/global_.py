"""Global compatibility checks for shared witness and constraints."""

from __future__ import annotations

import re

from mgvs.actions.models import ActionType
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
        if any(not isinstance(key, str) or not key.strip() for key in required):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="invalid_global_metadata",
                details={"field": "required_witness_keys", "error": "non_string_key"},
            )

        missing = [key for key in required if key not in state.witness_parameters]
        if missing:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="missing_required_witness",
                details={"missing_keys": missing},
            )

        required_constraints = action.metadata.get("required_global_constraints", [])
        if required_constraints and not isinstance(required_constraints, list):
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="invalid_global_metadata",
                details={"field": "required_global_constraints", "error": "must_be_list"},
            )
        if isinstance(required_constraints, list):
            absent = [c for c in required_constraints if isinstance(c, str) and c not in state.global_constraints]
            if absent:
                return VerificationResult(
                    passed=False,
                    level=self.level,
                    reason="missing_required_global_constraint",
                    details={"missing_constraints": absent},
                )

        if action.action_type == ActionType.BIND_WITNESS:
            witness_key = action.metadata.get("witness_key")
            if not isinstance(witness_key, str) or not witness_key.strip():
                return VerificationResult(
                    passed=False,
                    level=self.level,
                    reason="invalid_global_metadata",
                    details={"field": "witness_key", "error": "required_for_bind_witness"},
                )
            if witness_key not in state.witness_parameters and not action.outputs:
                return VerificationResult(
                    passed=False,
                    level=self.level,
                    reason="missing_required_witness",
                    details={"missing_keys": [witness_key], "hint": "bind_witness_needs_existing_or_output"},
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

        bad_mod_constraints = [c for c in state.global_constraints if self._invalid_mod_constraint(c)]
        if bad_mod_constraints:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="invalid_global_constraints",
                details={"error": "non_positive_modulus", "constraints": bad_mod_constraints[:2]},
            )

        unresolved_witness_refs = [
            key for key in self._referenced_witness_keys(state.global_constraints) if key not in state.witness_parameters
        ]
        if unresolved_witness_refs:
            return VerificationResult(
                passed=False,
                level=self.level,
                reason="missing_required_witness",
                details={"missing_keys": unresolved_witness_refs, "source": "global_constraints"},
            )

        return VerificationResult(
            passed=True,
            level=self.level,
            reason="ok",
            details={},
        )

    @staticmethod
    def _invalid_mod_constraint(constraint: str) -> bool:
        """Return True when constraint uses non-positive modulus."""

        match = re.search(r"mod\\s*(-?\\d+)", constraint.lower())
        if not match:
            return False
        modulus = int(match.group(1))
        return modulus <= 0

    @staticmethod
    def _referenced_witness_keys(constraints: list[str]) -> set[str]:
        """Extract witness:<key> references from global constraints."""

        refs: set[str] = set()
        for constraint in constraints:
            for key in re.findall(r"witness:([a-zA-Z_][a-zA-Z0-9_]*)", constraint):
                refs.add(key)
        return refs


def verify_global(state: ReasoningState, action: CandidateAction) -> VerificationResult:
    """Convenience function for global action checks."""

    return V0GlobalCompatibilityVerifier().verify_action(state, action)
