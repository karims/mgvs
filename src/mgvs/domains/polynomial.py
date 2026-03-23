"""Heuristic polynomial-domain plugin for polynomial/series flavored states."""

from __future__ import annotations

import re

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.state.models import ReasoningState

from mgvs.domains.base import BaseDomainPlugin, DomainValidationResult, ensure_goal, ensure_tag


class PolynomialDomainPlugin(BaseDomainPlugin):
    """Adds representation/factorization guidance for polynomial structures."""

    name = "polynomial"

    def matches(self, state: ReasoningState) -> bool:
        haystack = " ".join([state.raw_problem, *state.current_equations, *state.open_goals]).lower()
        return (
            "polynomial" in haystack
            or "generating function" in haystack
            or bool(re.search(r"\^\d", haystack))
        )

    def annotate_state(self, state: ReasoningState) -> None:
        ensure_tag(state, "domain:polynomial")
        ensure_tag(state, "strategy:representation")
        ensure_tag(state, "strategy:factorization")
        ensure_tag(state, "strategy:degree_analysis")
        ensure_tag(state, "strategy:root_structure")
        ensure_goal(state, "choose polynomial representation")
        ensure_goal(state, "estimate polynomial degree")
        ensure_goal(state, "consider factor-or-substitute move")

    def validate_action(self, state: ReasoningState, action: CandidateAction) -> DomainValidationResult:
        """Flag likely poor action choices for polynomial flavored states."""

        if action.action_type == ActionType.EXPAND and self._estimated_degree(state) >= 5:
            return DomainValidationResult(False, reason="polynomial_expand_likely_explosive")
        if action.action_type == ActionType.DETECT_SYMMETRY and "symmetric" not in state.raw_problem.lower():
            return DomainValidationResult(False, reason="polynomial_unmotivated_symmetry_detection")
        return DomainValidationResult(True)

    @staticmethod
    def _estimated_degree(state: ReasoningState) -> int:
        """Estimate highest exponent seen in problem/equations."""

        haystack = " ".join([state.raw_problem, *state.current_equations]).lower()
        powers = [int(match) for match in re.findall(r"\^(\d+)", haystack)]
        return max(powers) if powers else 1
