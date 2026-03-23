"""Heuristic algebra-domain plugin for equation-style problems."""

from __future__ import annotations

from mgvs.state.models import ReasoningState

from mgvs.domains.base import BaseDomainPlugin, DomainValidationResult, ensure_goal, ensure_tag


class AlgebraDomainPlugin(BaseDomainPlugin):
    """Adds algebraic strategies for equation/system reasoning."""

    name = "algebra"

    def matches(self, state: ReasoningState) -> bool:
        haystack = " ".join(
            [state.raw_problem, *state.current_equations, *state.open_goals, state.target_type]
        ).lower()
        equation_markers = ["=", "equation", "system", "solve"]
        return any(marker in haystack for marker in equation_markers)

    def annotate_state(self, state: ReasoningState) -> None:
        ensure_tag(state, "domain:algebra")
        ensure_tag(state, "strategy:eliminate")
        ensure_tag(state, "strategy:substitute")
        ensure_tag(state, "strategy:factor")

        if len(state.current_equations) > 1:
            ensure_goal(state, "reduce equation system")
        else:
            ensure_goal(state, "isolate primary unknown")

    def validate_state(self, state: ReasoningState) -> DomainValidationResult:
        # Keep v1 permissive; only fail clearly inconsistent solved states.
        if state.status.value == "solved" and not state.derived_facts:
            return DomainValidationResult(False, reason="algebra_solved_without_facts")
        return DomainValidationResult(True)
