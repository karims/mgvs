"""Heuristic algebra-domain plugin for equation-style problems."""

from __future__ import annotations

from mgvs.actions.models import ActionType, CandidateAction
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
        ensure_tag(state, "strategy:normalize_equations")
        ensure_tag(state, "strategy:eliminate")
        ensure_tag(state, "strategy:substitute")
        ensure_tag(state, "strategy:factor")
        ensure_tag(state, "strategy:check_consistency")

        equation_count = len(state.current_equations)
        if equation_count <= 1:
            equation_count = max(0, state.raw_problem.count("="))

        if equation_count > 1:
            ensure_goal(state, "reduce equation system")
            ensure_goal(state, "eliminate one variable")
        else:
            ensure_goal(state, "isolate primary unknown")
            ensure_goal(state, "simplify equation form")

    def validate_action(self, state: ReasoningState, action: CandidateAction) -> DomainValidationResult:
        """Flag likely low-value action families for algebraic states."""

        if action.action_type == ActionType.DETECT_SYMMETRY:
            haystack = " ".join([state.raw_problem, *state.current_equations]).lower()
            if "symmetric" not in haystack and "symmetry" not in haystack:
                return DomainValidationResult(False, reason="algebra_unmotivated_symmetry_detection")

        if action.action_type == ActionType.BRANCH and len(state.current_equations) <= 1:
            goals_text = " ".join(state.open_goals).lower()
            if "case" not in goals_text and "branch" not in goals_text:
                return DomainValidationResult(False, reason="algebra_unmotivated_branch")
        return DomainValidationResult(True)

    def validate_state(self, state: ReasoningState) -> DomainValidationResult:
        # Keep v1 permissive; only fail clearly inconsistent solved states.
        if state.status.value == "solved" and not state.derived_facts:
            return DomainValidationResult(False, reason="algebra_solved_without_facts")
        return DomainValidationResult(True)
