"""Heuristic number-theory plugin for modular/divisibility flavored states."""

from __future__ import annotations

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.state.models import ReasoningState

from mgvs.domains.base import (
    BaseDomainPlugin,
    DomainValidationResult,
    ensure_domain_constraint,
    ensure_goal,
    ensure_tag,
)


class NumberTheoryDomainPlugin(BaseDomainPlugin):
    """Adds modular/invariant heuristics and light NT-specific checks."""

    name = "number_theory"

    def matches(self, state: ReasoningState) -> bool:
        haystack = " ".join([state.raw_problem, *state.current_equations, *state.open_goals]).lower()
        markers = ["divisible", "mod", "modulo", "congruent", "prime", "gcd", "integer"]
        return any(marker in haystack for marker in markers)

    def annotate_state(self, state: ReasoningState) -> None:
        ensure_tag(state, "domain:number_theory")
        ensure_tag(state, "strategy:modular")
        ensure_tag(state, "strategy:invariant")
        ensure_tag(state, "strategy:parity")
        ensure_tag(state, "strategy:divisibility_chain")
        ensure_goal(state, "analyze modular constraints")
        ensure_goal(state, "track parity invariant")
        ensure_goal(state, "use divisibility relations")
        ensure_domain_constraint(state, "variables interpreted over integers")

    def validate_action(self, state: ReasoningState, action: CandidateAction) -> DomainValidationResult:
        haystack = " ".join([state.raw_problem, *state.current_equations, *state.open_goals]).lower()
        if action.action_type == ActionType.BIND_WITNESS and "mod" not in " ".join(action.inputs).lower():
            return DomainValidationResult(False, reason="number_theory_witness_without_mod_context")
        if action.action_type == ActionType.EXPAND and "mod" in haystack:
            return DomainValidationResult(False, reason="number_theory_expand_weak_for_modular_goal")
        if action.action_type == ActionType.BRANCH and len(action.branch_labels) > 3:
            return DomainValidationResult(False, reason="number_theory_branch_fanout_too_high")
        return DomainValidationResult(True)
