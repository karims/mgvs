"""Heuristic polynomial-domain plugin for polynomial/series flavored states."""

from __future__ import annotations

import re

from mgvs.state.models import ReasoningState

from mgvs.domains.base import BaseDomainPlugin, ensure_goal, ensure_tag


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
        ensure_goal(state, "choose polynomial representation")
