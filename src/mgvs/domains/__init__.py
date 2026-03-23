"""Domain plugin registry and helpers for MGVS guidance hooks."""

from __future__ import annotations

from mgvs.domains.algebra import AlgebraDomainPlugin
from mgvs.domains.base import DomainPlugin
from mgvs.domains.number_theory import NumberTheoryDomainPlugin
from mgvs.domains.polynomial import PolynomialDomainPlugin
from mgvs.state.models import ReasoningState


def default_domain_plugins() -> list[DomainPlugin]:
    """Return built-in v1 domain plugins."""

    return [
        AlgebraDomainPlugin(),
        PolynomialDomainPlugin(),
        NumberTheoryDomainPlugin(),
    ]


def active_domain_plugins(state: ReasoningState, plugins: list[DomainPlugin] | None = None) -> list[DomainPlugin]:
    """Filter plugins that match the current state."""

    candidates = plugins or default_domain_plugins()
    return [plugin for plugin in candidates if plugin.matches(state)]
