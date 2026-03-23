"""Lightweight domain plugin interfaces for MGVS guidance and checks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from mgvs.actions.models import CandidateAction
from mgvs.state.models import ReasoningState


@dataclass(frozen=True)
class DomainValidationResult:
    """Result of optional domain-specific validation."""

    passed: bool
    reason: str = "ok"


class DomainPlugin(Protocol):
    """Small composable plugin contract for domain guidance hooks."""

    name: str

    def matches(self, state: ReasoningState) -> bool:
        """Return whether plugin should activate for the given state."""

    def annotate_state(self, state: ReasoningState) -> None:
        """Mutate state with domain hints such as tags/goals/constraints."""

    def enrich_actions(
        self,
        state: ReasoningState,
        actions: list[CandidateAction],
    ) -> list[CandidateAction]:
        """Return action list enriched with plugin guidance metadata."""

    def validate_action(self, state: ReasoningState, action: CandidateAction) -> DomainValidationResult:
        """Optional domain-specific action checks."""

    def validate_state(self, state: ReasoningState) -> DomainValidationResult:
        """Optional domain-specific state checks."""


class BaseDomainPlugin:
    """Default no-op implementation helpers for domain plugins."""

    name = "base"

    def matches(self, state: ReasoningState) -> bool:
        _ = state
        return False

    def annotate_state(self, state: ReasoningState) -> None:
        _ = state

    def enrich_actions(
        self,
        state: ReasoningState,
        actions: list[CandidateAction],
    ) -> list[CandidateAction]:
        _ = state
        enriched: list[CandidateAction] = []
        for action in actions:
            metadata = dict(action.metadata)
            metadata.setdefault("domain_hints", [])
            hints = metadata["domain_hints"]
            if isinstance(hints, list):
                if self.name not in hints:
                    hints.append(self.name)
                metadata["domain_hints"] = hints
            else:
                metadata["domain_hints"] = [self.name]
            enriched.append(replace(action, metadata=metadata))
        return enriched

    def validate_action(self, state: ReasoningState, action: CandidateAction) -> DomainValidationResult:
        _ = state, action
        return DomainValidationResult(passed=True)

    def validate_state(self, state: ReasoningState) -> DomainValidationResult:
        _ = state
        return DomainValidationResult(passed=True)


def ensure_tag(state: ReasoningState, tag: str) -> None:
    """Append strategy tag once."""

    if tag not in state.strategy_tags:
        state.strategy_tags.append(tag)


def ensure_goal(state: ReasoningState, goal: str) -> None:
    """Append open goal once."""

    if goal not in state.open_goals:
        state.open_goals.append(goal)


def ensure_domain_constraint(state: ReasoningState, constraint: str) -> None:
    """Append domain constraint once."""

    if constraint not in state.domain_constraints:
        state.domain_constraints.append(constraint)
