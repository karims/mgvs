"""Termination criteria for the iterative controller loop."""

from __future__ import annotations

from dataclasses import dataclass

from mgvs.state.models import ReasoningState
from mgvs.types import StateStatus

TERMINAL_STATUSES: set[StateStatus] = {
    StateStatus.SOLVED,
    StateStatus.CONTRADICTION,
    StateStatus.DEAD_END,
    StateStatus.PARAMETRIC,
}


@dataclass(frozen=True)
class TerminationDecision:
    """Decision payload describing whether search should stop."""

    should_stop: bool
    reason: str | None = None


def is_terminal_state(state: ReasoningState) -> bool:
    """Return True if state is terminal for further expansion."""

    return state.status in TERMINAL_STATUSES


def is_high_priority_solved(state: ReasoningState) -> bool:
    """Return True when a solved state is marked high-priority."""

    if state.status != StateStatus.SOLVED:
        return False
    tags = set(state.strategy_tags)
    return (
        "high_priority" in tags
        or "high_priority_solved" in tags
        or state.score >= 1.0
    )


def should_terminate(
    *,
    depth: int,
    max_depth: int,
    beam_states: list[ReasoningState],
    has_valid_next_states: bool,
) -> TerminationDecision:
    """Evaluate termination conditions for the controller loop."""

    if any(is_high_priority_solved(state) for state in beam_states):
        return TerminationDecision(should_stop=True, reason="high_priority_solved")
    if depth >= max_depth:
        return TerminationDecision(should_stop=True, reason="max_depth_reached")
    if not has_valid_next_states:
        return TerminationDecision(should_stop=True, reason="no_valid_next_states")
    return TerminationDecision(should_stop=False, reason=None)
