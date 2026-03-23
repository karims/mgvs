"""Beam ranking and deduplication utilities for bounded exploration."""

from __future__ import annotations

from collections import defaultdict

from mgvs.state.models import ReasoningState
from mgvs.types import StateStatus


def _status_priority(status: StateStatus) -> int:
    """Priority used for deterministic beam ordering."""

    priorities = {
        StateStatus.SOLVED: 4,
        StateStatus.ACTIVE: 3,
        StateStatus.PARAMETRIC: 2,
        StateStatus.NEEDS_REINTERPRETATION: 1,
        StateStatus.CONTRADICTION: 0,
        StateStatus.DEAD_END: 0,
    }
    return priorities.get(status, 0)


def select_beam(states: list[ReasoningState], width: int) -> list[ReasoningState]:
    """Select top-k states by status priority then score."""

    if width <= 0:
        return []

    ranked = sorted(
        states,
        key=lambda state: (
            _status_priority(state.status),
            state.score,
            -len(state.open_goals),
        ),
        reverse=True,
    )
    return ranked[:width]


def deduplicate_states(states: list[ReasoningState]) -> list[ReasoningState]:
    """Deduplicate equivalent states using their `normalized_form` key."""

    keyed: dict[str, list[ReasoningState]] = defaultdict(list)
    passthrough: list[ReasoningState] = []

    for state in states:
        key = state.normalized_form
        if key is None:
            passthrough.append(state)
            continue
        keyed[key].append(state)

    deduped = list(passthrough)
    for candidates in keyed.values():
        best = max(
            candidates,
            key=lambda state: (_status_priority(state.status), state.score),
        )
        deduped.append(best)

    return deduped
