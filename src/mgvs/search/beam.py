"""Beam management primitives for bounded-width candidate exploration."""

from mgvs.state.models import ReasoningState


def select_beam(states: list[ReasoningState], width: int) -> list[ReasoningState]:
    """Return the first `width` states as a deterministic placeholder."""

    return states[:width]
