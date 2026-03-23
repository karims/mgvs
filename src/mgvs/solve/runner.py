"""Experiment runner glue for connecting config, search, and verification."""

from mgvs.state.models import ReasoningState


def run() -> ReasoningState:
    """Run a placeholder solve path and return an initial state."""

    return ReasoningState(state_id="initial")
