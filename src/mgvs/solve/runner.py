"""Experiment runner glue for connecting config, search, and verification."""

from mgvs.state.models import ReasoningState, create_initial_state


def run() -> ReasoningState:
    """Run a placeholder solve path and return an initial state."""

    return create_initial_state(
        raw_problem="Placeholder bootstrap problem.",
        target_type="unspecified",
    )
