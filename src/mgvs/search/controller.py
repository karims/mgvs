"""Top-level search controller coordinating propose-apply-verify loops."""

from mgvs.state.models import ReasoningState


def run_search(initial_state: ReasoningState) -> ReasoningState:
    """Bootstrap search controller that returns the input state."""

    return initial_state
