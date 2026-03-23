"""Action application stubs that transform canonical state objects."""

from mgvs.actions.models import ActionCandidate
from mgvs.state.models import ReasoningState


def apply_action(state: ReasoningState, action: ActionCandidate) -> ReasoningState:
    """Return an unchanged state placeholder until real transforms exist."""

    _ = action
    return state
