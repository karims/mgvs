"""Scoring helpers for action ranking in guided search loops."""

from mgvs.actions.models import ActionCandidate


def score_action(action: ActionCandidate) -> float:
    """Return the bootstrap score field as-is."""

    return action.score
