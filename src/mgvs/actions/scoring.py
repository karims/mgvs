"""Simple configurable scoring heuristics for action transitions."""

from __future__ import annotations

from dataclasses import dataclass

from mgvs.actions.models import ActionType
from mgvs.types import StateStatus


@dataclass(frozen=True)
class ScoreConfig:
    """Lightweight knobs controlling state-score deltas."""

    fact_reward: float = 0.25
    solved_bonus: float = 1.0
    prune_penalty: float = 0.8
    contradiction_penalty: float = 1.2
    dead_end_penalty: float = 0.6
    branch_fanout_penalty: float = 0.1


def score_action_delta(
    *,
    action_type: ActionType,
    previous_status: StateStatus,
    next_status: StateStatus,
    added_facts_count: int,
    branch_fanout: int = 1,
    config: ScoreConfig | None = None,
) -> float:
    """Compute a deterministic delta for state score updates."""

    cfg = config or ScoreConfig()
    delta = cfg.fact_reward * max(added_facts_count, 0)

    if previous_status != StateStatus.SOLVED and next_status == StateStatus.SOLVED:
        delta += cfg.solved_bonus

    if action_type == ActionType.PRUNE:
        delta -= cfg.prune_penalty

    if next_status == StateStatus.CONTRADICTION:
        delta -= cfg.contradiction_penalty
    elif next_status == StateStatus.DEAD_END:
        delta -= cfg.dead_end_penalty

    if action_type == ActionType.BRANCH:
        delta -= cfg.branch_fanout_penalty * max(branch_fanout - 1, 0)

    return delta
