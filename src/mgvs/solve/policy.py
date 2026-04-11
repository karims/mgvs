"""Deterministic solve-mode routing and fallback policy for runtime-constrained solving."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mgvs.config import SolveModeSettings, SolvePolicyConfig
from mgvs.state.models import ReasoningState
from mgvs.types import StateStatus


class SolveMode(str, Enum):
    """Submission-oriented solve modes."""

    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


@dataclass(frozen=True)
class ModeSelection:
    """Mode choice with traceable reason."""

    mode: SolveMode
    reason: str


@dataclass(frozen=True)
class FallbackDecision:
    """Fallback routing decision with explicit reason."""

    trigger: bool
    fallback_mode: SolveMode | None = None
    reason: str = ""


def mode_settings_for(mode: SolveMode, config: SolvePolicyConfig) -> SolveModeSettings:
    """Resolve mode profile from policy config."""

    if mode == SolveMode.FAST:
        return config.fast
    if mode == SolveMode.DEEP:
        return config.deep
    return config.balanced


def select_solve_mode(
    problem_text: str,
    initial_state: ReasoningState,
    config: SolvePolicyConfig,
    *,
    budget_pressure: float = 0.0,
) -> ModeSelection:
    """Select solve mode using transparent lightweight heuristics."""

    text = problem_text.strip()
    lower_text = text.lower()
    length = len(text)
    tags = set(initial_state.strategy_tags)
    target = initial_state.target_type.lower()

    if budget_pressure >= config.budget_pressure_fast_threshold:
        return ModeSelection(SolveMode.FAST, "budget_pressure")

    hard_signals = (
        length >= config.hard_problem_min_chars
        or any(
            marker in lower_text
            for marker in ("branch", "contradiction", "parametric", "mod", "polynomial", "divisible")
        )
        or "domain:number_theory" in tags
        or "domain:polynomial" in tags
        or target in {"number_theory", "polynomial"}
        or len(initial_state.current_equations) >= 2
    )
    if hard_signals:
        return ModeSelection(SolveMode.DEEP, "high_structural_difficulty")

    if length <= config.easy_problem_max_chars and len(initial_state.current_equations) <= 1:
        return ModeSelection(SolveMode.FAST, "short_low_complexity")

    return ModeSelection(SolveMode.BALANCED, "default_balanced")


def select_fallback(
    *,
    termination_reason: str,
    best_state: ReasoningState,
    budget_pressure: float,
    malformed_output_count: int,
    current_mode: SolveMode,
    config: SolvePolicyConfig,
) -> FallbackDecision:
    """Decide whether an explicit final fallback pass should run."""

    if current_mode == SolveMode.FAST:
        return FallbackDecision(False, None, "already_fast")

    if termination_reason == "no_valid_next_states":
        if best_state.status == StateStatus.ACTIVE:
            return FallbackDecision(True, SolveMode.FAST, "no_valid_branches_survived")
        return FallbackDecision(False, None, "terminal_state_already_reached")

    if termination_reason == "no_useful_progress":
        if best_state.status == StateStatus.ACTIVE:
            return FallbackDecision(True, SolveMode.FAST, "no_valid_branches_survived")
        return FallbackDecision(False, None, "terminal_state_already_reached")

    if malformed_output_count >= config.malformed_retry_fallback_threshold:
        return FallbackDecision(True, SolveMode.FAST, "repeated_malformed_llm_outputs")

    if budget_pressure >= config.budget_pressure_fallback_threshold:
        return FallbackDecision(True, SolveMode.FAST, "budget_nearly_exhausted")

    if best_state.status in {StateStatus.PARAMETRIC, StateStatus.ACTIVE} and budget_pressure >= 0.6:
        return FallbackDecision(True, SolveMode.FAST, "incomplete_near_budget_end")

    return FallbackDecision(False, None, "no_fallback")
