"""Top-level controller coordinating propose, verify, apply, and prune loops."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Callable, Protocol

from mgvs.actions.apply import apply_action
from mgvs.actions.models import CandidateAction
from mgvs.search.beam import deduplicate_states, select_beam
from mgvs.search.termination import is_terminal_state, should_terminate
from mgvs.state.models import ReasoningState
from mgvs.types import StateStatus


class ActionProposer(Protocol):
    """Protocol for action proposal backends."""

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        """Return bounded candidate actions for the given state."""


class StateVerifier(Protocol):
    """Protocol for transition and state validity checks."""

    def is_action_valid(self, state: ReasoningState, action: CandidateAction) -> bool:
        """Return True if action may be applied to the state."""

    def is_state_valid(self, state: ReasoningState) -> bool:
        """Return True if state passes verification hooks."""


class StateCanonicalizer(Protocol):
    """Protocol for canonical state normalization."""

    def canonicalize(self, state: ReasoningState) -> ReasoningState:
        """Return a canonicalized state for deduplication and ranking."""


@dataclass(frozen=True)
class IterationSummary:
    """Logging-friendly per-depth search statistics."""

    depth: int
    candidate_actions: int
    accepted_actions: int
    next_states: int
    kept_after_beam: int


@dataclass(frozen=True)
class ControllerConfig:
    """Configuration for generic controller loop behavior."""

    max_depth: int = 8
    beam_width: int = 4
    max_wall_time_s: float = 20.0
    candidate_cap_per_state: int = 3
    time_fn: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class ControllerResult:
    """Result payload for controller loop execution."""

    final_beam: list[ReasoningState]
    depth_reached: int
    termination_reason: str
    iteration_summaries: list[IterationSummary] = field(default_factory=list)

    @property
    def best_state(self) -> ReasoningState:
        """Return highest-scoring state from final beam."""

        return max(self.final_beam, key=lambda state: state.score)


class AlwaysPassVerifier:
    """Default verifier that accepts all actions and states."""

    def is_action_valid(self, state: ReasoningState, action: CandidateAction) -> bool:
        _ = state, action
        return True

    def is_state_valid(self, state: ReasoningState) -> bool:
        _ = state
        return True


class IdentityCanonicalizer:
    """Default canonicalizer that returns state unchanged."""

    def canonicalize(self, state: ReasoningState) -> ReasoningState:
        return state


def _debug_runtime_enabled() -> bool:
    """Return whether controller-level debug tracing is enabled."""

    return os.environ.get("MGVS_DEBUG_RUNTIME") == "1" or os.environ.get("MGVS_DEBUG_LLM") == "1"


def _debug_runtime_print(message: str) -> None:
    """Print controller debug message when enabled."""

    if _debug_runtime_enabled():
        print(message)


def run_search(
    initial_state: ReasoningState,
    proposer: ActionProposer,
    *,
    verifier: StateVerifier | None = None,
    canonicalizer: StateCanonicalizer | None = None,
    config: ControllerConfig | None = None,
) -> ControllerResult:
    """Run the generic search loop over a bounded beam."""

    cfg = config or ControllerConfig()
    state_verifier = verifier or AlwaysPassVerifier()
    state_canonicalizer = canonicalizer or IdentityCanonicalizer()

    beam = [state_canonicalizer.canonicalize(initial_state.clone())]
    summaries: list[IterationSummary] = []
    depth = 0
    start_time = cfg.time_fn()

    while True:
        if cfg.max_wall_time_s > 0 and (cfg.time_fn() - start_time) >= cfg.max_wall_time_s:
            _mark_budget_exhausted(beam)
            return ControllerResult(
                final_beam=beam,
                depth_reached=depth,
                termination_reason="budget_exhausted",
                iteration_summaries=summaries,
            )

        precheck = should_terminate(
            depth=depth,
            max_depth=cfg.max_depth,
            beam_states=beam,
            has_valid_next_states=True,
        )
        if precheck.should_stop:
            return ControllerResult(
                final_beam=beam,
                depth_reached=depth,
                termination_reason=precheck.reason or "terminated",
                iteration_summaries=summaries,
            )

        candidates_count = 0
        accepted_count = 0
        next_states: list[ReasoningState] = []
        budget_exhausted = False

        for state in beam:
            if cfg.max_wall_time_s > 0 and (cfg.time_fn() - start_time) >= cfg.max_wall_time_s:
                budget_exhausted = True
                break
            if is_terminal_state(state):
                continue

            candidates = proposer.propose(state, depth)
            if cfg.candidate_cap_per_state > 0:
                candidates = candidates[: cfg.candidate_cap_per_state]
            candidates_count += len(candidates)

            for action in candidates:
                if cfg.max_wall_time_s > 0 and (cfg.time_fn() - start_time) >= cfg.max_wall_time_s:
                    budget_exhausted = True
                    break
                if not state_verifier.is_action_valid(state, action):
                    rejection = getattr(state_verifier, "consume_last_action_rejection", lambda: None)()
                    if rejection:
                        _debug_runtime_print(
                            "[controller] "
                            f"{rejection.get('layer', 'local_verifier_reject')} "
                            f"title={rejection.get('title', '')!r} "
                            f"action_type={rejection.get('action_type', '')} "
                            f"reason={rejection.get('reason', '')} "
                            f"details={rejection.get('details', {})!r}"
                        )
                    continue
                accepted_count += 1

                children = apply_action(state, action)
                if not children:
                    _debug_runtime_print(
                        "[controller] state_transition_reject "
                        f"title={action.title!r} action_type={action.action_type.value} "
                        "reason=no_child_states_from_apply_action"
                    )
                for child in children:
                    canonical = state_canonicalizer.canonicalize(child)
                    if state_verifier.is_state_valid(canonical):
                        next_states.append(canonical)
                    else:
                        rejection = getattr(state_verifier, "consume_last_state_rejection", lambda: None)()
                        _debug_runtime_print(
                            "[controller] state_transition_reject "
                            f"title={action.title!r} action_type={action.action_type.value} "
                            f"reason={(rejection or {}).get('reason', 'invalid_child_state')} "
                            f"details={(rejection or {}).get('details', {})!r}"
                        )
            if budget_exhausted:
                break

        deduped = deduplicate_states(next_states)
        kept = select_beam(deduped, cfg.beam_width)

        summaries.append(
            IterationSummary(
                depth=depth,
                candidate_actions=candidates_count,
                accepted_actions=accepted_count,
                next_states=len(next_states),
                kept_after_beam=len(kept),
            )
        )
        if accepted_count > 0 and len(next_states) == 0:
            _debug_runtime_print("accepted candidate(s) produced no next states; see rejection logs above")

        if budget_exhausted:
            _mark_budget_exhausted(kept if kept else beam)
            return ControllerResult(
                final_beam=kept if kept else beam,
                depth_reached=depth + 1,
                termination_reason="budget_exhausted",
                iteration_summaries=summaries,
            )

        decision = should_terminate(
            depth=depth + 1,
            max_depth=cfg.max_depth,
            beam_states=kept,
            has_valid_next_states=bool(kept),
        )
        if decision.should_stop:
            return ControllerResult(
                final_beam=kept if kept else beam,
                depth_reached=depth + 1,
                termination_reason=decision.reason or "terminated",
                iteration_summaries=summaries,
            )

        beam = kept
        depth += 1


def _mark_budget_exhausted(states: list[ReasoningState]) -> None:
    """Mark active states as dead-end when runtime budget is exhausted."""

    for state in states:
        if state.status == StateStatus.ACTIVE:
            state.status = StateStatus.DEAD_END
            if "budget_exhausted" not in state.strategy_tags:
                state.strategy_tags.append("budget_exhausted")
