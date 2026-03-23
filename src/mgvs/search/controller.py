"""Top-level controller coordinating propose, verify, apply, and prune loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from mgvs.actions.apply import apply_action
from mgvs.actions.models import CandidateAction
from mgvs.search.beam import deduplicate_states, select_beam
from mgvs.search.termination import is_terminal_state, should_terminate
from mgvs.state.models import ReasoningState


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

    while True:
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

        for state in beam:
            if is_terminal_state(state):
                continue

            candidates = proposer.propose(state, depth)
            candidates_count += len(candidates)

            for action in candidates:
                if not state_verifier.is_action_valid(state, action):
                    continue
                accepted_count += 1

                for child in apply_action(state, action):
                    canonical = state_canonicalizer.canonicalize(child)
                    if state_verifier.is_state_valid(canonical):
                        next_states.append(canonical)

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
