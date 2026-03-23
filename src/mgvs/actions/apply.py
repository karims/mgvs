"""Deterministic application of structured actions to reasoning state."""

from __future__ import annotations

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.actions.scoring import ScoreConfig, score_action_delta
from mgvs.state.models import ReasoningState
from mgvs.state.trace import TraceStep
from mgvs.types import StateStatus


def _resolve_prune_status(action: CandidateAction) -> StateStatus:
    """Map prune metadata to the requested terminal status."""

    raw = str(action.metadata.get("prune_status", "dead_end")).strip().lower()
    if raw == StateStatus.CONTRADICTION.value:
        return StateStatus.CONTRADICTION
    return StateStatus.DEAD_END


def _apply_common_updates(state: ReasoningState, action: CandidateAction) -> None:
    """Apply generic action updates shared across non-branching branches."""

    for fact in action.added_facts:
        state.add_fact(fact)

    for constraint in action.added_constraints:
        state.add_constraint(constraint)

    if action.outputs:
        state.symbolic_objects["last_outputs"] = list(action.outputs)

    if "normalized_form" in action.metadata:
        state.normalized_form = str(action.metadata["normalized_form"])

    if action.metadata.get("mark_parametric", False):
        state.mark_status(StateStatus.PARAMETRIC)
    if action.metadata.get("mark_needs_reinterpretation", False):
        state.mark_status(StateStatus.NEEDS_REINTERPRETATION)
    if action.metadata.get("mark_solved", False):
        state.mark_status(StateStatus.SOLVED)


def _append_trace(state: ReasoningState, action: CandidateAction, *, branch_label: str | None) -> None:
    """Store a compact accepted trace summary for the transition."""

    updates: dict[str, object] = {
        "inputs": list(action.inputs),
        "outputs": list(action.outputs),
        "added_facts": list(action.added_facts),
        "added_constraints": list(action.added_constraints),
    }
    if branch_label is not None:
        updates["branch_label"] = branch_label

    state.add_trace_step(
        TraceStep(
            action=action.action_type.value,
            rationale=action.rationale,
            updates=updates,
        )
    )


def _apply_score(
    state: ReasoningState,
    *,
    action: CandidateAction,
    previous_status: StateStatus,
    next_status: StateStatus,
    branch_fanout: int,
    score_config: ScoreConfig | None,
) -> None:
    """Update score from a deterministic heuristic delta."""

    state.score += score_action_delta(
        action_type=action.action_type,
        previous_status=previous_status,
        next_status=next_status,
        added_facts_count=len(action.added_facts),
        branch_fanout=branch_fanout,
        config=score_config,
    )


def apply_action(
    state: ReasoningState,
    action: CandidateAction,
    *,
    score_config: ScoreConfig | None = None,
) -> list[ReasoningState]:
    """Apply a bounded action and return successor state(s)."""

    if action.action_type == ActionType.BRANCH:
        labels = action.branch_labels or ["branch_0"]
        children: list[ReasoningState] = []

        for label in labels:
            child = state.clone()
            previous_status = child.status
            child.branch_assignments.append(label)
            _apply_common_updates(child, action)
            _append_trace(child, action, branch_label=label)
            _apply_score(
                child,
                action=action,
                previous_status=previous_status,
                next_status=child.status,
                branch_fanout=len(labels),
                score_config=score_config,
            )
            children.append(child)
        return children

    updated = state.clone()
    previous_status = updated.status

    if action.action_type == ActionType.PRUNE:
        updated.mark_status(_resolve_prune_status(action))

    _apply_common_updates(updated, action)
    _append_trace(updated, action, branch_label=None)
    _apply_score(
        updated,
        action=action,
        previous_status=previous_status,
        next_status=updated.status,
        branch_fanout=1,
        score_config=score_config,
    )
    return [updated]
