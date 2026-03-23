"""Parsers mapping structured PT/PCT/LSS outputs to MGVS objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.state.models import ReasoningState


@dataclass(frozen=True)
class PTUpdate:
    """Parsed PT stage update payload."""

    symbolic_objects: dict[str, Any] = field(default_factory=dict)
    current_equations: list[str] = field(default_factory=list)
    domain_constraints: list[str] = field(default_factory=list)
    global_constraints: list[str] = field(default_factory=list)
    witness_parameters: dict[str, Any] = field(default_factory=dict)
    open_goals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PCTUpdate:
    """Parsed PCT stage update payload."""

    strategy_tags: list[str] = field(default_factory=list)
    open_goals: list[str] = field(default_factory=list)
    added_facts: list[str] = field(default_factory=list)
    added_constraints: list[str] = field(default_factory=list)


def parse_structured_json_object(text: str) -> dict[str, Any]:
    """Parse JSON object with best-effort recovery from mixed model text."""

    stripped = text.strip()
    if not stripped:
        return {}

    # Fast path for valid JSON object text.
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError:
        raw = None
    if isinstance(raw, dict):
        return raw

    # Recover from markdown fences or prefixed/suffixed chatter by slicing
    # the first balanced JSON object region.
    start = stripped.find("{")
    if start == -1:
        return {}

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        ch = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                try:
                    recovered = json.loads(candidate)
                except json.JSONDecodeError:
                    return {}
                return recovered if isinstance(recovered, dict) else {}

    return {}


def _load_json_object(text: str) -> dict[str, Any]:
    """Parse JSON text into an object with safe fallback."""

    return parse_structured_json_object(text)


def _string_list(value: Any) -> list[str]:
    """Convert unknown payload to list[str]."""

    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_payload(value: Any) -> dict[str, Any]:
    """Convert unknown payload to dict[str, Any]."""

    return dict(value) if isinstance(value, dict) else {}


def parse_pt_output(text: str) -> PTUpdate:
    """Parse PT JSON output into structured update fields."""

    obj = _load_json_object(text)
    return PTUpdate(
        symbolic_objects=_dict_payload(obj.get("symbolic_objects")),
        current_equations=_string_list(obj.get("current_equations")),
        domain_constraints=_string_list(obj.get("domain_constraints")),
        global_constraints=_string_list(obj.get("global_constraints")),
        witness_parameters=_dict_payload(obj.get("witness_parameters")),
        open_goals=_string_list(obj.get("open_goals")),
    )


def apply_pt_update(state: ReasoningState, update: PTUpdate) -> ReasoningState:
    """Apply parsed PT updates to a cloned state."""

    next_state = state.clone()
    next_state.symbolic_objects.update(update.symbolic_objects)
    next_state.current_equations.extend(update.current_equations)
    next_state.domain_constraints.extend(update.domain_constraints)
    next_state.global_constraints.extend(update.global_constraints)
    next_state.witness_parameters.update(update.witness_parameters)
    next_state.open_goals.extend(update.open_goals)
    return next_state


def parse_pct_output(text: str) -> PCTUpdate:
    """Parse PCT JSON output into structured update fields."""

    obj = _load_json_object(text)
    return PCTUpdate(
        strategy_tags=_string_list(obj.get("strategy_tags")),
        open_goals=_string_list(obj.get("open_goals")),
        added_facts=_string_list(obj.get("added_facts")),
        added_constraints=_string_list(obj.get("added_constraints")),
    )


def apply_pct_update(state: ReasoningState, update: PCTUpdate) -> ReasoningState:
    """Apply parsed PCT updates to a cloned state."""

    next_state = state.clone()
    next_state.strategy_tags.extend(update.strategy_tags)
    next_state.open_goals.extend(update.open_goals)
    for fact in update.added_facts:
        next_state.add_fact(fact)
    for constraint in update.added_constraints:
        next_state.add_constraint(constraint)
    return next_state


def parse_lss_output(text: str) -> list[CandidateAction]:
    """Parse LSS JSON output into bounded candidate actions."""

    obj = _load_json_object(text)
    raw_actions = obj.get("actions")
    if not isinstance(raw_actions, list):
        return []

    parsed: list[CandidateAction] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue

        action_type_raw = str(item.get("action_type", "")).strip().lower()
        try:
            action_type = ActionType(action_type_raw)
        except ValueError:
            continue

        title = str(item.get("title", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if not title or not rationale:
            continue

        parsed.append(
            CandidateAction(
                action_type=action_type,
                title=title,
                rationale=rationale,
                inputs=_string_list(item.get("inputs")),
                outputs=_string_list(item.get("outputs")),
                added_facts=_string_list(item.get("added_facts")),
                added_constraints=_string_list(item.get("added_constraints")),
                branch_labels=_string_list(item.get("branch_labels")),
                metadata=_dict_payload(item.get("metadata")),
            )
        )
    return parsed


def parse_action(text: str) -> CandidateAction:
    """Backward-compatible helper returning first parsed LSS action."""

    actions = parse_lss_output(text)
    if actions:
        return actions[0]
    return CandidateAction(
        action_type=ActionType.REWRITE,
        title="Parsed fallback action",
        rationale="Fallback parser output",
    )
