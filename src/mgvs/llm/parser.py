"""Parsers mapping structured PT/PCT/LSS outputs to MGVS objects."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from mgvs.actions.models import ActionType, CandidateAction
from mgvs.llm.prompts import STAGE_ENDGAME, STAGE_LSS, STAGE_PCT, STAGE_PT
from mgvs.state.models import ReasoningState


def _parser_debug_enabled() -> bool:
    """Return whether parser debug logging is enabled."""

    return os.environ.get("MGVS_DEBUG_PARSER") == "1"


def _parser_debug_print(message: str) -> None:
    """Print debug message when parser debug logging is enabled."""

    if _parser_debug_enabled():
        print(message)


def _stage_parse_error(stage: str, reason: str, *, details: str = "") -> None:
    """Emit a compact stage-specific parse failure message."""

    suffix = f" details={details}" if details else ""
    _parser_debug_print(f"[parser][{stage}] failure_reason={reason}{suffix}")


@dataclass(frozen=True)
class PTUpdate:
    """Parsed PT stage update payload."""

    symbolic_objects: dict[str, Any] = field(default_factory=dict)
    current_equations: list[str] = field(default_factory=list)
    domain_constraints: list[str] = field(default_factory=list)
    global_constraints: list[str] = field(default_factory=list)
    witness_parameters: dict[str, Any] = field(default_factory=dict)
    open_goals: list[str] = field(default_factory=list)
    derived_facts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PCTUpdate:
    """Parsed PCT stage update payload."""

    strategy_tags: list[str] = field(default_factory=list)
    open_goals: list[str] = field(default_factory=list)
    candidate_equations: list[str] = field(default_factory=list)
    answer_candidate: int | None = None


@dataclass(frozen=True)
class EndgameSolveOutput:
    """Parsed endgame stage answer payload."""

    answer: int | None = None
    ready: bool = False
    confidence: str = "low"
    justification: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)


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


def validate_stage_payload(stage: str, obj: dict[str, Any]) -> str | None:
    """Return a stage-specific schema failure reason, or None when acceptable."""

    if not isinstance(obj, dict) or not obj:
        return "invalid_json"

    if stage == STAGE_PT:
        required_any = {
            "objects",
            "variables",
            "domains",
            "relations",
            "constraints",
            "goal",
            "unknowns_remaining",
            "symbolic_objects",
            "current_equations",
            "domain_constraints",
            "global_constraints",
            "open_goals",
        }
        return None if any(key in obj for key in required_any) else "missing_required_fields"

    if stage == STAGE_PCT:
        required_any = {"strategy_tags", "open_goals", "candidate_equations", "answer_candidate"}
        return None if any(key in obj for key in required_any) else "missing_required_fields"

    if stage == STAGE_LSS:
        if "actions" not in obj:
            return "missing_required_fields"
        return None if isinstance(obj.get("actions"), list) else "schema_mismatch"

    if stage == STAGE_ENDGAME:
        if "ready" not in obj:
            return "missing_required_fields"
        return None if isinstance(obj.get("ready"), bool) else "schema_mismatch"

    return None


def _string_list(value: Any) -> list[str]:
    """Convert unknown payload to list[str]."""

    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_present(obj: dict[str, Any], keys: list[str]) -> Any:
    """Return the first present key from object or None."""

    for key in keys:
        if key in obj:
            return obj[key]
    return None


def _dict_payload(value: Any) -> dict[str, Any]:
    """Convert unknown payload to dict[str, Any]."""

    return dict(value) if isinstance(value, dict) else {}


def _merge_unique_strings(*groups: list[str]) -> list[str]:
    """Merge string groups while preserving first-seen order."""

    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def parse_pt_output(text: str) -> PTUpdate:
    """Parse PT JSON output into structured update fields."""

    obj = _load_json_object(text)
    payload_error = validate_stage_payload(STAGE_PT, obj)
    if payload_error is not None:
        _stage_parse_error(STAGE_PT, payload_error)
        return PTUpdate()
    objects = _string_list(_first_present(obj, ["objects", "entities"]))
    variables = _string_list(_first_present(obj, ["variables"]))
    unknowns_remaining = _string_list(_first_present(obj, ["unknowns_remaining", "unknowns"]))
    target = str(_first_present(obj, ["goal", "target"]) or "").strip()
    relations = _string_list(_first_present(obj, ["relations", "current_equations", "equations"]))
    domains = _string_list(_first_present(obj, ["domains", "domain_constraints"]))
    flat_constraints = _string_list(_first_present(obj, ["constraints"]))

    if objects or variables or unknowns_remaining or target or relations or domains or flat_constraints:
        entity_names = _merge_unique_strings(objects, variables, unknowns_remaining)
        return PTUpdate(
            symbolic_objects={name: {"kind": "entity"} for name in entity_names},
            current_equations=relations,
            domain_constraints=_merge_unique_strings(domains, flat_constraints),
            global_constraints=[],
            witness_parameters={},
            open_goals=[target] if target else [],
            derived_facts=[],
        )

    constraints_obj = obj.get("constraints")
    domain_constraints: list[str] = []
    global_constraints: list[str] = []
    if isinstance(constraints_obj, dict):
        domain_constraints = _string_list(_first_present(constraints_obj, ["domain", "local"]))
        global_constraints = _string_list(_first_present(constraints_obj, ["global"]))
    else:
        domain_constraints = _string_list(_first_present(obj, ["domain_constraints"]))
        global_constraints = _string_list(_first_present(obj, ["global_constraints"]))

    open_goals = _string_list(_first_present(obj, ["open_goals", "targets", "focus_goals"]))
    derived_facts = _string_list(_first_present(obj, ["facts", "derived_facts"]))

    return PTUpdate(
        symbolic_objects=_dict_payload(obj.get("symbolic_objects")),
        current_equations=_string_list(_first_present(obj, ["current_equations", "equations"])),
        domain_constraints=domain_constraints,
        global_constraints=global_constraints,
        witness_parameters=_dict_payload(_first_present(obj, ["witness_parameters", "witness"])),
        open_goals=open_goals,
        derived_facts=derived_facts,
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
    for fact in update.derived_facts:
        next_state.add_fact(fact)
    next_state.normalize_in_place()
    return next_state


def parse_pct_output(text: str) -> PCTUpdate:
    """Parse PCT JSON output into structured update fields."""

    obj = _load_json_object(text)
    payload_error = validate_stage_payload(STAGE_PCT, obj)
    if payload_error is not None:
        _stage_parse_error(STAGE_PCT, payload_error)
        return PCTUpdate()

    answer_candidate = _int_or_none(obj.get("answer_candidate"))
    if obj.get("answer_candidate") is not None and answer_candidate is None:
        _stage_parse_error(STAGE_PCT, "schema_mismatch", details="answer_candidate_not_integer")
    return PCTUpdate(
        strategy_tags=_string_list(obj.get("strategy_tags")),
        open_goals=_string_list(obj.get("open_goals")),
        candidate_equations=_string_list(obj.get("candidate_equations")),
        answer_candidate=answer_candidate,
    )


def apply_pct_update(state: ReasoningState, update: PCTUpdate) -> ReasoningState:
    """Apply parsed PCT updates to a cloned state."""

    next_state = state.clone()
    next_state.strategy_tags.extend(update.strategy_tags)
    next_state.open_goals.extend(update.open_goals)
    next_state.current_equations.extend(update.candidate_equations)
    if update.answer_candidate is not None:
        next_state.add_fact(f"answer_candidate = {update.answer_candidate}")
        next_state.answer_candidates.append(str(update.answer_candidate))
    next_state.normalize_in_place()
    return next_state


def _int_or_none(value: Any) -> int | None:
    """Convert payload value to int when it is clearly integer-like."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+", stripped):
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def _looks_generic_advice(text: str) -> bool:
    """Return whether text reads like advice instead of a state update."""

    lowered = text.strip().lower()
    if not lowered:
        return False
    generic_prefixes = (
        "consider ",
        "try ",
        "use ",
        "look for ",
        "analyze ",
        "simplify ",
        "eliminate ",
        "substitute ",
        "rewrite ",
    )
    generic_phrases = (
        "a good next step",
        "the next step",
        "it may help",
        "should help",
        "strategy",
        "tactic",
        "reason about",
    )
    if lowered.startswith(generic_prefixes):
        return True
    return any(phrase in lowered for phrase in generic_phrases)


def parse_lss_output(text: str) -> list[CandidateAction]:
    """Parse LSS JSON output into bounded candidate actions."""

    obj = _load_json_object(text)
    payload_error = validate_stage_payload(STAGE_LSS, obj)
    if payload_error is not None:
        _stage_parse_error(STAGE_LSS, payload_error)
    if os.environ.get("MGVS_DEBUG_PARSER") == "1":
        print("\n===== PARSE_LSS_OUTPUT INPUT START =====")
        print(text)
        print("===== PARSE_LSS_OUTPUT INPUT END =====")
        print(
            "parse_lss_output obj keys:",
            list(obj.keys()) if isinstance(obj, dict) else obj,
        )

    raw_actions = obj.get("actions")
    if not isinstance(raw_actions, list):
        if os.environ.get("MGVS_DEBUG_PARSER") == "1":
            print("parse_lss_output: raw_actions is not a list ->", raw_actions)
        return []

    parsed: list[CandidateAction] = []
    for item in raw_actions[:2]:
        if not isinstance(item, dict):
            if os.environ.get("MGVS_DEBUG_PARSER") == "1":
                print("parse_lss_output: skipping non-dict item ->", item)
            continue

        action_type_raw = str(
            _first_present(item, ["action_type", "type", "action"]) or ""
        ).strip().lower()
        try:
            action_type = ActionType(action_type_raw)
        except ValueError:
            _stage_parse_error(STAGE_LSS, "schema_mismatch", details=f"invalid_action_type:{action_type_raw}")
            if os.environ.get("MGVS_DEBUG_PARSER") == "1":
                print("parse_lss_output: invalid action_type ->", action_type_raw, "item:", item)
            continue

        title = str(_first_present(item, ["title", "name"]) or "").strip()
        rationale = str(_first_present(item, ["rationale", "why"]) or "").strip()
        if not title:
            _stage_parse_error(STAGE_LSS, "missing_required_fields", details="title")
            if os.environ.get("MGVS_DEBUG_PARSER") == "1":
                print("parse_lss_output: missing title ->", item)
            continue

        added_facts = _string_list(_first_present(item, ["added_facts", "facts"]))
        added_constraints = _string_list(_first_present(item, ["added_constraints", "constraints"]))
        if action_type not in {ActionType.BRANCH, ActionType.PRUNE} and not added_facts and not added_constraints:
            _stage_parse_error(STAGE_LSS, "empty_transition", details=title)
            continue
        if any(_looks_generic_advice(value) for value in [title, *added_facts, *added_constraints]):
            _stage_parse_error(STAGE_LSS, "schema_mismatch", details=f"generic_advice:{title}")
            continue

        parsed.append(
            CandidateAction(
                action_type=action_type,
                title=title,
                rationale=rationale or "unspecified rationale",
                inputs=_string_list(_first_present(item, ["inputs"])),
                outputs=_string_list(_first_present(item, ["outputs"])),
                added_facts=added_facts,
                added_constraints=added_constraints,
                branch_labels=_string_list(_first_present(item, ["branch_labels", "branches"])),
                metadata=_dict_payload(item.get("metadata")),
            )
        )

    if os.environ.get("MGVS_DEBUG_PARSER") == "1":
        print("parse_lss_output: parsed action count =", len(parsed))
        for index, action in enumerate(parsed):
            print(f"  action[{index}] =", action)
    return parsed


def parse_endgame_solve_output(text: str) -> EndgameSolveOutput:
    """Parse endgame JSON output into a compact answer payload."""

    obj = _load_json_object(text)
    payload_error = validate_stage_payload(STAGE_ENDGAME, obj)
    if payload_error is not None:
        _stage_parse_error(STAGE_ENDGAME, payload_error)
        return EndgameSolveOutput()
    answer = _int_or_none(obj.get("answer"))
    confidence = str(obj.get("confidence", "low") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    ready_raw = obj.get("ready")
    ready = ready_raw if isinstance(ready_raw, bool) else (answer is not None)
    missing_requirements = _string_list(obj.get("missing_requirements"))
    if obj.get("answer") is not None and answer is None:
        _stage_parse_error(STAGE_ENDGAME, "schema_mismatch", details="answer_not_integer")
    if not ready and answer is not None:
        _stage_parse_error(STAGE_ENDGAME, "schema_mismatch", details="answer_present_when_not_ready")
        answer = None
    if missing_requirements and answer is not None:
        _stage_parse_error(STAGE_ENDGAME, "schema_mismatch", details="answer_present_with_missing_requirements")
        answer = None
        ready = False
    return EndgameSolveOutput(
        answer=answer,
        ready=ready,
        confidence=confidence,
        justification=_string_list(obj.get("justification")),
        missing_requirements=missing_requirements,
    )


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
