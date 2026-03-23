"""Numeric answer extraction and beam-level answer resolution for MGVS."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mgvs.state.models import ReasoningState
from mgvs.types import StateStatus

ANSWER_STATUS_UNIQUE = "unique_integer"
ANSWER_STATUS_CONSENSUS = "consensus_integer"
ANSWER_STATUS_CONFLICTING = "conflicting_answers"
ANSWER_STATUS_PARAMETRIC = "parametric"
ANSWER_STATUS_MISSING = "missing_answer"


@dataclass(frozen=True)
class StateAnswerCandidate:
    """Answer extraction result for one state."""

    answer_status: str
    predicted_answer: int | None
    candidate_answers: list[int]


@dataclass(frozen=True)
class BeamAnswerDecision:
    """Resolved answer decision across a set of terminal/near-terminal states."""

    predicted_answer: int | None
    answer_status: str
    supporting_state_ids: list[str]
    supporting_trace_count: int


def extract_state_answer(state: ReasoningState) -> StateAnswerCandidate:
    """Extract integer answer candidates from a state using structured fields first."""

    if state.status == StateStatus.PARAMETRIC:
        return StateAnswerCandidate(
            answer_status=ANSWER_STATUS_PARAMETRIC,
            predicted_answer=None,
            candidate_answers=[],
        )

    candidates: list[int] = []

    # 1) Structured witness keys often hold final numeric outputs.
    preferred_keys = ["answer", "final_answer", "numeric_answer", "value"]
    for key in preferred_keys:
        if key in state.witness_parameters:
            parsed = _to_int_if_integral(state.witness_parameters[key])
            if parsed is not None:
                candidates.append(parsed)

    # 2) Derived facts often include equation-like forms (e.g., x = 1).
    for fact in state.derived_facts:
        candidates.extend(_extract_integers_from_equation_rhs(fact))

    # 3) Current equations as a weaker fallback.
    for equation in state.current_equations:
        candidates.extend(_extract_integers_from_equation_rhs(equation))

    # 4) Normalized-form fallback only for explicit assignment-like shapes.
    if state.normalized_form:
        candidates.extend(_extract_assignment_integers(state.normalized_form))

    unique = sorted(set(candidates))
    if len(unique) == 1:
        return StateAnswerCandidate(
            answer_status=ANSWER_STATUS_UNIQUE,
            predicted_answer=unique[0],
            candidate_answers=unique,
        )
    if len(unique) > 1:
        return StateAnswerCandidate(
            answer_status=ANSWER_STATUS_CONFLICTING,
            predicted_answer=None,
            candidate_answers=unique,
        )
    return StateAnswerCandidate(
        answer_status=ANSWER_STATUS_MISSING,
        predicted_answer=None,
        candidate_answers=[],
    )


def select_answer_across_states(states: list[ReasoningState]) -> BeamAnswerDecision:
    """Resolve a robust answer decision across terminal/near-terminal beam states."""

    solved_states = [state for state in states if state.status == StateStatus.SOLVED]
    if not solved_states:
        parametric_states = [state for state in states if state.status == StateStatus.PARAMETRIC]
        if parametric_states:
            return BeamAnswerDecision(
                predicted_answer=None,
                answer_status=ANSWER_STATUS_PARAMETRIC,
                supporting_state_ids=[_state_ref(state, index) for index, state in enumerate(parametric_states)],
                supporting_trace_count=sum(len(state.accepted_steps) for state in parametric_states),
            )
        return BeamAnswerDecision(
            predicted_answer=None,
            answer_status=ANSWER_STATUS_MISSING,
            supporting_state_ids=[],
            supporting_trace_count=0,
        )

    extracted: list[tuple[ReasoningState, StateAnswerCandidate]] = [
        (state, extract_state_answer(state)) for state in solved_states
    ]

    unique_with_answer = [item for item in extracted if item[1].answer_status == ANSWER_STATUS_UNIQUE]
    if not unique_with_answer:
        return BeamAnswerDecision(
            predicted_answer=None,
            answer_status=ANSWER_STATUS_MISSING,
            supporting_state_ids=[_state_ref(state, index) for index, (state, _) in enumerate(extracted)],
            supporting_trace_count=sum(len(state.accepted_steps) for state, _ in extracted),
        )

    grouped: dict[int, list[ReasoningState]] = {}
    for state, candidate in unique_with_answer:
        if candidate.predicted_answer is None:
            continue
        grouped.setdefault(candidate.predicted_answer, []).append(state)

    if len(grouped) == 1:
        answer = next(iter(grouped.keys()))
        supporters = grouped[answer]
        status = ANSWER_STATUS_CONSENSUS if len(supporters) > 1 else ANSWER_STATUS_UNIQUE
        return BeamAnswerDecision(
            predicted_answer=answer,
            answer_status=status,
            supporting_state_ids=[_state_ref(state, index) for index, state in enumerate(supporters)],
            supporting_trace_count=sum(len(state.accepted_steps) for state in supporters),
        )

    # Multiple solved states produce different integer answers.
    all_supporters = [state for states_group in grouped.values() for state in states_group]
    return BeamAnswerDecision(
        predicted_answer=None,
        answer_status=ANSWER_STATUS_CONFLICTING,
        supporting_state_ids=[_state_ref(state, index) for index, state in enumerate(all_supporters)],
        supporting_trace_count=sum(len(state.accepted_steps) for state in all_supporters),
    )


def _state_ref(state: ReasoningState, index: int) -> str:
    """Best-effort stable reference for supporting states."""

    if state.normalized_form:
        return state.normalized_form
    return f"beam_{index}"


def _to_int_if_integral(value: object) -> int | None:
    """Convert value to int only when it represents an integer exactly."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        return None
    return None


def _extract_integers_from_equation_rhs(text: str) -> list[int]:
    """Extract integral numeric tokens from the right-hand side of equations."""

    if "=" not in text:
        return []

    lhs, rhs = text.split("=", 1)
    lhs_clean = lhs.strip().lower()
    rhs_clean = rhs.strip()

    if not re.fullmatch(r"[a-z_][a-z0-9_]*", lhs_clean):
        return []
    parsed = _to_int_if_integral(rhs_clean)
    if parsed is None:
        return []
    return [parsed]


def _extract_all_integers(text: str) -> list[int]:
    """Extract all integer tokens from text."""

    found: list[int] = []
    for token in re.findall(r"[-+]?\d+", text):
        parsed = _to_int_if_integral(token)
        if parsed is not None:
            found.append(parsed)
    return found


def _extract_assignment_integers(text: str) -> list[int]:
    """Extract integers only from explicit assignment-like fragments."""

    matches: list[int] = []
    for token in re.findall(r"(?:answer|final|result|value|x)\s*=\s*([-+]?\d+)", text.lower()):
        parsed = _to_int_if_integral(token)
        if parsed is not None:
            matches.append(parsed)
    if re.fullmatch(r"[-+]?\d+", text.strip()):
        parsed = _to_int_if_integral(text.strip())
        if parsed is not None:
            matches.append(parsed)
    return matches
