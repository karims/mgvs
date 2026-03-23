"""Phase 11 tests for numeric answer extraction and ambiguity handling."""

import unittest

from mgvs.search.controller import ControllerResult
from mgvs.solve.answering import (
    ANSWER_STATUS_CONFLICTING,
    ANSWER_STATUS_CONSENSUS,
    ANSWER_STATUS_MISSING,
    ANSWER_STATUS_PARAMETRIC,
    ANSWER_STATUS_UNIQUE,
    extract_state_answer,
    select_answer_across_states,
)
from mgvs.solve.runner import _select_best_terminal
from mgvs.state.models import ReasoningState, create_initial_state
from mgvs.types import StateStatus


def _mk_state(status: StateStatus, facts: list[str], score: float, norm: str) -> ReasoningState:
    state = create_initial_state(raw_problem="p", target_type="competition")
    state.status = status
    state.derived_facts = list(facts)
    state.score = score
    state.normalized_form = norm
    return state


class TestAnsweringPhase11(unittest.TestCase):
    """Covers extraction and beam-level answer decision behavior."""

    def test_unique_numeric_answer(self) -> None:
        state = _mk_state(StateStatus.SOLVED, ["x = 7"], 1.0, "s1")
        extracted = extract_state_answer(state)
        self.assertEqual(extracted.answer_status, ANSWER_STATUS_UNIQUE)
        self.assertEqual(extracted.predicted_answer, 7)

    def test_missing_answer(self) -> None:
        state = _mk_state(StateStatus.SOLVED, ["x is positive"], 1.0, "s2")
        extracted = extract_state_answer(state)
        self.assertEqual(extracted.answer_status, ANSWER_STATUS_MISSING)
        self.assertIsNone(extracted.predicted_answer)

    def test_conflicting_terminal_answers(self) -> None:
        s1 = _mk_state(StateStatus.SOLVED, ["x = 2"], 1.0, "a")
        s2 = _mk_state(StateStatus.SOLVED, ["x = 3"], 0.9, "b")
        decision = select_answer_across_states([s1, s2])
        self.assertEqual(decision.answer_status, ANSWER_STATUS_CONFLICTING)
        self.assertIsNone(decision.predicted_answer)

    def test_parametric_branch(self) -> None:
        s1 = _mk_state(StateStatus.PARAMETRIC, ["x = k"], 0.5, "p")
        decision = select_answer_across_states([s1])
        self.assertEqual(decision.answer_status, ANSWER_STATUS_PARAMETRIC)
        self.assertIsNone(decision.predicted_answer)

    def test_consensus_across_multiple_branches(self) -> None:
        s1 = _mk_state(StateStatus.SOLVED, ["x = 11"], 0.6, "c1")
        s2 = _mk_state(StateStatus.SOLVED, ["answer = 11"], 0.7, "c2")
        decision = select_answer_across_states([s1, s2])
        self.assertEqual(decision.answer_status, ANSWER_STATUS_CONSENSUS)
        self.assertEqual(decision.predicted_answer, 11)

    def test_runner_terminal_selection_uses_answer_decision(self) -> None:
        s1 = _mk_state(StateStatus.SOLVED, ["x = 5"], 1.0, "hi_score")
        s2 = _mk_state(StateStatus.SOLVED, ["x = 5"], 0.2, "low_score")
        result = ControllerResult(final_beam=[s2, s1], depth_reached=1, termination_reason="done")
        best, decision = _select_best_terminal(result)
        self.assertEqual(best.normalized_form, "hi_score")
        self.assertEqual(decision.answer_status, ANSWER_STATUS_CONSENSUS)


if __name__ == "__main__":
    unittest.main()
