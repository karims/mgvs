"""Unit tests for the canonical reasoning state model."""

import unittest

from mgvs.state.models import ConstraintRecord, FactRecord, GoalRecord, ReasoningState, create_initial_state
from mgvs.state.trace import TraceStep
from mgvs.types import StateStatus


class TestReasoningStateModel(unittest.TestCase):
    """Behavioral checks for state construction and mutation helpers."""

    def test_initial_state_creation(self) -> None:
        state = create_initial_state(raw_problem="Solve x + 1 = 2", target_type="equation")
        self.assertEqual(state.raw_problem, "Solve x + 1 = 2")
        self.assertEqual(state.problem_text, "Solve x + 1 = 2")
        self.assertEqual(state.target_type, "equation")
        self.assertEqual(state.problem_id, "")
        self.assertEqual(state.goal, GoalRecord())
        self.assertEqual(state.status, StateStatus.ACTIVE)
        self.assertEqual(state.score, 0.0)
        self.assertEqual(state.derived_facts, [])
        self.assertEqual(state.accepted_steps, [])
        self.assertIsNone(state.normalized_form)
        self.assertEqual(state.objects, [])
        self.assertEqual(state.variables, [])
        self.assertEqual(state.relations, [])
        self.assertEqual(state.constraints, [])
        self.assertEqual(state.to_structured_dict()["problem_text"], "Solve x + 1 = 2")

    def test_clone_is_deep_copy(self) -> None:
        original = create_initial_state(raw_problem="p", target_type="proof")
        original.symbolic_objects["x"] = {"kind": "var"}
        original.add_fact("x is integer")
        original.add_constraint("x >= 0")

        cloned = original.clone()
        cloned.symbolic_objects["x"]["kind"] = "updated"
        cloned.add_fact("x <= 5")

        self.assertEqual(original.symbolic_objects["x"]["kind"], "var")
        self.assertEqual(original.derived_facts, ["x is integer"])
        self.assertEqual(cloned.derived_facts, ["x is integer", "x <= 5"])
        self.assertEqual(cloned.objects, ["x"])
        self.assertEqual(original.objects, ["x"])

    def test_append_helpers_for_facts_constraints_trace(self) -> None:
        state = create_initial_state(raw_problem="p", target_type="proof")
        step = TraceStep(
            action="substitute",
            rationale="Use known equality",
            updates={"added_fact": "a=b"},
        )

        state.add_fact(FactRecord(statement="a=b", source="local rule"))
        state.add_constraint("a >= 0")
        state.add_constraint(ConstraintRecord(statement="global invariant", is_global=True))
        state.add_trace_step(step)

        self.assertEqual(state.derived_facts, ["a=b"])
        self.assertEqual(state.domain_constraints, ["a >= 0"])
        self.assertEqual(state.global_constraints, ["global invariant"])
        self.assertEqual(state.constraints, ["a >= 0", "global invariant"])
        self.assertEqual(state.accepted_steps, [step])

    def test_status_transitions(self) -> None:
        state = ReasoningState(raw_problem="p", target_type="proof")

        state.mark_status(StateStatus.PARAMETRIC)
        self.assertEqual(state.status, StateStatus.PARAMETRIC)

        state.mark_status(StateStatus.SOLVED)
        self.assertEqual(state.status, StateStatus.SOLVED)

    def test_structured_normalization_deduplicates_and_syncs_aliases(self) -> None:
        state = ReasoningState(
            raw_problem="Count rectangles",
            target_type="competition",
            problem_id="r1",
            problem_text="",
            goal=GoalRecord(target="find K mod 10^5"),
            objects=["rectangle", "rectangle", ""],
            variables=["K", "K"],
            domains=["integer", "integer"],
            relations=["p = 2(a+b)", "p = 2(a+b)"],
            constraints=["p <= 2000", ""],
            derived_facts=["perimeter is even", "perimeter is even"],
            bounds=["4 <= p", "4 <= p"],
            invariants=["p is even", "p is even"],
            cases=["case_a", "case_a"],
            candidate_strategies=["count_perimeters", "count_perimeters"],
            progress_measure=["remaining bounds", "remaining bounds"],
            unknowns_remaining=["K", "K"],
            answer_candidates=["999", "999"],
            confidence=1.5,
            notes=["short note", "short note"],
            symbolic_objects={"K": {"kind": "entity"}},
            current_equations=["p = 2(a+b)", "p = 2(a+b)"],
            domain_constraints=["p <= 2000", "p <= 2000"],
            strategy_tags=["count_perimeters", "count_perimeters"],
            open_goals=["find K mod 10^5", "find K mod 10^5"],
            branch_assignments=["case_a", "case_a"],
        )

        structured = state.to_structured_dict(drop_empty=False)

        self.assertEqual(state.problem_text, "Count rectangles")
        self.assertEqual(state.objects, ["rectangle", "K"])
        self.assertEqual(state.variables, ["K"])
        self.assertEqual(state.relations, ["p = 2(a+b)"])
        self.assertEqual(state.constraints, ["p <= 2000"])
        self.assertEqual(state.cases, ["case_a"])
        self.assertEqual(state.candidate_strategies, ["count_perimeters"])
        self.assertEqual(state.unknowns_remaining, ["K"])
        self.assertEqual(state.answer_candidates, ["999"])
        self.assertEqual(state.confidence, 1.0)
        self.assertEqual(structured["goal"], {"type": "find_exact_answer", "target": "find K mod 10^5"})
        self.assertEqual(structured["notes"], ["short note"])


if __name__ == "__main__":
    unittest.main()
