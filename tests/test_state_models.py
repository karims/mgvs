"""Unit tests for the Phase 1 reasoning state model."""

import unittest

from mgvs.state.models import ConstraintRecord, FactRecord, ReasoningState, create_initial_state
from mgvs.state.trace import TraceStep
from mgvs.types import StateStatus


class TestReasoningStateModel(unittest.TestCase):
    """Behavioral checks for state construction and mutation helpers."""

    def test_initial_state_creation(self) -> None:
        state = create_initial_state(raw_problem="Solve x + 1 = 2", target_type="equation")
        self.assertEqual(state.raw_problem, "Solve x + 1 = 2")
        self.assertEqual(state.target_type, "equation")
        self.assertEqual(state.status, StateStatus.ACTIVE)
        self.assertEqual(state.score, 0.0)
        self.assertEqual(state.derived_facts, [])
        self.assertEqual(state.accepted_steps, [])
        self.assertIsNone(state.normalized_form)

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
        self.assertEqual(state.accepted_steps, [step])

    def test_status_transitions(self) -> None:
        state = ReasoningState(raw_problem="p", target_type="proof")

        state.mark_status(StateStatus.PARAMETRIC)
        self.assertEqual(state.status, StateStatus.PARAMETRIC)

        state.mark_status(StateStatus.SOLVED)
        self.assertEqual(state.status, StateStatus.SOLVED)


if __name__ == "__main__":
    unittest.main()
