"""Core datamodels describing canonical search state snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from mgvs.state.trace import TraceStep
from mgvs.types import StateStatus


@dataclass(frozen=True)
class FactRecord:
    """Optional structured container for a derived fact."""

    statement: str
    source: str | None = None


@dataclass(frozen=True)
class ConstraintRecord:
    """Optional structured container for a constraint."""

    statement: str
    is_global: bool = False


@dataclass
class ReasoningState:
    """Self-contained canonical state for Markov-style search progression."""

    raw_problem: str
    target_type: str
    symbolic_objects: dict[str, Any] = field(default_factory=dict)
    current_equations: list[str] = field(default_factory=list)
    derived_facts: list[str] = field(default_factory=list)
    domain_constraints: list[str] = field(default_factory=list)
    global_constraints: list[str] = field(default_factory=list)
    witness_parameters: dict[str, Any] = field(default_factory=dict)
    strategy_tags: list[str] = field(default_factory=list)
    open_goals: list[str] = field(default_factory=list)
    branch_assignments: list[str] = field(default_factory=list)
    status: StateStatus = StateStatus.ACTIVE
    score: float = 0.0
    normalized_form: str | None = None
    accepted_steps: list[TraceStep] = field(default_factory=list)

    def clone(self) -> "ReasoningState":
        """Create a deep copy so branch updates never alias mutable fields."""

        return deepcopy(self)

    def add_fact(self, fact: str | FactRecord) -> None:
        """Append a derived fact to the state."""

        self.derived_facts.append(fact.statement if isinstance(fact, FactRecord) else fact)

    def add_constraint(
        self,
        constraint: str | ConstraintRecord,
        *,
        global_scope: bool = False,
    ) -> None:
        """Append a constraint to domain or global scope."""

        if isinstance(constraint, ConstraintRecord):
            statement = constraint.statement
            scope = constraint.is_global
        else:
            statement = constraint
            scope = global_scope

        if scope:
            self.global_constraints.append(statement)
            return
        self.domain_constraints.append(statement)

    def add_trace_step(self, step: TraceStep) -> None:
        """Record an accepted summarized transition step."""

        self.accepted_steps.append(step)

    def mark_status(self, status: StateStatus) -> None:
        """Update lifecycle status for downstream controller logic."""

        self.status = status


def create_initial_state(raw_problem: str, target_type: str) -> ReasoningState:
    """Build a new empty reasoning state from a raw problem statement."""

    return ReasoningState(raw_problem=raw_problem, target_type=target_type)
