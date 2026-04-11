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
class GoalRecord:
    """Compact structured goal payload for state-first solver flow."""

    type: str = "find_exact_answer"
    target: str = ""


def _normalize_atom(value: object) -> str:
    """Convert arbitrary value to a compact atomic string."""

    return str(value).strip()


def _dedupe_items(values: list[object]) -> list[str]:
    """Deduplicate short atomic strings while preserving first-seen order."""

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        item = _normalize_atom(value)
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


@dataclass
class ReasoningState:
    """Self-contained canonical state for Markov-style search progression."""

    raw_problem: str
    target_type: str
    problem_id: str = ""
    problem_text: str = ""
    goal: GoalRecord = field(default_factory=GoalRecord)
    objects: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    symbolic_objects: dict[str, Any] = field(default_factory=dict)
    current_equations: list[str] = field(default_factory=list)
    derived_facts: list[str] = field(default_factory=list)
    domain_constraints: list[str] = field(default_factory=list)
    global_constraints: list[str] = field(default_factory=list)
    bounds: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)
    candidate_strategies: list[str] = field(default_factory=list)
    progress_measure: list[str] = field(default_factory=list)
    unknowns_remaining: list[str] = field(default_factory=list)
    answer_candidates: list[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    witness_parameters: dict[str, Any] = field(default_factory=dict)
    strategy_tags: list[str] = field(default_factory=list)
    open_goals: list[str] = field(default_factory=list)
    branch_assignments: list[str] = field(default_factory=list)
    status: StateStatus = StateStatus.ACTIVE
    score: float = 0.0
    normalized_form: str | None = None
    accepted_steps: list[TraceStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Hydrate structured aliases from legacy fields and normalize lists."""

        if isinstance(self.goal, dict):
            self.goal = GoalRecord(
                type=_normalize_atom(self.goal.get("type", "find_exact_answer")) or "find_exact_answer",
                target=_normalize_atom(self.goal.get("target", "")),
            )
        self.normalize_in_place()

    def clone(self) -> "ReasoningState":
        """Create a deep copy so branch updates never alias mutable fields."""

        cloned = deepcopy(self)
        cloned.normalize_in_place()
        return cloned

    def add_fact(self, fact: str | FactRecord) -> None:
        """Append a derived fact to the state."""

        self.derived_facts.append(fact.statement if isinstance(fact, FactRecord) else fact)
        self.normalize_in_place()

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
            self.normalize_in_place()
            return
        self.domain_constraints.append(statement)
        self.normalize_in_place()

    def add_trace_step(self, step: TraceStep) -> None:
        """Record an accepted summarized transition step."""

        self.accepted_steps.append(step)

    def mark_status(self, status: StateStatus) -> None:
        """Update lifecycle status for downstream controller logic."""

        self.status = status

    def normalize_in_place(self) -> "ReasoningState":
        """Normalize structured/legacy fields into one canonical machine-usable view."""

        if not self.problem_text:
            self.problem_text = self.raw_problem
        if not self.raw_problem:
            self.raw_problem = self.problem_text

        if not self.goal.target:
            self.goal.target = self.open_goals[0] if self.open_goals else ""
        if self.status != StateStatus.SOLVED and not self.open_goals and self.goal.target:
            self.open_goals.append(self.goal.target)

        symbol_names = [name for name in self.symbolic_objects.keys()]
        self.objects = _dedupe_items(self.objects + symbol_names)
        self.variables = _dedupe_items(self.variables + symbol_names)

        self.problem_id = _normalize_atom(self.problem_id)
        self.problem_text = _normalize_atom(self.problem_text)
        self.raw_problem = _normalize_atom(self.raw_problem)
        self.target_type = _normalize_atom(self.target_type)
        self.goal.type = _normalize_atom(self.goal.type) or "find_exact_answer"
        self.goal.target = _normalize_atom(self.goal.target)

        self.current_equations = _dedupe_items(self.current_equations)
        self.domain_constraints = _dedupe_items(self.domain_constraints)
        self.global_constraints = _dedupe_items(self.global_constraints)
        self.derived_facts = _dedupe_items(self.derived_facts)
        self.strategy_tags = _dedupe_items(self.strategy_tags)
        self.open_goals = _dedupe_items(self.open_goals)
        self.branch_assignments = _dedupe_items(self.branch_assignments)

        self.relations = _dedupe_items(self.relations + self.current_equations)
        self.domains = _dedupe_items(self.domains)
        self.constraints = _dedupe_items(self.constraints + self.domain_constraints + self.global_constraints)
        self.bounds = _dedupe_items(self.bounds)
        self.invariants = _dedupe_items(self.invariants)
        self.cases = _dedupe_items(self.cases + self.branch_assignments)
        self.candidate_strategies = _dedupe_items(self.candidate_strategies + self.strategy_tags)
        self.progress_measure = _dedupe_items(self.progress_measure)
        self.unknowns_remaining = _dedupe_items(self.unknowns_remaining)
        self.answer_candidates = _dedupe_items(self.answer_candidates)
        self.notes = _dedupe_items(self.notes)

        if self.status != StateStatus.SOLVED and self.goal.target and self.goal.target not in self.open_goals:
            self.open_goals.insert(0, self.goal.target)

        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0
        return self

    def to_structured_dict(self, *, drop_empty: bool = True) -> dict[str, object]:
        """Return a machine-usable structured view of the canonical state."""

        self.normalize_in_place()
        payload: dict[str, object] = {
            "problem_id": self.problem_id,
            "problem_text": self.problem_text,
            "goal": {"type": self.goal.type, "target": self.goal.target},
            "objects": list(self.objects),
            "variables": list(self.variables),
            "domains": list(self.domains),
            "relations": list(self.relations),
            "constraints": list(self.constraints),
            "derived_facts": list(self.derived_facts),
            "bounds": list(self.bounds),
            "invariants": list(self.invariants),
            "cases": list(self.cases),
            "candidate_strategies": list(self.candidate_strategies),
            "progress_measure": list(self.progress_measure),
            "unknowns_remaining": list(self.unknowns_remaining),
            "answer_candidates": list(self.answer_candidates),
            "confidence": float(self.confidence),
            "notes": list(self.notes),
        }
        if not drop_empty:
            return payload
        compact: dict[str, object] = {}
        for key, value in payload.items():
            if value in ("", None, [], {}):
                continue
            if key == "goal" and isinstance(value, dict) and not value.get("target"):
                continue
            compact[key] = value
        return compact


def create_initial_state(
    raw_problem: str,
    target_type: str,
    *,
    problem_id: str = "",
) -> ReasoningState:
    """Build a new empty reasoning state from a raw problem statement."""

    return ReasoningState(
        raw_problem=raw_problem,
        target_type=target_type,
        problem_id=problem_id,
        problem_text=raw_problem,
        goal=GoalRecord(),
    )
