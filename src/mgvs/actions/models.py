"""Structured action schema for bounded, deterministic solver transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Supported bounded action kinds proposed by the planner/LLM."""

    REWRITE = "rewrite"
    SUBSTITUTE = "substitute"
    ELIMINATE = "eliminate"
    FACTOR = "factor"
    EXPAND = "expand"
    INTRODUCE_REPRESENTATION = "introduce_representation"
    HYPOTHESIZE_WITNESS = "hypothesize_witness"
    BIND_WITNESS = "bind_witness"
    DERIVE_CONSTRAINT = "derive_constraint"
    DETECT_SYMMETRY = "detect_symmetry"
    BRANCH = "branch"
    PRUNE = "prune"


@dataclass(frozen=True)
class CandidateAction:
    """Typed bounded action proposal consumed by the transition layer."""

    action_type: ActionType
    title: str
    rationale: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    added_facts: list[str] = field(default_factory=list)
    added_constraints: list[str] = field(default_factory=list)
    branch_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
