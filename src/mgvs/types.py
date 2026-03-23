"""Shared type aliases and enums used across MGVS modules."""

from __future__ import annotations

from enum import Enum
from typing import NewType

StateId = NewType("StateId", str)
ActionId = NewType("ActionId", str)


class StateStatus(str, Enum):
    """Lifecycle statuses for a reasoning state."""

    ACTIVE = "active"
    SOLVED = "solved"
    CONTRADICTION = "contradiction"
    DEAD_END = "dead_end"
    PARAMETRIC = "parametric"
    NEEDS_REINTERPRETATION = "needs_reinterpretation"
