"""Trace utilities for recording state transitions during search."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TraceStep:
    """A compact accepted transition summary stored on solver state."""

    action: str
    rationale: str
    updates: dict[str, Any]
