"""Core datamodels describing canonical search state snapshots."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningState:
    """Represents a single canonical reasoning state in the search."""

    state_id: str
    payload: dict[str, Any] = field(default_factory=dict)
