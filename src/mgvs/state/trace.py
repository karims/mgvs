"""Trace utilities for recording state transitions during search."""

from dataclasses import dataclass


@dataclass
class TraceStep:
    """A single trace step linking prior and next state ids."""

    from_state: str
    to_state: str
    action: str
