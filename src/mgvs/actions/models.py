"""Data structures for action proposals produced by search/LLM layers."""

from dataclasses import dataclass


@dataclass
class ActionCandidate:
    """Represents an action candidate and its raw confidence score."""

    action_id: str
    description: str
    score: float = 0.0
