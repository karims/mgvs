"""Parsers that map raw LLM text outputs to structured action candidates."""

from mgvs.actions.models import ActionCandidate


def parse_action(text: str) -> ActionCandidate:
    """Convert model text to a placeholder action candidate."""

    return ActionCandidate(action_id="stub-action", description=text.strip(), score=0.0)
