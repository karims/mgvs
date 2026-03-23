"""Parsers that map raw LLM text outputs to structured action candidates."""

from mgvs.actions.models import ActionType, CandidateAction


def parse_action(text: str) -> CandidateAction:
    """Convert model text to a placeholder action candidate."""

    return CandidateAction(
        action_type=ActionType.REWRITE,
        title="Parsed stub action",
        rationale="Default parser placeholder",
        outputs=[text.strip()],
    )
