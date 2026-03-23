"""Prompt template helpers for search-step action proposal requests."""


def build_action_prompt(context: str) -> str:
    """Build a minimal prompt shell for bootstrap integration."""

    return f"Propose next action for context:\n{context}".strip()
