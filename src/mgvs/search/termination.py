"""Termination criteria for ending iterative search safely and deterministically."""


def should_terminate(step: int, max_steps: int) -> bool:
    """Stop when step count reaches the configured limit."""

    return step >= max_steps
