"""Base interfaces for LLM providers used in action proposal stages."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for text-generation backends used by MGVS."""

    def generate(self, prompt: str) -> str:
        """Generate raw model output for a prompt."""

        ...
