"""Deterministic local stub LLM backend for offline development and tests."""

from mgvs.llm.base import LLMClient


class StubLLMClient(LLMClient):
    """Simple LLM client that returns a fixed response."""

    def generate(self, prompt: str) -> str:
        _ = prompt
        return "stub action"
