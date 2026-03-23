"""vLLM client integration boundary kept as a placeholder for future wiring."""

from mgvs.llm.base import LLMClient


class VLLMClient(LLMClient):
    """Placeholder vLLM-backed client interface."""

    def generate(self, prompt: str) -> str:
        _ = prompt
        raise NotImplementedError("vLLM client is not implemented in bootstrap")
