"""vLLM client integration boundary kept as a future PT/PCT/LSS adapter."""

from mgvs.llm.base import UnifiedLLMClient


class VLLMClient(UnifiedLLMClient):
    """Placeholder vLLM-backed unified PT/PCT/LSS client."""

    def generate_pt(self, prompt: str) -> str:
        _ = prompt
        raise NotImplementedError("vLLM PT endpoint is not implemented in bootstrap")

    def generate_pct(self, prompt: str) -> str:
        _ = prompt
        raise NotImplementedError("vLLM PCT endpoint is not implemented in bootstrap")

    def generate_lss(self, prompt: str) -> str:
        _ = prompt
        raise NotImplementedError("vLLM LSS endpoint is not implemented in bootstrap")
