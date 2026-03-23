"""Runtime configuration primitives for MGVS experiments and local runs."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """Minimal configuration container for bootstrap execution paths."""

    seed: int = 0
    max_steps: int = 100


@dataclass(frozen=True)
class VLLMRuntimeConfig:
    """Runtime configuration for OpenAI-compatible vLLM endpoints."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model_name: str = "default"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "VLLMRuntimeConfig":
        """Load vLLM configuration from environment variables."""

        return cls(
            base_url=os.getenv("MGVS_VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("MGVS_VLLM_API_KEY", ""),
            model_name=os.getenv("MGVS_VLLM_MODEL_NAME", "default"),
            temperature=float(os.getenv("MGVS_VLLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MGVS_VLLM_MAX_TOKENS", "512")),
            timeout=float(os.getenv("MGVS_VLLM_TIMEOUT", "30.0")),
        )
