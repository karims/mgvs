"""OpenAI-compatible vLLM client for PT/PCT/LSS structured generation."""

from __future__ import annotations

import json
import socket
from typing import Any, Callable
from urllib import error, request

from mgvs.config import VLLMRuntimeConfig
from mgvs.llm.base import LLMRequestOptions, UnifiedLLMClient
from mgvs.llm.parser import parse_structured_json_object
from mgvs.llm.prompts import STAGE_LSS, STAGE_PCT, STAGE_PT, build_stage_system_prompt

Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class VLLMClient(UnifiedLLMClient):
    """vLLM-backed unified PT/PCT/LSS client using chat-completions API."""

    def __init__(
        self,
        *,
        runtime: VLLMRuntimeConfig | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._runtime = runtime or VLLMRuntimeConfig.from_env()
        self._transport = transport or self._default_transport

    @classmethod
    def from_env(cls) -> "VLLMClient":
        """Build a configured client from environment variables."""

        return cls(runtime=VLLMRuntimeConfig.from_env())

    def generate_pt(self, prompt: str) -> str:
        """Generate structured PT output."""

        return self._generate(stage=STAGE_PT, prompt=prompt)

    def generate_pct(self, prompt: str) -> str:
        """Generate structured PCT output."""

        return self._generate(stage=STAGE_PCT, prompt=prompt)

    def generate_lss(self, prompt: str) -> str:
        """Generate structured LSS output."""

        return self._generate(stage=STAGE_LSS, prompt=prompt)

    def _generate(self, *, stage: str, prompt: str) -> str:
        """Issue one structured stage request with robust fallbacks."""

        options = LLMRequestOptions(
            temperature=self._runtime.temperature,
            max_tokens=self._runtime.max_tokens,
            timeout=self._runtime.timeout,
        )
        payload = {
            "model": self._runtime.model_name,
            "messages": [
                {"role": "system", "content": build_stage_system_prompt(stage)},
                {"role": "user", "content": prompt},
            ],
            "temperature": options.temperature,
            "max_tokens": options.max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self._runtime.api_key:
            headers["Authorization"] = f"Bearer {self._runtime.api_key}"

        endpoint = f"{self._runtime.base_url.rstrip('/')}/chat/completions"

        try:
            response = self._transport(endpoint, payload, headers, options.timeout)
        except (TimeoutError, socket.timeout, error.URLError, OSError):
            return self._fallback_for_stage(stage, reason="timeout_or_network")

        content = self._extract_content(response)
        if not content.strip():
            return self._fallback_for_stage(stage, reason="empty_response")

        parsed = parse_structured_json_object(content)
        if parsed:
            return json.dumps(parsed, sort_keys=True)

        return self._fallback_for_stage(stage, reason="malformed_json")

    @staticmethod
    def _default_transport(
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        """Default HTTP transport using urllib for stdlib-only dependency footprint."""

        req = request.Request(
            endpoint,
            method="POST",
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )
        with request.urlopen(req, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        """Extract assistant content from OpenAI-compatible response shape."""

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first = choices[0]
        if not isinstance(first, dict):
            return ""

        message = first.get("message")
        if not isinstance(message, dict):
            return ""

        content = message.get("content", "")
        return str(content)

    @staticmethod
    def _fallback_for_stage(stage: str, *, reason: str) -> str:
        """Return stage-safe fallback JSON so parser layer can proceed."""

        if stage == STAGE_PT:
            payload = {
                "symbolic_objects": {},
                "current_equations": [],
                "domain_constraints": [],
                "global_constraints": [],
                "witness_parameters": {},
                "open_goals": [],
                "metadata": {"fallback_reason": reason},
            }
            return json.dumps(payload, sort_keys=True)

        if stage == STAGE_PCT:
            payload = {
                "strategy_tags": ["llm_fallback"],
                "open_goals": [],
                "added_facts": [],
                "added_constraints": [],
                "metadata": {"fallback_reason": reason},
            }
            return json.dumps(payload, sort_keys=True)

        payload = {
            "actions": [],
            "metadata": {"fallback_reason": reason},
        }
        return json.dumps(payload, sort_keys=True)
