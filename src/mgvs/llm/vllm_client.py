"""OpenAI-compatible vLLM client for PT/PCT/LSS structured generation."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
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

    def with_overrides(self, *, retries: int | None = None) -> "VLLMClient":
        """Create a copy with runtime overrides while reusing transport."""

        runtime = self._runtime
        if retries is not None:
            runtime = replace(runtime, retries=max(0, int(retries)))
        return VLLMClient(runtime=runtime, transport=self._transport)

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

        attempts = max(1, 1 + int(self._runtime.retries))
        active_prompt = prompt
        last_reason = "unknown"

        for attempt in range(attempts):
            result = self._generate_once(stage=stage, prompt=active_prompt)
            if result is not None:
                return result

            # Retry path for malformed/empty/timeout responses.
            if stage == STAGE_LSS and attempt < attempts - 1:
                active_prompt = self._reduce_lss_candidates(active_prompt)
            last_reason = self._last_failure_reason

        return self._fallback_for_stage(stage, reason=last_reason)

    def _generate_once(self, *, stage: str, prompt: str) -> str | None:
        """Single attempt for structured stage request."""

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
            self._last_failure_reason = "timeout_or_network"
            return None

        content = self._extract_content(response)
        if not content.strip():
            self._last_failure_reason = "empty_response"
            return None

        parsed = parse_structured_json_object(content)
        if parsed:
            return json.dumps(parsed, sort_keys=True)

        self._last_failure_reason = "malformed_json"
        return None

    _last_failure_reason = "unknown"

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

    def _reduce_lss_candidates(self, prompt: str) -> str:
        """Reduce max candidate count in LSS prompt for retry stabilization."""

        payload = parse_structured_json_object(prompt)
        if not payload:
            return prompt

        constraints = payload.get("constraints")
        if not isinstance(constraints, dict):
            return prompt
        raw_max = constraints.get("max_candidates")
        if not isinstance(raw_max, int) or raw_max <= 1:
            return prompt

        reduced = max(1, int(raw_max * float(self._runtime.lss_retry_candidate_decay)))
        if reduced == raw_max:
            reduced = raw_max - 1
        constraints["max_candidates"] = max(1, reduced)
        payload["constraints"] = constraints
        return json.dumps(payload, sort_keys=True)

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
