"""OpenAI-compatible vLLM client for PT/PCT/LSS structured generation."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import replace
from typing import Any, Callable
from urllib import error, request

from mgvs.config import VLLMRuntimeConfig
from mgvs.llm.base import LLMRequestOptions, UnifiedLLMClient
from mgvs.llm.parser import parse_structured_json_object, validate_stage_payload
from mgvs.llm.prompts import STAGE_ENDGAME, STAGE_LSS, STAGE_PCT, STAGE_PT, build_stage_system_prompt

Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]
_STRUCTURED_STAGES = {STAGE_PT, STAGE_PCT, STAGE_LSS, STAGE_ENDGAME}
_PHASE1_TRACE_STAGES = {STAGE_PT, STAGE_PCT, STAGE_LSS}


def _debug_enabled() -> bool:
    """Return whether verbose LLM debug logging is enabled."""

    return os.environ.get("MGVS_DEBUG_LLM") == "1"


def _debug_print(message: str) -> None:
    """Print debug message when LLM debug logging is enabled."""

    if _debug_enabled():
        print(message)


def _phase1_trace_enabled() -> bool:
    """Return whether temporary Phase 1 free-text trace mode is enabled."""

    return os.environ.get("MGVS_PHASE1_TRACE") == "1"


def _redacted_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return headers safe for debug printing."""

    redacted = dict(headers)
    if "Authorization" in redacted:
        redacted["Authorization"] = "Bearer <redacted>"
    return redacted


def _normalize_message_field(raw: Any) -> str:
    """Normalize assistant message field content to text safely."""

    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in ("text", "content", "reasoning", "reasoning_content"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value)
                    break
        if parts:
            return "\n".join(parts)
        try:
            return json.dumps(raw)
        except (TypeError, ValueError):
            return str(raw)
    if isinstance(raw, dict):
        for key in ("text", "content", "reasoning", "reasoning_content"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        try:
            return json.dumps(raw)
        except (TypeError, ValueError):
            return str(raw)
    return str(raw)


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

    def with_overrides(
        self,
        *,
        retries: int | None = None,
        pt_retries: int | None = None,
        pct_retries: int | None = None,
        lss_retries: int | None = None,
        pt_max_tokens: int | None = None,
        pct_max_tokens: int | None = None,
        lss_max_tokens: int | None = None,
        endgame_max_tokens: int | None = None,
        debug_single_path: bool | None = None,
    ) -> "VLLMClient":
        """Create a copy with runtime overrides while reusing transport."""

        runtime = self._runtime
        if retries is not None:
            runtime = replace(runtime, retries=max(0, int(retries)))
        if pt_retries is not None:
            runtime = replace(runtime, pt_retries=max(0, int(pt_retries)))
        if pct_retries is not None:
            runtime = replace(runtime, pct_retries=max(0, int(pct_retries)))
        if lss_retries is not None:
            runtime = replace(runtime, lss_retries=max(0, int(lss_retries)))
        if pt_max_tokens is not None:
            runtime = replace(runtime, pt_max_tokens=max(1, int(pt_max_tokens)))
        if pct_max_tokens is not None:
            runtime = replace(runtime, pct_max_tokens=max(1, int(pct_max_tokens)))
        if lss_max_tokens is not None:
            runtime = replace(runtime, lss_max_tokens=max(1, int(lss_max_tokens)))
        if endgame_max_tokens is not None:
            runtime = replace(runtime, endgame_max_tokens=max(1, int(endgame_max_tokens)))
        if debug_single_path is not None:
            runtime = replace(runtime, debug_single_path=bool(debug_single_path))
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

    def generate_endgame(self, prompt: str) -> str:
        """Generate structured endgame solve output."""

        return self._generate(stage=STAGE_ENDGAME, prompt=prompt)

    def _generate(self, *, stage: str, prompt: str) -> str:
        """Issue one structured stage request with robust fallbacks."""

        retry_cap = min(max(0, int(self._runtime.retries)), self._stage_retry_cap(stage))
        attempts = max(1, 1 + retry_cap)
        active_prompt = prompt
        last_reason = "unknown"

        for attempt in range(attempts):
            result = self._generate_once(stage=stage, prompt=active_prompt, attempt_index=attempt, total_attempts=attempts)
            if result is not None:
                return result

            # Retry path for malformed/empty/timeout responses.
            if stage == STAGE_LSS and attempt < attempts - 1:
                active_prompt = self._reduce_lss_candidates(active_prompt)
            last_reason = self._last_failure_reason

        return self._fallback_for_stage(stage, reason=last_reason)

    def _generate_once(
        self,
        *,
        stage: str,
        prompt: str,
        attempt_index: int = 0,
        total_attempts: int = 1,
    ) -> str | None:
        """Single attempt for structured stage request."""

        options = LLMRequestOptions(
            temperature=self._runtime.temperature,
            max_tokens=self._stage_max_tokens(stage),
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

        if os.environ.get("MGVS_DEBUG_LLM") == "1":
            print(f"\n===== GENERATE_ONCE {stage.upper()} START =====")
            print(f"[{stage}] attempt={attempt_index + 1}/{total_attempts}")
            print(f"[{stage}] endpoint={endpoint}")
            print(f"[{stage}] model={self._runtime.model_name}")
            print(
                f"[{stage}] options="
                f"temperature={options.temperature} max_tokens={options.max_tokens} timeout={options.timeout}"
            )
            print(f"[{stage}] headers={_redacted_headers(headers)}")
            print(f"[{stage}] payload_keys={list(payload.keys())}")
            print(f"[{stage}] system_prompt_start")
            print(payload["messages"][0]["content"])
            print(f"[{stage}] system_prompt_end")
            print(f"[{stage}] user_prompt_start")
            print(prompt)
            print(f"[{stage}] user_prompt_end")

        try:
            response = self._transport(endpoint, payload, headers, options.timeout)
        except (TimeoutError, socket.timeout, error.URLError, OSError):
            self._last_failure_reason = "timeout_or_network"
            if os.environ.get("MGVS_DEBUG_LLM") == "1":
                print(f"[{stage}] failure_reason=timeout_or_network")
                print(f"===== GENERATE_ONCE {stage.upper()} END =====\n")
            return None

        if os.environ.get("MGVS_DEBUG_LLM") == "1":
            print("\n===== FULL RESPONSE OBJECT START =====")
            print(response)
            print("===== FULL RESPONSE OBJECT END =====\n")
            print(
                f"[{stage}] response_top_level_keys="
                f"{list(response.keys()) if isinstance(response, dict) else response}"
            )

        selected_field, content, used_lower_priority_field = self._extract_message_text(response, stage=stage)
        finish_reason = self._extract_finish_reason(response)
        if os.environ.get("MGVS_DEBUG_LLM") == "1":
            print(f"[{stage}] selected_response_field={selected_field}")
            print(f"[{stage}] used_lower_priority_field={used_lower_priority_field}")
            print(f"\n=== {stage.upper()} RAW OUTPUT ===")
            print(f"===== RAW {stage.upper()} CONTENT START =====")
            print(content)
            print(f"===== RAW {stage.upper()} CONTENT END =====\n")
            print(f"[{stage}] raw_content_length={len(content)}")
            print(f"[{stage}] raw_content_repr={content!r}")
            print(f"[{stage}] finish_reason={finish_reason}")

        if not content.strip():
            self._last_failure_reason = "empty_response"
            if os.environ.get("MGVS_DEBUG_LLM") == "1":
                print(
                    f"[{stage}] attempt_summary retry_index={attempt_index} "
                    f"selected_response_field={selected_field} finish_reason={finish_reason} parsed_success=False"
                )
                print(f"[{stage}] failure_reason=empty_response")
                print(f"===== GENERATE_ONCE {stage.upper()} END =====\n")
            return None

        if _phase1_trace_enabled() and stage in _PHASE1_TRACE_STAGES:
            # PHASE1_TRACE: Temporary passthrough to preserve readable free-text
            # stage traces for PT/PCT/LSS debugging without strict JSON gating.
            if os.environ.get("MGVS_DEBUG_LLM") == "1":
                print(f"[{stage}] phase1_trace_passthrough=true")
                print(
                    f"[{stage}] attempt_summary retry_index={attempt_index} "
                    f"selected_response_field={selected_field} finish_reason={finish_reason} parsed_success=skipped"
                )
                print(f"===== GENERATE_ONCE {stage.upper()} END =====\n")
            self._last_failure_reason = "none"
            return content

        parsed = parse_structured_json_object(content)
        parsed_success = bool(parsed)
        schema_error = validate_stage_payload(stage, parsed) if parsed_success else "invalid_json"
        if os.environ.get("MGVS_DEBUG_LLM") == "1":
            print(
                f"[{stage}] parsed_json_keys:",
                list(parsed.keys()) if isinstance(parsed, dict) else parsed,
            )
            print(
                f"[{stage}] attempt_summary retry_index={attempt_index} "
                f"selected_response_field={selected_field} finish_reason={finish_reason} "
                f"parsed_success={parsed_success}"
            )
            print(f"[{stage}] schema_error={schema_error}")
        if parsed and schema_error is None:
            if os.environ.get("MGVS_DEBUG_LLM") == "1":
                print(f"[{stage}] parse_status=success")
                print(f"===== GENERATE_ONCE {stage.upper()} END =====\n")
            return json.dumps(parsed, sort_keys=True)

        if parsed and schema_error is not None:
            self._last_failure_reason = schema_error
        else:
            self._last_failure_reason = "truncated_output" if finish_reason == "length" else "malformed_json"
        if os.environ.get("MGVS_DEBUG_LLM") == "1":
            if finish_reason == "length":
                print(f"[{stage}] finish_reason_length_detected")
            print(f"[{stage}] failure_reason={self._last_failure_reason}")
            print(f"===== GENERATE_ONCE {stage.upper()} END =====\n")
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
    def _field_order_for_stage(stage: str | None = None) -> tuple[str, ...]:
        """Return response-field precedence for the given stage."""

        if stage in _STRUCTURED_STAGES:
            return ("reasoning_content", "reasoning", "content")
        return ("content", "reasoning_content", "reasoning")

    @staticmethod
    def _extract_message_text(response: dict[str, Any], *, stage: str | None = None) -> tuple[str, str, bool]:
        """Extract assistant text and the source field from response shape."""

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return "missing", "", False

        first = choices[0]
        if not isinstance(first, dict):
            return "missing", "", False

        message = first.get("message")
        if not isinstance(message, dict):
            return "missing", "", False

        field_order = VLLMClient._field_order_for_stage(stage)
        for index, field_name in enumerate(field_order):
            text = _normalize_message_field(message.get(field_name, ""))
            if text.strip():
                return field_name, text, index > 0
        return "missing", "", False

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        """Extract assistant content from OpenAI-compatible response shape."""

        _, content, _ = VLLMClient._extract_message_text(response)
        return content

    @staticmethod
    def _extract_finish_reason(response: dict[str, Any]) -> str:
        """Extract finish_reason from the first choice when available."""

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        finish_reason = first.get("finish_reason", "")
        return str(finish_reason) if finish_reason is not None else ""

    def _stage_max_tokens(self, stage: str) -> int:
        """Return stage-specific max_tokens with safe fallback."""

        if stage == STAGE_PT:
            return max(1, int(self._runtime.pt_max_tokens))
        if stage == STAGE_PCT:
            return max(1, int(self._runtime.pct_max_tokens))
        if stage == STAGE_LSS:
            return max(1, int(self._runtime.lss_max_tokens))
        if stage == STAGE_ENDGAME:
            return max(1, int(self._runtime.endgame_max_tokens))
        return max(1, int(self._runtime.max_tokens))

    def _stage_retry_cap(self, stage: str) -> int:
        """Return stage-specific retry cap with safe fallback."""

        if stage == STAGE_PT:
            return max(0, int(self._runtime.pt_retries))
        if stage == STAGE_PCT:
            return max(0, int(self._runtime.pct_retries))
        if stage == STAGE_LSS:
            return max(0, int(self._runtime.lss_retries))
        return max(0, int(self._runtime.retries))

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
                "metadata": {"fallback_reason": reason, "stage": stage},
            }
            return json.dumps(payload, sort_keys=True)

        if stage == STAGE_PCT:
            payload = {
                "strategy_tags": ["llm_fallback"],
                "open_goals": [],
                "added_facts": [],
                "added_constraints": [],
                "metadata": {"fallback_reason": reason, "stage": stage},
            }
            return json.dumps(payload, sort_keys=True)

        if stage == STAGE_ENDGAME:
            payload = {
                "answer": None,
                "ready": False,
                "confidence": "low",
                "justification": [],
                "missing_requirements": [reason],
                "metadata": {"fallback_reason": reason, "stage": stage},
            }
            return json.dumps(payload, sort_keys=True)

        payload = {
            "actions": [],
            "metadata": {"fallback_reason": reason, "stage": stage},
        }
        return json.dumps(payload, sort_keys=True)
