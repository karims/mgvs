"""Phase 7 tests for OpenAI-compatible vLLM client integration."""

import json
import unittest
from unittest.mock import patch

from mgvs.config import VLLMRuntimeConfig
from mgvs.llm.parser import (
    parse_endgame_solve_output,
    parse_lss_output,
    parse_pct_output,
    parse_pt_output,
    parse_structured_json_object,
)
from mgvs.llm.prompts import build_endgame_solve_prompt
from mgvs.llm.vllm_client import VLLMClient


class TestVLLMClientPhase7(unittest.TestCase):
    """Validates structured generation and error fallback behavior."""

    def test_extract_content_prefers_content(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"ok": 1}',
                        "reasoning_content": '{"fallback": 2}',
                        "reasoning": '{"fallback": 3}',
                    }
                }
            ]
        }

        self.assertEqual(VLLMClient._extract_content(response), '{"ok": 1}')

    def test_extract_content_uses_reasoning_content_when_content_empty(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "   ",
                        "reasoning_content": '{"ok": 2}',
                        "reasoning": '{"fallback": 3}',
                    }
                }
            ]
        }

        self.assertEqual(VLLMClient._extract_content(response), '{"ok": 2}')

    def test_extract_content_uses_reasoning_when_other_fields_empty(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": " ",
                        "reasoning": '{"ok": 3}',
                    }
                }
            ]
        }

        self.assertEqual(VLLMClient._extract_content(response), '{"ok": 3}')

    def test_extract_content_returns_empty_when_no_fields_present(self) -> None:
        response = {"choices": [{"message": {}}]}

        self.assertEqual(VLLMClient._extract_content(response), "")

    def test_extract_content_uses_reasoning_object_payload(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": {
                            "constraints": ["x+1=2"],
                            "entities": ["x"],
                            "target": "solve x",
                        },
                    }
                }
            ]
        }

        extracted = VLLMClient._extract_content(response)
        self.assertIn('"entities": ["x"]', extracted)

    def test_extract_content_uses_reasoning_content_block_list(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": [
                            {"type": "text", "text": '{"strategy_tags":["tag_a"]}'},
                        ],
                    }
                }
            ]
        }

        self.assertEqual(VLLMClient._extract_content(response), '{"strategy_tags":["tag_a"]}')

    def test_structured_stage_prefers_reasoning_content_over_content(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "long free-form essay",
                        "reasoning_content": '{"actions":[{"action_type":"rewrite","title":"ok"}]}',
                        "reasoning": '{"fallback": 3}',
                    }
                }
            ]
        }

        field, content, used_lower_priority = VLLMClient._extract_message_text(response, stage="lss")
        self.assertEqual(field, "reasoning_content")
        self.assertEqual(content, '{"actions":[{"action_type":"rewrite","title":"ok"}]}')
        self.assertFalse(used_lower_priority)

    def test_structured_stage_uses_reasoning_when_reasoning_content_missing(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "free-form essay",
                        "reasoning": '{"strategy_tags":["tag_a"]}',
                    }
                }
            ]
        }

        field, content, used_lower_priority = VLLMClient._extract_message_text(response, stage="pct")
        self.assertEqual(field, "reasoning")
        self.assertEqual(content, '{"strategy_tags":["tag_a"]}')
        self.assertTrue(used_lower_priority)

    def test_structured_stage_uses_content_when_reasoning_fields_missing(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": 42}',
                    }
                }
            ]
        }

        field, content, used_lower_priority = VLLMClient._extract_message_text(response, stage="endgame")
        self.assertEqual(field, "content")
        self.assertEqual(content, '{"answer": 42}')
        self.assertTrue(used_lower_priority)

    def test_generate_pt_success(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, payload, headers, timeout
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "entities": ["x"],
                                    "target": "solve for x",
                                    "constraints": ["x+1=2"],
                                }
                            )
                        }
                    }
                ]
            }

        client = VLLMClient(
            runtime=VLLMRuntimeConfig(model_name="stub"),
            transport=transport,
        )
        output = client.generate_pt("prompt")
        parsed = parse_pt_output(output)

        self.assertIn("x", parsed.symbolic_objects)
        self.assertIn("x+1=2", parsed.domain_constraints)

    def test_generate_endgame_success(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, headers, timeout
            self.assertEqual(payload["max_tokens"], 384)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "ready": True,
                                    "answer": 50,
                                    "confidence": "high",
                                    "justification": ["Reduced state isolates a unique count."],
                                    "missing_requirements": [],
                                }
                            )
                        }
                    }
                ]
            }

        prompt = build_endgame_solve_prompt(
            raw_problem="Find K.",
            pt_target="find K",
            pt_constraints=["distinct perimeters"],
            current_equations=["K <= 998"],
            derived_facts=["K is uniquely determined"],
            open_goals=["confirm final count"],
            strategy_tags=["counting"],
            trace_summary=["derive bound"],
        )
        client = VLLMClient(
            runtime=VLLMRuntimeConfig(model_name="stub", endgame_max_tokens=384),
            transport=transport,
        )
        output = client.generate_endgame(prompt)
        parsed = parse_endgame_solve_output(output)

        self.assertEqual(parsed.answer, 50)
        self.assertTrue(parsed.ready)
        self.assertEqual(parsed.confidence, "high")
        self.assertEqual(parsed.justification, ["Reduced state isolates a unique count."])

    def test_partial_json_recovery_from_wrapped_content(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, payload, headers, timeout
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Result:\n```json\n{\"strategy_tags\":[\"tag_a\"],\"open_goals\":[\"g1\"]}\n```"
                        }
                    }
                ]
            }

        client = VLLMClient(runtime=VLLMRuntimeConfig(model_name="stub"), transport=transport)
        output = client.generate_pct("prompt")
        parsed = parse_pct_output(output)

        self.assertEqual(parsed.strategy_tags, ["tag_a"])
        self.assertEqual(parsed.open_goals, ["g1"])

    def test_empty_response_fallback(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, payload, headers, timeout
            return {"choices": [{"message": {"content": "   "}}]}

        client = VLLMClient(runtime=VLLMRuntimeConfig(model_name="stub"), transport=transport)
        output = client.generate_lss("prompt")

        self.assertEqual(parse_lss_output(output), [])

    def test_finish_reason_length_without_valid_json_falls_back(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, payload, headers, timeout
            return {
                "choices": [
                    {
                        "message": {"content": '{"actions": ['},
                        "finish_reason": "length",
                    }
                ]
            }

        client = VLLMClient(runtime=VLLMRuntimeConfig(model_name="stub"), transport=transport)
        output = client.generate_lss("prompt")

        self.assertEqual(parse_lss_output(output), [])

    def test_timeout_fallback(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, payload, headers, timeout
            raise TimeoutError("timeout")

        client = VLLMClient(runtime=VLLMRuntimeConfig(model_name="stub"), transport=transport)
        output = client.generate_pct("prompt")
        parsed = parse_pct_output(output)

        self.assertEqual(parsed.strategy_tags, ["llm_fallback"])

    def test_endgame_timeout_fallback(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, payload, headers, timeout
            raise TimeoutError("timeout")

        client = VLLMClient(runtime=VLLMRuntimeConfig(model_name="stub"), transport=transport)
        output = client.generate_endgame("prompt")
        parsed = parse_endgame_solve_output(output)

        self.assertIsNone(parsed.answer)
        self.assertFalse(parsed.ready)
        self.assertEqual(parsed.confidence, "low")
        self.assertEqual(parsed.justification, [])

    def test_endgame_schema_mismatch_falls_back(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, payload, headers, timeout
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": 17,
                                    "confidence": "high",
                                    "justification": ["missing ready field"],
                                }
                            )
                        }
                    }
                ]
            }

        client = VLLMClient(runtime=VLLMRuntimeConfig(model_name="stub"), transport=transport)
        output = client.generate_endgame("prompt")
        parsed = parse_endgame_solve_output(output)

        self.assertIsNone(parsed.answer)
        self.assertFalse(parsed.ready)
        self.assertIn("missing_required_fields", parsed.missing_requirements)

    def test_malformed_response_shape_fallback(self) -> None:
        def transport(endpoint, payload, headers, timeout):
            _ = endpoint, payload, headers, timeout
            return {"id": "no-choices"}

        client = VLLMClient(runtime=VLLMRuntimeConfig(model_name="stub"), transport=transport)
        output = client.generate_pt("prompt")
        parsed = parse_pt_output(output)

        self.assertEqual(parsed.current_equations, [])

    def test_runtime_config_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MGVS_VLLM_BASE_URL": "http://localhost:9000/v1",
                "MGVS_VLLM_API_KEY": "k",
                "MGVS_VLLM_MODEL_NAME": "m",
                "MGVS_VLLM_TEMPERATURE": "0.2",
                "MGVS_VLLM_MAX_TOKENS": "256",
                "MGVS_VLLM_PT_MAX_TOKENS": "640",
                "MGVS_VLLM_PCT_MAX_TOKENS": "320",
                "MGVS_VLLM_LSS_MAX_TOKENS": "192",
                "MGVS_VLLM_ENDGAME_MAX_TOKENS": "384",
                "MGVS_VLLM_TIMEOUT": "11",
                "MGVS_VLLM_PT_RETRIES": "2",
                "MGVS_VLLM_PCT_RETRIES": "2",
                "MGVS_VLLM_LSS_RETRIES": "2",
            },
            clear=False,
        ):
            cfg = VLLMRuntimeConfig.from_env()

        self.assertEqual(cfg.base_url, "http://localhost:9000/v1")
        self.assertEqual(cfg.api_key, "k")
        self.assertEqual(cfg.model_name, "m")
        self.assertEqual(cfg.temperature, 0.2)
        self.assertEqual(cfg.max_tokens, 256)
        self.assertEqual(cfg.pt_max_tokens, 640)
        self.assertEqual(cfg.pct_max_tokens, 320)
        self.assertEqual(cfg.lss_max_tokens, 192)
        self.assertEqual(cfg.endgame_max_tokens, 384)
        self.assertEqual(cfg.timeout, 11.0)
        self.assertEqual(cfg.pt_retries, 2)
        self.assertEqual(cfg.pct_retries, 2)
        self.assertEqual(cfg.lss_retries, 2)

    def test_with_overrides_updates_endgame_token_cap(self) -> None:
        client = VLLMClient(runtime=VLLMRuntimeConfig())
        overridden = client.with_overrides(endgame_max_tokens=384)
        self.assertEqual(overridden._runtime.endgame_max_tokens, 384)

    def test_parser_recovery_helper(self) -> None:
        wrapped = "prefix {\"actions\":[{\"action_type\":\"rewrite\"}]} suffix"
        parsed = parse_structured_json_object(wrapped)
        self.assertIn("actions", parsed)


if __name__ == "__main__":
    unittest.main()
