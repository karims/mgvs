"""Deterministic PT/PCT/LSS stub backend for offline integration tests."""

from __future__ import annotations

import json
from typing import Any

from mgvs.actions.models import CandidateAction
from mgvs.llm.base import UnifiedLLMClient
from mgvs.llm.parser import (
    apply_pct_update,
    apply_pt_update,
    parse_lss_output,
    parse_pct_output,
    parse_pt_output,
)
from mgvs.llm.prompts import build_lss_prompt, build_pct_prompt, build_pt_prompt
from mgvs.state.models import ReasoningState


class StubLLMClient(UnifiedLLMClient):
    """Deterministic stage-aware fake backend keyed by problem patterns."""

    def generate_pt(self, prompt: str) -> str:
        payload = _parse_prompt(prompt)
        raw_problem = str(payload.get("input", {}).get("raw_problem", "")).lower()

        if "x + 1 = 2" in raw_problem:
            return json.dumps(
                {
                    "symbolic_objects": {"x": {"kind": "scalar"}},
                    "current_equations": ["x + 1 = 2"],
                    "domain_constraints": ["x in R"],
                    "global_constraints": [],
                    "witness_parameters": {},
                    "open_goals": ["solve for x"],
                }
            )

        return json.dumps(
            {
                "symbolic_objects": {"obj_0": {"kind": "unknown"}},
                "current_equations": [],
                "domain_constraints": [],
                "global_constraints": [],
                "witness_parameters": {},
                "open_goals": ["clarify target structure"],
            }
        )

    def generate_pct(self, prompt: str) -> str:
        payload = _parse_prompt(prompt)
        goals = payload.get("context", {}).get("open_goals", [])

        if isinstance(goals, list) and any("solve for x" in str(goal).lower() for goal in goals):
            return json.dumps(
                {
                    "strategy_tags": ["isolate_variable", "linear_equation"],
                    "open_goals": ["isolate x"],
                    "added_facts": ["equation is linear"],
                    "added_constraints": [],
                }
            )

        return json.dumps(
            {
                "strategy_tags": ["generic_decomposition"],
                "open_goals": ["propose first bounded rewrite"],
                "added_facts": [],
                "added_constraints": [],
            }
        )

    def generate_lss(self, prompt: str) -> str:
        payload = _parse_prompt(prompt)
        context = payload.get("context", {})
        goals = context.get("open_goals", []) if isinstance(context, dict) else []

        if isinstance(goals, list) and any("isolate x" in str(goal).lower() for goal in goals):
            return json.dumps(
                {
                    "actions": [
                        {
                            "action_type": "rewrite",
                            "title": "subtract_one_both_sides",
                            "rationale": "isolate variable by inverse operation",
                            "inputs": ["x + 1 = 2"],
                            "outputs": ["x = 1"],
                            "added_facts": ["x = 1"],
                            "added_constraints": [],
                            "branch_labels": [],
                            "metadata": {"mark_solved": True, "normalized_form": "x=1"},
                        },
                        {
                            "action_type": "derive_constraint",
                            "title": "record_solution_membership",
                            "rationale": "retain domain compatibility",
                            "inputs": ["x = 1"],
                            "outputs": [],
                            "added_facts": [],
                            "added_constraints": ["x in R"],
                            "branch_labels": [],
                            "metadata": {},
                        },
                    ]
                }
            )

        return json.dumps(
            {
                "actions": [
                    {
                        "action_type": "introduce_representation",
                        "title": "normalize_problem_statement",
                        "rationale": "produce canonical placeholder state",
                        "inputs": [],
                        "outputs": ["canonical_placeholder"],
                        "added_facts": [],
                        "added_constraints": [],
                        "branch_labels": [],
                        "metadata": {"normalized_form": "placeholder"},
                    }
                ]
            }
        )


class StubPipelineProposer:
    """Controller-compatible proposer that uses stub LSS generation."""

    def __init__(self, client: UnifiedLLMClient, *, max_candidates: int = 3) -> None:
        self._client = client
        self._max_candidates = max_candidates

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        _ = depth
        prompt = build_lss_prompt(state, max_candidates=self._max_candidates)
        raw = self._client.generate_lss(prompt)
        return parse_lss_output(raw)[: self._max_candidates]


def run_pt_stage(state: ReasoningState, client: UnifiedLLMClient) -> ReasoningState:
    """Apply deterministic PT stage output to state."""

    prompt = build_pt_prompt(raw_problem=state.raw_problem, target_type=state.target_type)
    update = parse_pt_output(client.generate_pt(prompt))
    return apply_pt_update(state, update)


def run_pct_stage(state: ReasoningState, client: UnifiedLLMClient) -> ReasoningState:
    """Apply deterministic PCT stage output to state."""

    prompt = build_pct_prompt(state)
    update = parse_pct_output(client.generate_pct(prompt))
    return apply_pct_update(state, update)


def _parse_prompt(prompt: str) -> dict[str, Any]:
    """Best-effort JSON extraction from structured prompts."""

    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
