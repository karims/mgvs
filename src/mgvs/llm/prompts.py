"""Prompt builders for PT, PCT, and LSS structured-output stages."""

from __future__ import annotations

import json

from mgvs.llm.base import DEFAULT_PCT_MAX_TACTICS, LSSContractVersion, PCTContractVersion, PTContractVersion
from mgvs.state.models import ReasoningState

STAGE_PT = "pt"
STAGE_PCT = "pct"
STAGE_LSS = "lss"


def _json_block(payload: dict[str, object]) -> str:
    """Render deterministic pretty JSON for prompt context blocks."""

    return json.dumps(payload, indent=2, sort_keys=True)


def _pt_entities(state: ReasoningState) -> list[str]:
    """Extract PT-style entity names from the current state."""

    return sorted(str(name) for name in state.symbolic_objects.keys())


def _pt_constraints(state: ReasoningState) -> list[str]:
    """Extract PT-style constraints from the current state."""

    return list(state.domain_constraints) + list(state.global_constraints)


def _pt_target(state: ReasoningState) -> str:
    """Extract PT-style target from the current state."""

    return state.open_goals[0] if state.open_goals else ""


def build_pt_prompt(raw_problem: str, target_type: str) -> str:
    """Build PT prompt contract for extracting structured problem state."""

    contract = {
        "contract": PTContractVersion,
        "task": "problem_translation",
        "input": {
            "raw_problem": raw_problem,
            "target_type": target_type,
        },
        "output_schema": {
            "entities": ["entity"],
            "target": "short target",
            "constraints": ["constraint"],
        },
        "example_output": {
            "entities": ["x"],
            "target": "solve for x",
            "constraints": ["x + 1 = 2"],
        },
        "instructions": [
            "Return JSON only.",
            "Do not include markdown fences.",
            "Do not include derivations.",
            "Do not include explanations.",
            "Do not include prose outside the JSON fields.",
            "Keep the object lightweight and concise.",
        ],
    }
    return _json_block(contract)


def build_pct_prompt(state: ReasoningState, *, max_tactics: int = DEFAULT_PCT_MAX_TACTICS) -> str:
    """Build PCT prompt contract for strategy and goal proposal."""

    context = {
        "raw_problem": state.raw_problem,
        "target_type": state.target_type,
        "pt_entities": _pt_entities(state),
        "pt_constraints": _pt_constraints(state),
        "pt_target": _pt_target(state),
        "current_equations": state.current_equations,
        "derived_facts": state.derived_facts,
        "open_goals": state.open_goals,
        "strategy_tags": state.strategy_tags,
    }
    contract = {
        "contract": PCTContractVersion,
        "task": "concept_tactic_proposal",
        "constraints": {"max_tactics": max_tactics},
        "context": context,
        "output_schema": {
            "strategy_tags": ["actionable_tag"],
            "open_goals": ["short_goal"],
            "candidate_equations": ["short_equation"],
            "answer_candidate": None,
        },
        "example_output": {
            "strategy_tags": ["substitute", "eliminate"],
            "open_goals": ["isolate x", "reduce to one equation"],
            "candidate_equations": ["x + y = 10"],
            "answer_candidate": None,
        },
        "instructions": [
            "Return JSON only.",
            "Do not include markdown fences.",
            "Do not include prose explanation.",
            "Do not include a full derivation.",
            "Keep the object small and concise.",
            "Use only these top-level keys: strategy_tags, open_goals, candidate_equations, answer_candidate.",
            "If no integer answer candidate is available, set answer_candidate to null.",
        ],
    }
    return _json_block(contract)


def build_lss_prompt(state: ReasoningState, max_candidates: int) -> str:
    """Build LSS prompt contract for bounded candidate action synthesis."""

    context = {
        "raw_problem": state.raw_problem,
        "status": state.status.value,
        "target_type": state.target_type,
        "pt_entities": _pt_entities(state),
        "pt_constraints": _pt_constraints(state),
        "pt_target": _pt_target(state),
        "current_equations": state.current_equations,
        "open_goals": state.open_goals,
        "strategy_tags": state.strategy_tags,
        "derived_facts": state.derived_facts,
        "branch_assignments": state.branch_assignments,
    }
    contract = {
        "contract": LSSContractVersion,
        "task": "local_step_synthesis",
        "context": context,
        "constraints": {
            "max_candidates": min(2, max_candidates),
            "preferred_candidates": 1,
            "bounded_actions_only": True,
        },
        "output_schema": {
            "actions": [
                {
                    "action_type": "derive_constraint|rewrite|substitute|eliminate",
                    "title": "short title",
                    "added_facts": ["fact"],
                    "added_constraints": ["constraint"],
                }
            ]
        },
        "example_outputs": [
            {
                "actions": [
                    {
                        "action_type": "rewrite",
                        "title": "subtract_1",
                        "added_facts": ["x = 1"],
                        "added_constraints": [],
                    }
                ]
            },
            {
                "actions": [
                    {
                        "action_type": "derive_constraint",
                        "title": "record_domain",
                        "added_facts": [],
                        "added_constraints": ["x in R"],
                    },
                    {
                        "action_type": "substitute",
                        "title": "replace_y",
                        "added_facts": ["x + 2 = 10"],
                        "added_constraints": [],
                    }
                ]
            },
        ],
        "instructions": [
            "Return JSON only.",
            "Do not include prose.",
            "Do not include markdown fences.",
            "Emit at most 2 actions.",
            "Prefer 1 action when possible.",
            "Use only the fields shown in output_schema.",
        ],
    }
    return _json_block(contract)


def build_stage_system_prompt(stage: str) -> str:
    """Return stage-specific system instruction for structured generation."""

    base = (
        "You are a structured planning assistant. "
        "Respond with JSON only and no markdown fences."
    )
    if stage == STAGE_PT:
        return f"{base} Return only entities, target, and constraints as a tiny JSON object."
    if stage == STAGE_PCT:
        return f"{base} Return only a tiny JSON object with strategy tags, open goals, candidate equations, and optional integer answer candidate."
    if stage == STAGE_LSS:
        return f"{base} Return one or two tiny candidate actions only, using only action_type, title, added_facts, and added_constraints."
    return base
