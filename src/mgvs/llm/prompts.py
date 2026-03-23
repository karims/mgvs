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
            "unknowns": ["symbol"],
            "targets": ["target"],
            "facts": ["fact"],
            "constraints": {
                "domain": ["constraint"],
                "global": ["constraint"],
            },
            "symbolic_objects": {"name": {"kind": "type", "attrs": {"key": "value"}}},
            "equations": ["equation"],
        },
        "instructions": [
            "Return JSON only.",
            "No free-form reasoning paragraphs.",
            "Use short machine-friendly tokens.",
            "Do not include markdown fences.",
        ],
    }
    return _json_block(contract)


def build_pct_prompt(state: ReasoningState, *, max_tactics: int = DEFAULT_PCT_MAX_TACTICS) -> str:
    """Build PCT prompt contract for strategy and goal proposal."""

    context = {
        "raw_problem": state.raw_problem,
        "target_type": state.target_type,
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
            "tactic_candidates": [
                {
                    "tag": "actionable_tag",
                    "goal": "short_goal",
                    "priority": 1,
                }
            ],
            "focus_goals": ["goal"],
        },
        "instructions": [
            "Return JSON only.",
            "At most max_tactics tactic candidates.",
            "Prefer tags/goals, avoid essays.",
        ],
    }
    return _json_block(contract)


def build_lss_prompt(state: ReasoningState, max_candidates: int) -> str:
    """Build LSS prompt contract for bounded candidate action synthesis."""

    context = {
        "status": state.status.value,
        "target_type": state.target_type,
        "open_goals": state.open_goals,
        "derived_facts": state.derived_facts,
        "branch_assignments": state.branch_assignments,
    }
    contract = {
        "contract": LSSContractVersion,
        "task": "local_step_synthesis",
        "context": context,
        "constraints": {
            "max_candidates": max_candidates,
            "bounded_actions_only": True,
        },
        "output_schema": {
            "actions": [
                {
                    "action_type": "rewrite|substitute|eliminate|factor|expand|introduce_representation|hypothesize_witness|bind_witness|derive_constraint|detect_symmetry|branch|prune",
                    "title": "short title",
                    "rationale": "short why <= 20 words",
                    "inputs": ["item"],
                    "outputs": ["item"],
                    "added_facts": ["fact"],
                    "added_constraints": ["constraint"],
                    "branch_labels": ["label"],
                    "metadata": {
                        "mark_solved": False,
                        "mark_parametric": False,
                        "prune_status": "dead_end|contradiction",
                    },
                }
            ]
        },
        "instructions": [
            "Return JSON only.",
            "Bounded candidate actions only; no prose outside action fields.",
            "Only include fields from output_schema.",
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
        return f"{base} Extract objects, equations, constraints, witnesses, and goals."
    if stage == STAGE_PCT:
        return f"{base} Propose concept/tactic tags and incremental goals."
    if stage == STAGE_LSS:
        return f"{base} Propose bounded candidate actions only."
    return base
