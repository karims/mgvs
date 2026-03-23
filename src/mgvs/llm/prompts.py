"""Prompt builders for PT, PCT, and LSS structured-output stages."""

from __future__ import annotations

import json

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
        "task": "problem_translation",
        "input": {
            "raw_problem": raw_problem,
            "target_type": target_type,
        },
        "output_schema": {
            "symbolic_objects": {"name": "descriptor"},
            "current_equations": ["equation"],
            "domain_constraints": ["constraint"],
            "global_constraints": ["constraint"],
            "witness_parameters": {"key": "value"},
            "open_goals": ["goal"],
        },
        "instructions": [
            "Return JSON only.",
            "Use concise machine-friendly strings.",
            "Do not include markdown fences.",
        ],
    }
    return _json_block(contract)


def build_pct_prompt(state: ReasoningState) -> str:
    """Build PCT prompt contract for strategy and goal proposal."""

    context = {
        "raw_problem": state.raw_problem,
        "target_type": state.target_type,
        "derived_facts": state.derived_facts,
        "open_goals": state.open_goals,
        "strategy_tags": state.strategy_tags,
    }
    contract = {
        "task": "concept_tactic_proposal",
        "context": context,
        "output_schema": {
            "strategy_tags": ["tag"],
            "open_goals": ["goal"],
            "added_facts": ["fact"],
            "added_constraints": ["constraint"],
        },
        "instructions": [
            "Return JSON only.",
            "Prefer generic tactics, not full proofs.",
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
        "task": "local_step_synthesis",
        "context": context,
        "constraints": {
            "max_candidates": max_candidates,
            "bounded_actions_only": True,
        },
        "output_schema": {
            "actions": [
                {
                    "action_type": "rewrite|substitute|...|prune",
                    "title": "short title",
                    "rationale": "short why",
                    "inputs": ["item"],
                    "outputs": ["item"],
                    "added_facts": ["fact"],
                    "added_constraints": ["constraint"],
                    "branch_labels": ["label"],
                    "metadata": {"key": "value"},
                }
            ]
        },
        "instructions": [
            "Return JSON only.",
            "Propose partial next actions, not full derivations.",
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
