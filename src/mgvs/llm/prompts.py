"""Prompt builders for PT, PCT, LSS, and endgame structured-output stages."""

from __future__ import annotations

import json

from mgvs.llm.base import (
    DEFAULT_PCT_MAX_TACTICS,
    ENDGAMEContractVersion,
    LSSContractVersion,
    PCTContractVersion,
    PTContractVersion,
)
from mgvs.state.models import ReasoningState

STAGE_PT = "pt"
STAGE_PCT = "pct"
STAGE_LSS = "lss"
STAGE_ENDGAME = "endgame"


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
        "example_outputs": [
            {
                "strategy_tags": ["system_of_equations", "direct_translation"],
                "open_goals": ["introduce variable definitions for unknown quantities", "decide which equation to use first"],
                "candidate_equations": ["A + a = 2(B + b)", "A*a = 4(B*b)"],
                "answer_candidate": None,
            },
            {
                "strategy_tags": ["variable_definition", "translation"],
                "open_goals": ["if transfer variables are needed, define them explicitly before rewriting", "defer algebraic manipulation to later steps"],
                "candidate_equations": ["total_red + moved_red = total_blue + moved_blue + 10"],
                "answer_candidate": None,
            },
        ],
        "instructions": [
            "Return JSON only.",
            "Do not include markdown fences.",
            "Do not include explanations.",
            "Do not include a full derivation.",
            "Do not solve the full problem in this stage.",
            "Do not provide bullet lists.",
            "Do not include prose before or after the JSON object.",
            "If tempted to explain, instead return shorter lists.",
            "Keep the object small and concise.",
            "Use only these top-level keys: strategy_tags, open_goals, candidate_equations, answer_candidate.",
            "candidate_equations must be directly grounded in the problem statement or be simple variable-definition equations introduced explicitly.",
            "Do not invent transformed equations such as shifted sums or shifted products unless they are explicitly stated in the problem.",
            "Do not rewrite equations after hypothetical transfers unless the transfer variables are introduced explicitly and the transformed relation is exact.",
            "Prefer direct statement equations first.",
            "If a relation requires later manipulation, include it in open_goals, not candidate_equations.",
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
            "max_candidates": 1 if max_candidates >= 1 else 1,
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
                        "action_type": "substitute",
                        "title": "derive_shifted_sum_relation",
                        "added_facts": ["a + s_a = b + s_b + 10"],
                        "added_constraints": [],
                    }
                ]
            },
            {
                "label": "bad_example_do_not_copy",
                "actions": [
                    {
                        "action_type": "eliminate",
                        "title": "eliminate variable",
                        "added_facts": ["a + b = c"],
                        "added_constraints": [],
                    }
                ],
                "reason": "does not introduce new information",
            }
        ],
        "instructions": [
            "Return JSON only.",
            "Do not include prose.",
            "Do not include markdown fences.",
            "Emit at most 1 action.",
            "Prefer 1 action when possible.",
            "Use only the fields shown in output_schema.",
            "The action must be grounded in the current problem context, PT constraints, PT target, current equations, or known facts.",
            "Do not invent generic placeholder variables unless they already appear in the context.",
            "Do not copy toy algebra patterns that are unrelated to the current problem.",
            "Do not output high-level tactics like eliminate variable or simplify without showing the result.",
            "Every action must introduce NEW information not already present in current_equations.",
            "If action_type is substitute or eliminate, the resulting simplified or derived relation MUST appear in added_facts.",
            "Do not copy or restate existing equations into added_facts or added_constraints.",
            "Prefer explicit derived equations such as simplified equations, substituted expressions, reduced forms, or new equalities between variables.",
            "Do not repeat already-known equations or constraints.",
            "Do not restate target-only facts such as the final modulus target without adding new information.",
            "Do not propose actions that merely restate current equations.",
            "Do not propose empty eliminations with no downstream consequence.",
            "Propose one action that introduces a genuinely new constraint, bound, counting relation, or case distinction tied to the current problem.",
            "Bad example reason: an eliminate action that only repeats an existing relation does not count as progress.",
            "Good example pattern: substitute or eliminate only when the transformed result is written explicitly in added_facts.",
            'If no materially advancing action is available, return {"actions": []}.',
        ],
    }
    return _json_block(contract)


def build_endgame_solve_prompt(
    *,
    raw_problem: str,
    pt_target: str,
    pt_constraints: list[str],
    current_equations: list[str],
    derived_facts: list[str],
    open_goals: list[str],
    strategy_tags: list[str],
    trace_summary: list[str] | None = None,
) -> str:
    """Build endgame prompt contract for solving from a reduced structured state."""

    contract = {
        "contract": ENDGAMEContractVersion,
        "task": "endgame_solve_from_reduced_state",
        "context": {
            "raw_problem": raw_problem,
            "pt_target": pt_target,
            "pt_constraints": list(pt_constraints),
            "current_equations": list(current_equations),
            "derived_facts": list(derived_facts),
            "open_goals": list(open_goals),
            "strategy_tags": list(strategy_tags),
            "trace_summary": list(trace_summary or []),
        },
        "output_schema": {
            "answer": None,
            "confidence": "high|medium|low",
            "justification": ["short bullet"],
        },
        "example_output": {
            "answer": 42,
            "confidence": "medium",
            "justification": [
                "Use the reduced constraints and current equations.",
                "Do not restart from scratch if the state already narrows the answer.",
            ],
        },
        "instructions": [
            "Return JSON only.",
            "Do not include markdown fences.",
            "Do not include prose before or after the JSON object.",
            "Use the reduced state as the primary input.",
            "Do not restart the whole problem from scratch unless necessary.",
            "Set answer to an integer or null.",
            "Set confidence to exactly one of: high, medium, low.",
            "Keep justification extremely short.",
            "Return 1 to 3 short justification strings at most.",
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
        return (
            f"{base} Return at most one tiny grounded candidate action, using only "
            "action_type, title, added_facts, and added_constraints. "
            "The action must reference the actual problem context rather than generic placeholder algebra."
        )
    if stage == STAGE_ENDGAME:
        return (
            f"{base} Return only a tiny endgame JSON object with answer, confidence, "
            "and very short justification strings based on the reduced state."
        )
    return base
