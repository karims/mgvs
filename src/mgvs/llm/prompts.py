"""Centralized compact prompt contracts for state-first PT/PCT/LSS/endgame stages."""

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


def _lss_strategy_tags(state: ReasoningState) -> list[str]:
    """Return LSS strategy tags with conservative fallback sanitization."""

    tags = list(state.strategy_tags)
    # PHASE3_MOVE_EXTRACTION: when fallback tags are present, avoid forcing strong
    # domain priors into LSS context unless backed by concrete PT structure.
    if "llm_fallback" in tags and not state.current_equations and not state.domain_constraints:
        blocked = ("modular", "parity", "divisibility", "number_theory")
        tags = [tag for tag in tags if not any(token in tag.lower() for token in blocked)]
    return tags


def build_pt_prompt(raw_problem: str, target_type: str) -> str:
    """PT extracts a structured base state and does no solving."""

    contract = {
        "contract": PTContractVersion,
        "task": "structured_state_extraction_trace",
        "input": {
            "raw_problem": raw_problem,
            "target_type": target_type,
        },
        "output_format": {
            "required_sections": [
                "Restatement",
                "What is given",
                "What must be found or proved",
                "Key mathematical structure",
                "Suspicious quantities / invariants / substitutions",
                "Plausible first directions",
            ],
            "style": "concise mathematical notes",
        },
        "example_trace_output": [
            "Restatement: Restate the exact objective in one line.",
            "What is given:",
            "- List explicit relations and conditions from the problem text only.",
            "What must be found or proved:",
            "- State the exact unknown target.",
            "Key mathematical structure:",
            "- Identify core structure (algebraic system, counting with bounds, modular relation, etc.).",
            "Suspicious quantities / invariants / substitutions:",
            "- Name likely useful substitutions or invariant-like quantities.",
            "Plausible first directions:",
            "- Give 1-3 concrete first reduction directions.",
        ],
        "output_schema": {
            "objects": ["named object from problem"],
            "variables": ["explicit symbolic quantity"],
            "domains": ["domain or type restriction"],
            "relations": ["exact stated relation"],
            "constraints": ["constraint"],
            "goal": "short target",
            "unknowns_remaining": ["unresolved quantity"],
        },
        "example_output": {
            "objects": ["x"],
            "variables": ["x"],
            "domains": ["x is an integer"],
            "relations": ["x + 1 = 2"],
            "constraints": ["x + 1 = 2"],
            "goal": "determine x",
            "unknowns_remaining": ["x"],
        },
        "instructions": [
            "Do not solve the problem.",
            "Do not include explanations.",
            "Extract only machine-usable base state.",
            "Use exact relations from the problem statement when possible.",
            "Keep the object compact.",
            "Return sectioned free-text notes using exactly the required headings.",
            "Keep each section short and mathematically precise.",
            "Use numbered items when listing multiple points (1., 2., 3.).",
        ],
    }
    return _json_block(contract)


def build_pct_prompt(state: ReasoningState, *, max_tactics: int = DEFAULT_PCT_MAX_TACTICS) -> str:
    """PCT strengthens state with compact, machine-usable additions."""

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
        "task": "structured_state_strengthening_trace",
        "constraints": {"max_tactics": max_tactics},
        "limits": {
            "candidate_approaches_max": 2,
            "possible_intermediate_lemmas_max": 2,
            "useful_reformulations_max": 2,
            "risks_max": 2,
            "one_sentence_per_item": True,
        },
        "context": context,
        "output_format": {
            "required_sections": [
                "Candidate approaches",
                "Why each approach might help",
                "Possible intermediate lemmas",
                "Useful reformulations",
                "Risks / dead ends",
            ],
            "style": "serious exploratory planning notes",
        },
        "example_trace_output": [
            "Candidate approaches:",
            "- Approach A: direct elimination on primary equations.",
            "- Approach B: derive explicit bounds before elimination.",
            "Why each approach might help:",
            "- A may remove one unknown rapidly.",
            "- B may reduce to finite cases.",
            "Possible intermediate lemmas:",
            "- A lemma that converts target into bounded parameters.",
            "Useful reformulations:",
            "- Rewrite target in terms of lower-dimensional quantities.",
            "Risks / dead ends:",
            "- Symmetric manipulations that only restate given equations.",
        ],
        "output_schema": {
            "strategy_tags": ["concrete strategy tag"],
            "open_goals": ["specific strengthening target"],
            "candidate_equations": ["canonical definition | justified bound | invariant | exact case relation"],
            "answer_candidate": None,
        },
        "example_outputs": [
            {
                "strategy_tags": ["canonical_variables", "bound_search_space"],
                "open_goals": ["introduce explicit variables for side lengths", "derive a usable perimeter bound"],
                "candidate_equations": ["p = 2(a + b)", "4 <= p <= 2000"],
                "answer_candidate": None,
            },
        ],
        "instructions": [
            "Do not include explanations.",
            "Do not include a derivation.",
            "Do not solve the full problem in this stage.",
            "Strengthen the state, do not narrate strategy.",
            "Use short list items only.",
            "strategy_tags must be concrete and operational.",
            "open_goals must describe specific state improvements, not vague plans.",
            "candidate_equations may contain canonical variable definitions, justified bounds, invariants, or exact case relations supported by current state.",
            "Do not invent unsupported transformed equations or hidden assumptions.",
            "If a useful strengthening idea cannot yet be written as an exact relation, put it in open_goals instead.",
            "If no integer answer candidate is available, set answer_candidate to null.",
            "Return sectioned free-text notes using exactly the required headings.",
            "Keep each section concise and mathematically serious.",
            "Prefer numbered items for candidate approaches and lemmas.",
            "At most 2 candidate approaches.",
            "At most 2 possible intermediate lemmas.",
            "At most 2 useful reformulations.",
            "At most 2 risks/dead ends.",
            "Each listed item must be one short sentence.",
        ],
    }
    return _json_block(contract)


def build_lss_prompt(state: ReasoningState, max_candidates: int) -> str:
    """LSS proposes exactly one typed state transition for the validator/updater."""

    context = {
        "raw_problem": state.raw_problem,
        "status": state.status.value,
        "target_type": state.target_type,
        "pt_entities": _pt_entities(state),
        "pt_constraints": _pt_constraints(state),
        "pt_target": _pt_target(state),
        "current_equations": state.current_equations,
        "open_goals": state.open_goals,
        "strategy_tags": _lss_strategy_tags(state),
        "derived_facts": state.derived_facts,
        "branch_assignments": state.branch_assignments,
    }
    contract = {
        "contract": LSSContractVersion,
        "task": "local_step_synthesis_trace",
        "context": context,
        "constraints": {
            "max_candidates": 1 if max_candidates >= 1 else 1,
            "preferred_candidates": 1,
            "bounded_actions_only": True,
        },
        "output_format": {
            "required_sections": [
                "Candidate next steps",
                "Why each step helps",
                "What each step would establish",
                "Most promising immediate continuation",
            ],
            "style": "compact transition notes",
        },
        "example_trace_output": [
            "Candidate next steps:",
            "- Derive one explicit transformed relation from the currently active equations.",
            "Why each step helps:",
            "- It removes one free degree of freedom.",
            "What each step would establish:",
            "- A concrete relation that can be reused in the next reduction.",
            "Most promising immediate continuation:",
            "- Continue from the strongest newly derived relation.",
        ],
        "output_schema": {
            "actions": [
                {
                    "action_type": (
                        "derive_relation|tighten_bound|introduce_invariant|case_split|"
                        "normalize_representation|reduce_to_finite_search|convert_objective|"
                        "construct_candidate_formula"
                    ),
                    "title": "short title",
                    "added_facts": ["atomic new fact"],
                    "added_constraints": ["atomic new constraint"],
                }
            ]
        },
        "example_outputs": [
            {
                "actions": [
                    {
                        "action_type": "derive_relation",
                        "title": "derive_even_perimeter_form",
                        "added_facts": ["p = 2(a + b)"],
                        "added_constraints": [],
                    }
                ]
            },
            {
                "actions": [
                    {
                        "action_type": "tighten_bound",
                        "title": "derive_explicit_perimeter_count_bound",
                        "added_facts": ["Possible rectangle perimeters are even integers from 4 to 2000."],
                        "added_constraints": ["The number of distinct perimeters is at most 999."],
                    }
                ]
            },
            {
                "label": "bad_example",
                "actions": [
                    {
                        "action_type": "derive_relation",
                        "title": "repeat_goal",
                        "added_facts": ["Need the final answer modulo 10^5."],
                        "added_constraints": [],
                    }
                ],
                "reason": "restates context instead of adding a usable transition",
            },
        ],
        "instructions": [
            "The action must be atomic and machine-usable.",
            "added_facts and added_constraints must contain only new items, not explanations.",
            "Do not restate the current state, target, or existing equations.",
            "Do not add vague prose, unjustified assumptions, or weak observations.",
            "Prefer explicit new relations, explicit bounds, explicit invariants, or explicit finite case splits.",
            "Return sectioned free-text notes using exactly the required headings.",
            "List at most two candidate next steps.",
            "Keep each candidate concrete and directly actionable.",
            "Prefer numbered next steps (1., 2.) for stable parsing.",
            "Each candidate next step must be atomic.",
            "Each candidate must include an explicit equation, substitution, or exact derived target.",
            "Do not output generic steps such as: derive a relation, use substitution, solve the equations, check consistency.",
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
    """Endgame answers only from accumulated state and reports insufficiency explicitly."""

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
            "ready": False,
            "answer": None,
            "confidence": "high|medium|low",
            "justification": ["short bullet"],
            "missing_requirements": ["what is still missing"],
        },
        "example_outputs": [
            {
                "ready": True,
                "answer": 42,
                "confidence": "medium",
                "justification": [
                    "Use the reduced relations only.",
                ],
                "missing_requirements": [],
            },
            {
                "ready": False,
                "answer": None,
                "confidence": "low",
                "justification": ["The reduced state does not yet determine a unique integer."],
                "missing_requirements": ["Need one more explicit derived relation or bound."],
            },
        ],
        "instructions": [
            "Return JSON only.",
            "Do not include markdown fences.",
            "Do not include prose before or after the JSON object.",
            "Use the reduced state as the primary input.",
            "Answer only from accumulated state.",
            'If the state is insufficient, return {"ready": false, "answer": null, ...}.',
            "Do not guess.",
            "Do not output a confident integer answer from vague qualitative bounds alone.",
            "Only return a numeric answer when the equations, facts, and constraints are sufficient to justify a unique integer.",
            "Do not restart the whole problem from scratch unless necessary.",
            "Set answer to an integer or null.",
            "Set ready to true only when the current state is sufficient.",
            "Set confidence to exactly one of: high, medium, low.",
            "Keep justification extremely short.",
            "Return 1 to 3 short justification strings at most.",
            "If not ready, list the missing requirements compactly.",
        ],
    }
    return _json_block(contract)


def build_stage_system_prompt(stage: str) -> str:
    """Return stage-specific system instruction for structured generation."""

    base = "You are a state-first reasoning assistant."
    if stage == STAGE_PT:
        return (
            f"{base} Produce serious sectioned trace notes with the PT headings exactly as requested. "
            "Do not solve; extract and organize the mathematical state."
        )
    if stage == STAGE_PCT:
        return (
            f"{base} Produce sectioned PCT planning notes only. "
            "Keep content compact, tactical, and non-derivational."
        )
    if stage == STAGE_LSS:
        return (
            f"{base} Produce sectioned LSS local-step notes with concrete next steps. "
            "Avoid vague advice; focus on immediate mathematical reductions."
        )
    if stage == STAGE_ENDGAME:
        return (
            f"{base} Answer only from accumulated state. If the state is insufficient, return ready=false "
            "with missing_requirements instead of guessing."
        )
    return base
