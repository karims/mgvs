"""Local end-to-end runner wiring PT/PCT/LSS, verification, and search."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, replace

from mgvs.actions.apply import apply_action
from mgvs.actions.models import ActionType, CandidateAction
from mgvs.config import RuntimeBudgetConfig, SolveModeSettings, SolvePolicyConfig
from mgvs.domains import active_domain_plugins
from mgvs.domains.base import DomainPlugin
from mgvs.llm.base import UnifiedLLMClient
from mgvs.llm.parser import (
    EndgameSolveOutput,
    PCTUpdate,
    apply_pct_update,
    parse_endgame_solve_output,
    apply_pt_update,
    parse_lss_output,
    parse_pct_output,
    parse_pt_output,
    parse_structured_json_object,
)
from mgvs.llm.prompts import (
    build_endgame_solve_prompt,
    build_lss_prompt,
    build_pct_prompt,
    build_pt_prompt,
)
from mgvs.llm.stub import StubLLMClient
from mgvs.llm.vllm_client import VLLMClient
from mgvs.search.controller import ControllerResult, IterationSummary, StateCanonicalizer
from mgvs.search.termination import is_terminal_state, terminal_states
from mgvs.solve.answering import BeamAnswerDecision, select_answer_across_states
from mgvs.solve.cache import StageCache
from mgvs.solve.policy import (
    FallbackDecision,
    ModeSelection,
    SolveMode,
    mode_settings_for,
    select_fallback,
    select_solve_mode,
)
from mgvs.state.models import ReasoningState, create_initial_state
from mgvs.types import StateStatus
from mgvs.verify.base import CombinedVerificationResult, CompositeVerifier
from mgvs.verify.consistency import V0StateConsistencyVerifier
from mgvs.verify.global_ import V0GlobalCompatibilityVerifier
from mgvs.verify.local import V0LocalValidityVerifier

@dataclass(frozen=True)
class SolveConfig:
    """Configuration for local solve orchestration."""

    target_type: str = "unspecified"
    max_depth: int = 4
    lss_transition_budget: int = 3
    beam_width: int = 3
    max_candidates: int = 3
    max_wall_time_s: float = 20.0
    session_max_wall_time_s: float = 0.0
    requested_mode: str = "auto"
    policy_config: SolvePolicyConfig = field(default_factory=SolvePolicyConfig.default)
    debug_single_path: bool = False
    experiment_state_first: bool = False
    exploratory_search: bool = False
    max_initial_candidate_moves: int = 3
    max_branches_to_expand: int = 2
    max_search_depth: int = 2
    enable_verbose_trace: bool = True
    enable_phase6_synthesis: bool = True
    # Backward-compatible aliases for exploratory knobs.
    exploratory_max_initial_moves: int = 3
    exploratory_expand_top_branches: int = 2
    exploratory_max_depth: int = 2

    @classmethod
    def from_env(cls, *, target_type: str = "unspecified") -> "SolveConfig":
        """Build solve config from runtime budget environment variables."""

        budget = RuntimeBudgetConfig.from_env()
        return cls(
            target_type=target_type,
            max_depth=budget.max_depth,
            lss_transition_budget=3,
            beam_width=budget.beam_width,
            max_candidates=budget.candidate_action_cap_per_state,
            max_wall_time_s=budget.per_problem_max_wall_time_s,
            session_max_wall_time_s=budget.session_max_wall_time_s,
            requested_mode="auto",
            policy_config=SolvePolicyConfig.from_env(),
            debug_single_path=os.getenv("MGVS_DEBUG_SINGLE_PATH", "0").strip().lower() in {"1", "true", "yes", "on"},
            experiment_state_first=os.getenv("MGVS_EXPERIMENT_STATE_FIRST", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            exploratory_search=os.getenv("MGVS_EXPLORATORY_SEARCH", "0").strip().lower() in {"1", "true", "yes", "on"},
            max_initial_candidate_moves=max(
                1,
                int(
                    os.getenv(
                        "MGVS_MAX_INITIAL_CANDIDATE_MOVES",
                        os.getenv("MGVS_EXPLORATORY_MAX_INITIAL_MOVES", "3"),
                    )
                ),
            ),
            max_branches_to_expand=max(
                1,
                int(
                    os.getenv(
                        "MGVS_MAX_BRANCHES_TO_EXPAND",
                        os.getenv("MGVS_EXPLORATORY_EXPAND_TOP_BRANCHES", "2"),
                    )
                ),
            ),
            max_search_depth=max(
                1,
                int(
                    os.getenv(
                        "MGVS_MAX_SEARCH_DEPTH",
                        os.getenv("MGVS_EXPLORATORY_MAX_DEPTH", "2"),
                    )
                ),
            ),
            enable_verbose_trace=os.getenv("MGVS_ENABLE_VERBOSE_TRACE", "1").strip().lower()
            in {"1", "true", "yes", "on"},
            enable_phase6_synthesis=os.getenv("MGVS_ENABLE_PHASE6_SYNTHESIS", "1").strip().lower()
            in {"1", "true", "yes", "on"},
            exploratory_max_initial_moves=max(
                1,
                int(
                    os.getenv(
                        "MGVS_EXPLORATORY_MAX_INITIAL_MOVES",
                        os.getenv("MGVS_MAX_INITIAL_CANDIDATE_MOVES", "3"),
                    )
                ),
            ),
            exploratory_expand_top_branches=max(
                1,
                int(
                    os.getenv(
                        "MGVS_EXPLORATORY_EXPAND_TOP_BRANCHES",
                        os.getenv("MGVS_MAX_BRANCHES_TO_EXPAND", "2"),
                    )
                ),
            ),
            exploratory_max_depth=max(
                1,
                int(
                    os.getenv(
                        "MGVS_EXPLORATORY_MAX_DEPTH",
                        os.getenv("MGVS_MAX_SEARCH_DEPTH", "2"),
                    )
                ),
            ),
        )


@dataclass(frozen=True)
class SolveResult:
    """Result payload from end-to-end local solve run."""

    best_state: ReasoningState
    trace_summary: list[str]
    termination_reason: str
    depth_reached: int
    predicted_answer: int | None = None
    answer_status: str = "missing_answer"
    answer_source: str = ""
    supporting_state_ids: list[str] = field(default_factory=list)
    supporting_trace_count: int = 0
    solve_mode: str = "balanced"
    fallback_used: bool = False
    fallback_reason: str = ""
    policy_trace: list[str] = field(default_factory=list)
    iteration_summaries: list[IterationSummary] = field(default_factory=list)
    verifier_rejections_by_level: dict[str, int] = field(default_factory=dict)


class DefaultStateCanonicalizer(StateCanonicalizer):
    """Deterministic canonicalizer for search dedup in local runs."""

    def canonicalize(self, state: ReasoningState) -> ReasoningState:
        if state.normalized_form:
            if state.branch_assignments:
                branch_suffix = f"::{state.branch_assignments[-1]}"
                if not state.normalized_form.endswith(branch_suffix):
                    state.normalized_form = f"{state.normalized_form}{branch_suffix}"
            return state

        facts = sorted(state.derived_facts)
        branches = list(state.branch_assignments)
        goals = sorted(state.open_goals)
        state.normalized_form = (
            f"status={state.status.value}|facts={','.join(facts)}|"
            f"branches={','.join(branches)}|goals={','.join(goals)}"
        )
        return state


class TrackingCompositeVerifier:
    """Controller-compatible verifier wrapper that tracks rejection counts."""

    def __init__(self, verifier: CompositeVerifier, domain_plugins: list[DomainPlugin] | None = None) -> None:
        self._verifier = verifier
        self._domain_plugins = domain_plugins or []
        self.rejections_by_level: dict[str, int] = {
            "local": 0,
            "consistency": 0,
            "global": 0,
            "domain": 0,
        }
        self._last_action_rejection: dict[str, object] | None = None
        self._last_state_rejection: dict[str, object] | None = None

    def is_action_valid(self, state: ReasoningState, action: CandidateAction) -> bool:
        self._last_action_rejection = None
        result = self._verifier.verify_action(state, action)
        self._record_rejections(result)
        if not result.passed:
            failure = result.first_failure
            if failure is not None:
                self._last_action_rejection = {
                    "layer": f"{failure.level}_verifier_reject",
                    "reason": failure.reason,
                    "details": dict(failure.details),
                    "title": action.title,
                    "action_type": action.action_type.value,
                }
            if failure is not None and failure.level == "local":
                detail_hint = _format_rejection_details(failure.details)
                _debug_runtime_print(
                    "[runner][verify] local_reject "
                    f"action_title={action.title!r} action_type={action.action_type.value} "
                    f"reason={failure.reason} details={detail_hint}"
                )
            return False

        for plugin in self._domain_plugins:
            domain_result = plugin.validate_action(state, action)
            if not domain_result.passed:
                self.rejections_by_level["domain"] = self.rejections_by_level.get("domain", 0) + 1
                self._last_action_rejection = {
                    "layer": "domain_verifier_reject",
                    "reason": domain_result.reason,
                    "details": dict(domain_result.details),
                    "title": action.title,
                    "action_type": action.action_type.value,
                }
                return False
        return True

    def is_state_valid(self, state: ReasoningState) -> bool:
        self._last_state_rejection = None
        result = self._verifier.verify_state(state)
        self._record_rejections(result)
        if not result.passed:
            failure = result.first_failure
            if failure is not None:
                self._last_state_rejection = {
                    "layer": "state_transition_reject",
                    "reason": failure.reason,
                    "details": dict(failure.details),
                }
            return False

        for plugin in self._domain_plugins:
            domain_result = plugin.validate_state(state)
            if not domain_result.passed:
                self.rejections_by_level["domain"] = self.rejections_by_level.get("domain", 0) + 1
                self._last_state_rejection = {
                    "layer": "state_transition_reject",
                    "reason": domain_result.reason,
                    "details": dict(domain_result.details),
                }
                return False
        return True

    def consume_last_action_rejection(self) -> dict[str, object] | None:
        """Return and clear last action-level rejection details."""

        rejection = self._last_action_rejection
        self._last_action_rejection = None
        return rejection

    def consume_last_state_rejection(self) -> dict[str, object] | None:
        """Return and clear last state-level rejection details."""

        rejection = self._last_state_rejection
        self._last_state_rejection = None
        return rejection

    def _record_rejections(self, result: CombinedVerificationResult) -> None:
        if result.passed:
            return
        for level_result in result.results:
            if level_result.passed:
                continue
            self.rejections_by_level[level_result.level] = self.rejections_by_level.get(level_result.level, 0) + 1


def build_default_verifier() -> CompositeVerifier:
    """Construct the default three-level composite verifier."""

    return CompositeVerifier(
        local_verifier=V0LocalValidityVerifier(),
        consistency_verifier=V0StateConsistencyVerifier(),
        global_verifier=V0GlobalCompatibilityVerifier(),
    )


def select_llm_client(backend: str) -> UnifiedLLMClient:
    """Select LLM backend by name."""

    normalized = backend.strip().lower()
    _debug_runtime_print(f"[runner] select_llm_client backend={normalized}")
    if normalized == "vllm":
        return VLLMClient.from_env()
    return StubLLMClient()


_STAGE_CACHE = StageCache()
_SESSION_STARTED_AT: float | None = None


def _debug_runtime_enabled() -> bool:
    """Return whether runner-level debug tracing is enabled."""

    return os.environ.get("MGVS_DEBUG_RUNTIME") == "1" or os.environ.get("MGVS_DEBUG_LLM") == "1"


def _debug_runtime_print(message: str) -> None:
    """Print runner-level debug trace when enabled."""

    if _debug_runtime_enabled():
        print(message)


def _phase1_trace_persist_enabled() -> bool:
    """Return whether Phase 1 trace artifacts should be persisted."""

    return os.environ.get("MGVS_PHASE1_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _phase1_trace_dir() -> str:
    """Return output directory for Phase 1 trace artifacts."""

    return os.environ.get("MGVS_PHASE1_TRACE_DIR", "out/phase1_traces")


def _persist_phase1_trace_artifact(
    *,
    problem_text: str,
    mode: SolveMode,
    fallback_used: bool,
    enable_verbose_trace: bool,
    attempt_context: RunAttemptContext,
    result: SolveResult,
) -> None:
    """Write one readable PT/PCT/LSS trace artifact for this run.

    PHASE1_TRACE: temporary debugging persistence for offline inspection.
    """

    if not (enable_verbose_trace or _phase1_trace_persist_enabled()):
        return
    try:
        output_dir = _phase1_trace_dir()
        os.makedirs(output_dir, exist_ok=True)
        problem_key = _problem_hash(problem_text)[:12]
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        fallback_suffix = "fallback" if fallback_used else "primary"
        filename = f"{stamp}_{problem_key}_{mode.value}_{fallback_suffix}.md"
        path = os.path.join(output_dir, filename)

        lss_blocks = attempt_context.lss_raw_outputs or []
        if lss_blocks:
            lss_rendered = []
            for index, block in enumerate(lss_blocks, start=1):
                lss_rendered.append(f"### LSS CALL {index}\n{block.strip() or '(empty)'}")
            lss_text = "\n\n".join(lss_rendered)
        else:
            lss_text = "(none)"

        final_answer = "NA" if result.predicted_answer is None else str(result.predicted_answer)
        moves_text = "(none)"
        if attempt_context.extracted_candidate_moves:
            rendered = []
            for index, move in enumerate(attempt_context.extracted_candidate_moves, start=1):
                rendered.append(
                    f"{index}. move_text={move.get('move_text', '')}\n"
                    f"   why_it_helps={move.get('why_it_helps', '')}\n"
                    f"   what_it_establishes={move.get('what_it_establishes', '')}\n"
                    f"   source_stage={move.get('source_stage', '')} score={move.get('score', '')}"
                )
            moves_text = "\n".join(rendered)
        branch_expansions_text = (
            "\n".join(attempt_context.branch_decisions) if attempt_context.branch_decisions else "(none)"
        )
        branch_scores_text = "\n".join(attempt_context.branch_scores) if attempt_context.branch_scores else "(none)"
        pruning_text = (
            "\n".join(attempt_context.pruning_decisions) if attempt_context.pruning_decisions else "(none)"
        )
        best_branch_text = attempt_context.best_branch_summary or "(none)"
        synthesis_text = attempt_context.final_synthesis_text.strip() or "(none)"
        body = "\n".join(
            [
                "# PHASE1_TRACE RUN",
                "",
                "## PROBLEM",
                problem_text.strip() or "(empty)",
                "",
                "## PT RAW OUTPUT",
                attempt_context.pt_raw_output.strip() or "(none)",
                "",
                "## PCT RAW OUTPUT",
                attempt_context.pct_raw_output.strip() or "(none)",
                "",
                "## LSS RAW OUTPUT",
                lss_text,
                "",
                "## EXTRACTED CANDIDATE MOVES",
                moves_text,
                "",
                "## BRANCH EXPANSIONS",
                branch_expansions_text,
                "",
                "## BRANCH SCORES",
                branch_scores_text,
                "",
                "## PRUNING DECISIONS",
                pruning_text,
                "",
                "## BEST BRANCH",
                best_branch_text,
                "",
                "## FINAL SYNTHESIS",
                synthesis_text,
                "",
                "## FINAL OUTPUT / FINAL ANSWER",
                f"- termination_reason: {result.termination_reason}",
                f"- answer_status: {result.answer_status}",
                f"- answer_source: {result.answer_source or 'NA'}",
                f"- predicted_answer: {final_answer}",
            ]
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        _debug_runtime_print(f"[runner][phase1_trace] wrote {path}")
    except OSError as exc:
        _debug_runtime_print(f"[runner][phase1_trace] write_failed error={exc}")


def _format_rejection_details(details: dict[str, object]) -> str:
    """Render compact rejection details for debug logs."""

    if not details:
        return "none"
    preferred_keys = ("field", "error", "examples", "action_type")
    parts: list[str] = []
    for key in preferred_keys:
        if key not in details:
            continue
        parts.append(f"{key}={details[key]!r}")
    if not parts:
        parts = [f"{key}={value!r}" for key, value in details.items()]
    return ", ".join(parts)


def _stage_cache_disabled() -> bool:
    """Return whether stage cache should be bypassed for debugging."""

    return os.environ.get("MGVS_DISABLE_STAGE_CACHE") == "1"


def _normalize_items(items: list[str]) -> tuple[str, ...]:
    """Normalize string items for stable action-signature comparison."""

    return tuple(sorted(str(item).strip() for item in items if str(item).strip()))


def _action_signature(action: CandidateAction) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Build a canonical action signature for duplicate detection."""

    return (
        action.action_type.value,
        action.title.strip(),
        _normalize_items(action.added_facts),
        _normalize_items(action.added_constraints),
    )


def _semantic_action_signature(action: CandidateAction) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Build a title-independent action signature for semantic duplicate detection."""

    return (
        action.action_type.value,
        _normalize_items(action.added_facts),
        _normalize_items(action.added_constraints),
    )


def _accepted_action_signatures(state: ReasoningState) -> set[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    """Build canonical signatures for accepted actions already on this path."""

    signatures: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for step in state.accepted_steps:
        title = str(step.updates.get("title", "")).strip()
        if not title:
            continue
        raw_facts = step.updates.get("added_facts", [])
        facts = [str(item) for item in raw_facts] if isinstance(raw_facts, list) else []
        raw_constraints = step.updates.get("added_constraints", [])
        constraints = [str(item) for item in raw_constraints] if isinstance(raw_constraints, list) else []
        signatures.add((step.action, title, _normalize_items(facts), _normalize_items(constraints)))
    return signatures


def _accepted_semantic_action_signatures(state: ReasoningState) -> set[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Build title-independent signatures for prior accepted actions on this path."""

    signatures: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for step in state.accepted_steps:
        raw_facts = step.updates.get("added_facts", [])
        facts = [str(item) for item in raw_facts] if isinstance(raw_facts, list) else []
        raw_constraints = step.updates.get("added_constraints", [])
        constraints = [str(item) for item in raw_constraints] if isinstance(raw_constraints, list) else []
        signatures.add((step.action, _normalize_items(facts), _normalize_items(constraints)))
    return signatures


def _is_equation_restatement(state: ReasoningState, action: CandidateAction) -> bool:
    """Return True when an action just copies current equations back into state fields."""

    known_equations = set(str(item).strip() for item in state.current_equations if str(item).strip())
    if not known_equations:
        return False

    normalized_facts = _normalize_items(action.added_facts)
    normalized_constraints = _normalize_items(action.added_constraints)
    if not normalized_facts and not normalized_constraints:
        return False

    facts_are_equations = bool(normalized_facts) and all(item in known_equations for item in normalized_facts)
    constraints_are_equations = bool(normalized_constraints) and all(
        item in known_equations for item in normalized_constraints
    )
    if facts_are_equations and constraints_are_equations:
        return True
    return False


def _action_adds_no_new_information(state: ReasoningState, action: CandidateAction) -> bool:
    """Return True when an action adds no facts or constraints beyond current state."""

    if action.action_type.value in {"branch", "prune"}:
        return False
    if action.outputs or action.branch_labels or action.metadata:
        return False

    known_facts = set(str(item).strip() for item in state.derived_facts + state.current_equations if str(item).strip())
    known_constraints = set(
        str(item).strip()
        for item in state.domain_constraints + state.global_constraints + state.current_equations
        if str(item).strip()
    )
    normalized_facts = _normalize_items(action.added_facts)
    normalized_constraints = _normalize_items(action.added_constraints)
    if _is_equation_restatement(state, action):
        return True
    if not normalized_facts and not normalized_constraints:
        return True
    facts_known = all(item in known_facts for item in normalized_facts)
    constraints_known = all(item in known_constraints for item in normalized_constraints)
    return facts_known and constraints_known


def _score_lss_action(state: ReasoningState, action: CandidateAction) -> tuple[float, dict[str, float]]:
    """Compute a lightweight local relevance/novelty score for LSS actions."""

    components: dict[str, float] = {}
    score = 0.0

    known_entities = set(str(name).strip().lower() for name in state.symbolic_objects.keys() if str(name).strip())
    known_constraints = [
        str(item).strip().lower()
        for item in state.domain_constraints + state.global_constraints
        if str(item).strip()
    ]
    known_equations = [str(item).strip().lower() for item in state.current_equations if str(item).strip()]
    known_goals = [str(item).strip().lower() for item in state.open_goals if str(item).strip()]
    prior_content = set(
        str(item).strip().lower()
        for step in state.accepted_steps
        for key in ("added_facts", "added_constraints")
        for item in (step.updates.get(key, []) if isinstance(step.updates.get(key, []), list) else [])
        if str(item).strip()
    )

    added_facts = [str(item).strip() for item in action.added_facts if str(item).strip()]
    added_constraints = [str(item).strip() for item in action.added_constraints if str(item).strip()]
    added_text = " ".join([action.title] + added_facts + added_constraints).lower()

    new_items = 0
    for item in added_facts:
        if item not in state.derived_facts and item not in state.current_equations:
            new_items += 1
    for item in added_constraints:
        if item not in state.domain_constraints and item not in state.global_constraints:
            new_items += 1
    if new_items:
        components["new_information"] = 1.5 * new_items
        score += components["new_information"]

    if known_entities and any(entity in added_text for entity in known_entities):
        components["entity_grounding"] = 0.75
        score += components["entity_grounding"]

    if any(constraint and constraint in added_text for constraint in known_constraints):
        components["constraint_grounding"] = 0.75
        score += components["constraint_grounding"]

    if any(equation and equation in added_text for equation in known_equations):
        components["equation_relevance"] = 0.5
        score += components["equation_relevance"]

    if any(goal and goal in added_text for goal in known_goals):
        components["goal_relevance"] = 0.5
        score += components["goal_relevance"]

    if action.action_type.value in {"derive_constraint", "rewrite", "substitute", "eliminate"}:
        components["preferred_action_type"] = 0.25
        score += components["preferred_action_type"]

    target_text = known_goals[0] if known_goals else ""
    target_only = False
    if target_text:
        target_tokens = [token for token in target_text.split() if token]
        if target_tokens and any(token in added_text for token in target_tokens):
            grounded_elsewhere = any(constraint and constraint in added_text for constraint in known_constraints) or any(
                equation and equation in added_text for equation in known_equations
            )
            if not grounded_elsewhere:
                target_only = True
    if target_only:
        components["target_restatement_penalty"] = -1.5
        score += components["target_restatement_penalty"]

    boilerplate_tokens = ("mod", "modular", "parity", "even", "odd")
    if any(token in added_text for token in boilerplate_tokens):
        grounded_to_context = any(token in item for token in known_constraints + known_equations + known_goals for token in boilerplate_tokens)
        if not grounded_to_context:
            components["generic_boilerplate_penalty"] = -0.75
            score += components["generic_boilerplate_penalty"]

    if any(item.lower() in prior_content for item in added_facts + added_constraints):
        components["repeated_content_penalty"] = -1.0
        score += components["repeated_content_penalty"]

    return score, components


@dataclass
class RunAttemptContext:
    """Mutable execution context for one solve attempt."""

    malformed_output_count: int = 0
    llm_fallback_reasons: list[str] = field(default_factory=list)
    # PHASE1_TRACE: Keep raw stage outputs for per-run readable trace persistence.
    pt_raw_output: str = ""
    pct_raw_output: str = ""
    lss_raw_outputs: list[str] = field(default_factory=list)
    # PHASE3_MOVE_EXTRACTION / PHASE5_BRANCH_SEARCH / PHASE6_SYNTHESIS trace state.
    extracted_candidate_moves: list[dict[str, object]] = field(default_factory=list)
    branch_decisions: list[str] = field(default_factory=list)
    branch_scores: list[str] = field(default_factory=list)
    pruning_decisions: list[str] = field(default_factory=list)
    best_branch_summary: str = ""
    final_synthesis_text: str = ""


@dataclass(frozen=True)
class ExtractedCandidateMove:
    """PHASE3_MOVE_EXTRACTION: lightweight move record from free-text traces."""

    move_text: str
    why_it_helps: str
    what_it_establishes: str
    source_stage: str
    score: float = 0.0


@dataclass
class ExploratoryReasoningState:
    """PHASE4_REASONING_STATE: local lightweight branch-search state container."""

    problem_text: str
    pt_text: str
    pct_text: str
    accepted_facts: list[str] = field(default_factory=list)
    open_goals: list[str] = field(default_factory=list)
    candidate_moves: list[ExtractedCandidateMove] = field(default_factory=list)
    chosen_move_history: list[str] = field(default_factory=list)
    current_branch_text: str = ""
    score: float = 0.0
    depth: int = 0


@dataclass(frozen=True)
class SequentialLSSResult:
    """Outcome of the small-budget sequential LSS transition loop."""

    final_state: ReasoningState
    termination_reason: str
    depth_reached: int
    iteration_summaries: list[IterationSummary]
    log_events: list[str] = field(default_factory=list)


def _progress_vector(state: ReasoningState) -> dict[str, float]:
    """Compute compact progress signals for state-first sequential loop decisions."""

    unknowns = float(len(state.unknowns_remaining))
    bounds = float(len(state.bounds))
    invariants = float(len(state.invariants))
    cases = float(len(state.cases))
    finite_reduction = 1.0 if any(
        token in " ".join(state.candidate_strategies + state.strategy_tags).lower()
        for token in ("finite_search", "reduce_to_finite_search")
    ) else 0.0
    candidate_count = float(len(state.answer_candidates))
    return {
        "unknowns": unknowns,
        "bounds": bounds,
        "invariants": invariants,
        "cases": cases,
        "finite_reduction": finite_reduction,
        "answer_candidates": candidate_count,
    }


def _progress_delta(previous: ReasoningState, current: ReasoningState) -> tuple[float, dict[str, float]]:
    """Score useful state progress using a few explicit, easy-to-audit criteria."""

    before = _progress_vector(previous)
    after = _progress_vector(current)
    components = {
        "unknowns_reduced": max(0.0, before["unknowns"] - after["unknowns"]) * 2.0,
        "bounds_added": max(0.0, after["bounds"] - before["bounds"]) * 1.5,
        "invariants_added": max(0.0, after["invariants"] - before["invariants"]) * 1.5,
        "cases_added": max(0.0, after["cases"] - before["cases"]) * 1.0,
        "finite_search_reduction": max(0.0, after["finite_reduction"] - before["finite_reduction"]) * 2.0,
        "answer_set_reduced": max(0.0, before["answer_candidates"] - after["answer_candidates"]) * 2.0,
    }
    total = sum(components.values())
    if current.status == StateStatus.SOLVED and previous.status != StateStatus.SOLVED:
        components["solved_bonus"] = 3.0
        total += 3.0
    return total, components


def _terminal_reason_for_state(state: ReasoningState) -> str:
    """Map terminal states to stable termination reasons expected by policy/tests."""

    if state.status == StateStatus.SOLVED:
        tags = set(state.strategy_tags)
        if "high_priority" in tags or "high_priority_solved" in tags or state.score >= 1.0:
            return "high_priority_solved"
    return "terminal_state_reached"


def _extract_heading_sections(text: str) -> dict[str, list[str]]:
    """PHASE3_MOVE_EXTRACTION: parse heading-based free-text blocks into line items."""

    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":") and len(line) < 120:
            current = line[:-1].strip().lower()
            sections.setdefault(current, [])
            continue
        if not current:
            continue
        item = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        if item:
            sections[current].append(item)
    return sections


def _score_extracted_move(problem_text: str, move: ExtractedCandidateMove) -> float:
    """PHASE5_BRANCH_SEARCH: transparent local heuristic for candidate-move utility."""

    text = " ".join([move.move_text, move.why_it_helps, move.what_it_establishes]).lower()
    score = 0.0
    if len(move.move_text) >= 20:
        score += 0.5
    if any(token in text for token in ("=", "<=", ">=", "bound", "invariant", "case", "substitute", "eliminate")):
        score += 1.0
    if any(token in text for token in ("let ", "try ", "maybe ", "could ")):
        score -= 0.4
    if any(token in text for token in ("vague", "general", "strategy", "approach only")):
        score -= 0.4
    # Small relevance boost when move shares non-trivial terms with the problem.
    problem_tokens = {token for token in re.findall(r"[a-zA-Z]{4,}", problem_text.lower())}
    move_tokens = {token for token in re.findall(r"[a-zA-Z]{4,}", text)}
    if problem_tokens and move_tokens and problem_tokens.intersection(move_tokens):
        score += 0.5
    if move.what_it_establishes.strip():
        score += 0.4
    vague_phrases = (
        "derive a relation",
        "use substitution",
        "solve the equations",
        "check consistency",
    )
    if any(phrase in text for phrase in vague_phrases):
        score -= 1.5
    return score


def _is_vague_lss_move(move_text: str) -> bool:
    """Return whether an LSS move is generic/vague and should be filtered/downranked."""

    lowered = move_text.strip().lower()
    if not lowered:
        return True
    vague_phrases = (
        "derive a relation",
        "use substitution",
        "solve the equations",
        "check consistency",
    )
    if any(phrase in lowered for phrase in vague_phrases):
        return True
    if not any(token in lowered for token in ("=", "<=", ">=", "->", "substitute", "eliminate", "target", "derive")):
        return True
    return False


def _extract_candidate_moves_from_traces(
    *,
    problem_text: str,
    pct_text: str,
    lss_text: str,
    max_moves: int,
) -> list[ExtractedCandidateMove]:
    """PHASE3_MOVE_EXTRACTION: extract lightweight moves from PCT/LSS free-text sections."""

    extracted: list[ExtractedCandidateMove] = []

    pct_sections = _extract_heading_sections(pct_text)
    pct_moves = pct_sections.get("candidate approaches", [])
    pct_whys = pct_sections.get("why each approach might help", [])
    pct_establish = pct_sections.get("possible intermediate lemmas", []) + pct_sections.get("useful reformulations", [])
    for index, move_text in enumerate(pct_moves):
        move = ExtractedCandidateMove(
            move_text=move_text,
            why_it_helps=pct_whys[index] if index < len(pct_whys) else "",
            what_it_establishes=pct_establish[index] if index < len(pct_establish) else "",
            source_stage="pct",
        )
        extracted.append(replace(move, score=_score_extracted_move(problem_text, move)))

    lss_sections = _extract_heading_sections(lss_text)
    lss_moves = lss_sections.get("candidate next steps", [])
    lss_whys = lss_sections.get("why each step helps", [])
    lss_establish = lss_sections.get("what each step would establish", [])
    for index, move_text in enumerate(lss_moves):
        if _is_vague_lss_move(move_text):
            _debug_runtime_print(f"[runner][explore] lss_move_filtered reason=vague move={move_text!r}")
            continue
        move = ExtractedCandidateMove(
            move_text=move_text,
            why_it_helps=lss_whys[index] if index < len(lss_whys) else "",
            what_it_establishes=lss_establish[index] if index < len(lss_establish) else "",
            source_stage="lss",
        )
        extracted.append(replace(move, score=_score_extracted_move(problem_text, move)))

    # Graceful degradation: keep one generic move from raw text if headings were not captured.
    if not extracted:
        fallback_line = ""
        for line in lss_text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.endswith(":"):
                fallback_line = stripped
                break
        if fallback_line:
            move = ExtractedCandidateMove(
                move_text=fallback_line,
                why_it_helps="",
                what_it_establishes="",
                source_stage="lss",
            )
            extracted.append(replace(move, score=_score_extracted_move(problem_text, move)))

    extracted.sort(key=lambda item: item.score, reverse=True)
    return extracted[:max(1, max_moves)]


def _run_shallow_branch_search(
    *,
    cfg: SolveConfig,
    base_state: ReasoningState,
    pt_text: str,
    pct_text: str,
    lss_text: str,
    attempt_context: RunAttemptContext,
) -> ExploratoryReasoningState:
    """PHASE5_BRANCH_SEARCH: tiny branch exploration over extracted moves."""

    # PHASE5_BRANCH_SEARCH: keep backward-compatible alias support.
    max_initial = max(
        1,
        cfg.max_initial_candidate_moves
        if cfg.max_initial_candidate_moves != 3 or cfg.exploratory_max_initial_moves == 3
        else cfg.exploratory_max_initial_moves,
    )
    expand_top = max(
        1,
        cfg.max_branches_to_expand
        if cfg.max_branches_to_expand != 2 or cfg.exploratory_expand_top_branches == 2
        else cfg.exploratory_expand_top_branches,
    )
    max_depth = max(
        1,
        cfg.max_search_depth if cfg.max_search_depth != 2 or cfg.exploratory_max_depth == 2 else cfg.exploratory_max_depth,
    )
    all_moves = _extract_candidate_moves_from_traces(
        problem_text=base_state.raw_problem,
        pct_text=pct_text,
        lss_text=lss_text,
        max_moves=max_initial,
    )
    attempt_context.extracted_candidate_moves = [
        {
            "move_text": move.move_text,
            "why_it_helps": move.why_it_helps,
            "what_it_establishes": move.what_it_establishes,
            "source_stage": move.source_stage,
            "score": round(move.score, 3),
        }
        for move in all_moves
    ]
    _debug_runtime_print(
        f"[runner][explore] extracted_moves={len(all_moves)} "
        f"scores={[round(move.score, 2) for move in all_moves]}"
    )

    root = ExploratoryReasoningState(
        problem_text=base_state.raw_problem,
        pt_text=pt_text,
        pct_text=pct_text,
        accepted_facts=list(base_state.derived_facts),
        open_goals=list(base_state.open_goals),
        candidate_moves=list(all_moves),
        chosen_move_history=[],
        current_branch_text="",
        score=0.0,
        depth=0,
    )

    branches: list[ExploratoryReasoningState] = [root]
    for depth in range(1, max_depth + 1):
        expanded: list[ExploratoryReasoningState] = []
        for branch in branches[:expand_top]:
            for move in all_moves[:max_initial]:
                if move.move_text in branch.chosen_move_history:
                    attempt_context.pruning_decisions.append(
                        f"depth={depth} prune=duplicate_move move={move.move_text}"
                    )
                    continue
                next_branch = ExploratoryReasoningState(
                    problem_text=branch.problem_text,
                    pt_text=branch.pt_text,
                    pct_text=branch.pct_text,
                    accepted_facts=branch.accepted_facts
                    + ([move.what_it_establishes] if move.what_it_establishes.strip() else []),
                    open_goals=branch.open_goals,
                    candidate_moves=branch.candidate_moves,
                    chosen_move_history=branch.chosen_move_history + [move.move_text],
                    current_branch_text=(branch.current_branch_text + "\n" + move.move_text).strip(),
                    score=branch.score + move.score,
                    depth=depth,
                )
                expanded.append(next_branch)
                attempt_context.branch_decisions.append(
                    f"depth={depth} choose={move.move_text} score={next_branch.score:.2f}"
                )
                attempt_context.branch_scores.append(
                    f"depth={depth} branch_history={next_branch.chosen_move_history} score={next_branch.score:.2f}"
                )
        if not expanded:
            attempt_context.pruning_decisions.append(f"depth={depth} prune=no_expandable_branches")
            break
        expanded.sort(key=lambda item: item.score, reverse=True)
        if len(expanded) > expand_top:
            pruned = expanded[expand_top:]
            for item in pruned:
                attempt_context.pruning_decisions.append(
                    f"depth={depth} prune=top_k_cut score={item.score:.2f} branch_history={item.chosen_move_history}"
                )
        branches = expanded[:expand_top]
        _debug_runtime_print(
            f"[runner][explore] depth={depth} kept={len(branches)} "
            f"top_scores={[round(branch.score, 2) for branch in branches]}"
        )

    best = branches[0] if branches else root
    attempt_context.best_branch_summary = (
        f"depth={best.depth} score={best.score:.2f} history={best.chosen_move_history}"
    )
    return best


def _synthesize_final_attempt(
    *,
    state: ReasoningState,
    exploratory: ExploratoryReasoningState,
) -> str:
    """PHASE6_SYNTHESIS: build a coherent final draft from best branch context."""

    lines = [
        "SYNTHESIS DRAFT",
        "",
        "Problem:",
        state.raw_problem,
        "",
        "PT summary:",
        exploratory.pt_text.strip() or "(none)",
        "",
        "PCT summary:",
        exploratory.pct_text.strip() or "(none)",
        "",
        "Best branch move history:",
    ]
    if exploratory.chosen_move_history:
        lines.extend([f"- {item}" for item in exploratory.chosen_move_history])
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("Accepted facts from branch:")
    if exploratory.accepted_facts:
        lines.extend([f"- {item}" for item in exploratory.accepted_facts])
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _candidate_from_salvage_patch(patch: object) -> CandidateAction | None:
    """Best-effort conversion of optional validator salvage patch into CandidateAction."""

    if not isinstance(patch, dict):
        return None
    action_type_raw = str(patch.get("action_type", "")).strip().lower()
    title = str(patch.get("title", "")).strip()
    if not action_type_raw or not title:
        return None
    try:
        action_type = ActionType(action_type_raw)
    except ValueError:
        return None
    return CandidateAction(
        action_type=action_type,
        title=title,
        rationale=str(patch.get("rationale", "validator salvage patch")).strip() or "validator salvage patch",
        added_facts=[str(item).strip() for item in patch.get("added_facts", []) if str(item).strip()]
        if isinstance(patch.get("added_facts"), list)
        else [],
        added_constraints=[str(item).strip() for item in patch.get("added_constraints", []) if str(item).strip()]
        if isinstance(patch.get("added_constraints"), list)
        else [],
        inputs=[str(item).strip() for item in patch.get("inputs", []) if str(item).strip()]
        if isinstance(patch.get("inputs"), list)
        else [],
        outputs=[str(item).strip() for item in patch.get("outputs", []) if str(item).strip()]
        if isinstance(patch.get("outputs"), list)
        else [],
        branch_labels=[str(item).strip() for item in patch.get("branch_labels", []) if str(item).strip()]
        if isinstance(patch.get("branch_labels"), list)
        else [],
        metadata=dict(patch.get("metadata", {})) if isinstance(patch.get("metadata"), dict) else {},
    )


class DomainAwareProposer:
    """Wraps a proposer and applies domain plugin action enrichment."""

    def __init__(
        self,
        *,
        client: UnifiedLLMClient,
        plugins: list[DomainPlugin],
        max_candidates: int,
        cache_prefix: str,
        allow_expensive_branching: bool,
        attempt_context: RunAttemptContext,
    ) -> None:
        self._client = client
        self._plugins = plugins
        self._max_candidates = max_candidates
        self._cache_prefix = cache_prefix
        self._allow_expensive_branching = allow_expensive_branching
        self._attempt_context = attempt_context

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        _ = depth
        prompt = build_lss_prompt(state, max_candidates=self._max_candidates)
        if _debug_runtime_enabled():
            prompt_obj = parse_structured_json_object(prompt)
            _debug_runtime_print(
                f"[runner][lss] prompt_context={json.dumps(prompt_obj.get('context', {}), sort_keys=True)}"
            )
        state_key = f"{self._cache_prefix}:{_state_hash(state)}"
        cached = None if _stage_cache_disabled() else _STAGE_CACHE.get("lss", state_key)
        if cached is not None:
            _debug_runtime_print(
                f"[runner][lss] cache_hit key={state_key[:16]}... len={len(cached)}"
            )
        else:
            _debug_runtime_print(
                f"[runner][lss] cache_miss key={state_key[:16]}... calling {self._client.__class__.__name__}"
            )
            if _stage_cache_disabled():
                _debug_runtime_print("[runner][lss] cache_bypass enabled")
        raw = cached if cached is not None else self._client.generate_lss(prompt)
        self._attempt_context.lss_raw_outputs.append(raw)
        if cached is None and not _stage_cache_disabled():
            _STAGE_CACHE.set("lss", state_key, raw)
            _debug_runtime_print(f"[runner][lss] cache_store key={state_key[:16]}... len={len(raw)}")
        _debug_runtime_print(f"[runner][lss] raw_preview={raw[:240]!r}")
        _record_llm_fallback_metadata(raw, self._attempt_context)

        actions = parse_lss_output(raw)[: self._max_candidates]
        if not self._allow_expensive_branching:
            actions = [action for action in actions if action.action_type.value != "branch"]
        active = active_domain_plugins(state, self._plugins)
        for plugin in active:
            actions = plugin.enrich_actions(state, actions)

        accepted_signatures = _accepted_action_signatures(state)
        accepted_semantic_signatures = _accepted_semantic_action_signatures(state)
        filtered: list[CandidateAction] = []
        seen_signatures: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
        seen_semantic_signatures: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
        for action in actions:
            _debug_runtime_print(
                "[runner][lss] candidate "
                f"title={action.title!r} action_type={action.action_type.value} "
                f"added_facts={action.added_facts!r} added_constraints={action.added_constraints!r}"
            )
            signature = _action_signature(action)
            semantic_signature = _semantic_action_signature(action)
            if signature in accepted_signatures or signature in seen_signatures:
                _debug_runtime_print(
                    f"[runner][lss] duplicate_reject title={action.title!r} "
                    f"action_type={action.action_type.value} signature={signature}"
                )
                continue
            if (
                action.action_type.value in {"eliminate", "substitute"}
                and (semantic_signature in accepted_semantic_signatures or semantic_signature in seen_semantic_signatures)
            ):
                _debug_runtime_print(
                    f"[runner][lss] duplicate_reject title={action.title!r} "
                    f"action_type={action.action_type.value} reason=duplicate_action_semantic "
                    f"signature={semantic_signature}"
                )
                continue
            if _is_equation_restatement(state, action):
                _debug_runtime_print(
                    f"[runner][lss] no_new_information_reject title={action.title!r} "
                    f"action_type={action.action_type.value} reason=equation_restatement"
                )
                continue
            if _action_adds_no_new_information(state, action):
                _debug_runtime_print(
                    f"[runner][lss] no_new_information_reject title={action.title!r} "
                    f"action_type={action.action_type.value} signature={semantic_signature}"
                )
                continue
            filtered.append(action)
            seen_signatures.add(signature)
            seen_semantic_signatures.add(semantic_signature)

        scored: list[tuple[float, CandidateAction]] = []
        for action in filtered:
            action_score, components = _score_lss_action(state, action)
            _debug_runtime_print(
                f"[runner][lss] score title={action.title} total={action_score:.2f} components={components}"
            )
            scored.append((action_score, action))

        scored.sort(key=lambda item: (item[0], item[1].title), reverse=True)
        return [action for _, action in scored]


def _run_sequential_lss_loop(
    *,
    initial_state: ReasoningState,
    proposer: DomainAwareProposer,
    verifier: TrackingCompositeVerifier,
    transition_budget: int,
    experiment_mode: bool = False,
) -> SequentialLSSResult:
    """Run a fixed-budget sequential PT->PCT->LSS transition loop.

    Stop conditions:
    - terminal state reached
    - no valid/mergeable transition from current state
    - accepted transition produced no useful progress
    - transition budget exhausted
    """

    current = initial_state
    canonicalizer = DefaultStateCanonicalizer()
    summaries: list[IterationSummary] = []
    log_events: list[str] = []
    depth = 0
    weak_transition_streak = 0
    salvage_used = False

    while depth < transition_budget:
        if is_terminal_state(current):
            return SequentialLSSResult(
                final_state=current,
                termination_reason=_terminal_reason_for_state(current),
                depth_reached=depth,
                iteration_summaries=summaries,
                log_events=log_events,
            )

        candidates = proposer.propose(current, depth)
        merged_state: ReasoningState | None = None
        accepted_actions = 0
        candidate_count = len(candidates)
        accepted_score = 0.0
        accepted_components: dict[str, float] = {}
        accepted_unknowns_before = len(current.unknowns_remaining)
        accepted_unknowns_after = accepted_unknowns_before

        for action in candidates:
            log_events.append(f"lss.action_type={action.action_type.value}")
            if not verifier.is_action_valid(current, action):
                rejection = verifier.consume_last_action_rejection() or {}
                log_events.append(
                    f"validator.reject action={action.action_type.value} reason={rejection.get('reason', 'unknown')}"
                )
                _debug_runtime_print(
                    "[runner][lss] reject "
                    f"layer={rejection.get('layer', 'local_verifier_reject')} "
                    f"title={action.title!r} action_type={action.action_type.value} "
                    f"reason={rejection.get('reason', 'unknown')}"
                )
                salvage_patch = rejection.get("details", {}).get("salvage_patch") if isinstance(
                    rejection.get("details"), dict
                ) else None
                if experiment_mode and salvage_used:
                    continue
                salvage_action = _candidate_from_salvage_patch(salvage_patch)
                if salvage_action is None:
                    continue
                if not verifier.is_action_valid(current, salvage_action):
                    _debug_runtime_print(
                        f"[runner][lss] salvage_patch_reject title={salvage_action.title!r} "
                        f"action_type={salvage_action.action_type.value}"
                    )
                    continue
                salvage_used = True
                log_events.append(f"validator.salvage_accept action={salvage_action.action_type.value}")
                _debug_runtime_print(
                    f"[runner][lss] salvage_patch_accept title={salvage_action.title!r} "
                    f"action_type={salvage_action.action_type.value}"
                )
                action = salvage_action
            else:
                log_events.append(f"validator.accept action={action.action_type.value}")

            children = apply_action(current, action)
            valid_children: list[ReasoningState] = []
            for child in children:
                canonical = canonicalizer.canonicalize(child)
                if verifier.is_state_valid(canonical):
                    valid_children.append(canonical)
                    continue
                rejection = verifier.consume_last_state_rejection() or {}
                _debug_runtime_print(
                    "[runner][lss] state_transition_reject "
                    f"title={action.title!r} action_type={action.action_type.value} "
                    f"reason={rejection.get('reason', 'invalid_child_state')}"
                )

            if not valid_children:
                continue

            valid_children.sort(
                key=lambda child: (_progress_delta(current, child)[0], child.score),
                reverse=True,
            )
            merged_state = valid_children[0]
            accepted_actions = 1
            accepted_score, accepted_components = _progress_delta(current, merged_state)
            accepted_unknowns_after = len(merged_state.unknowns_remaining)
            break

        summaries.append(
            IterationSummary(
                depth=depth,
                candidate_actions=candidate_count,
                accepted_actions=accepted_actions,
                next_states=1 if merged_state is not None else 0,
                kept_after_beam=1 if merged_state is not None else 0,
            )
        )

        if merged_state is None:
            return SequentialLSSResult(
                final_state=current,
                termination_reason="no_useful_transition" if experiment_mode else "no_valid_next_states",
                depth_reached=depth + 1,
                iteration_summaries=summaries,
                log_events=log_events + ["stop_reason=no_useful_transition"],
            )

        progress_delta, progress_components = accepted_score, accepted_components
        log_events.append(
            f"validator.accept_score={progress_delta:.2f} unknowns_remaining:{accepted_unknowns_before}->{accepted_unknowns_after}"
        )
        _debug_runtime_print(
            f"[runner][lss] progress depth={depth} delta={progress_delta:.2f} components={progress_components}"
        )
        current = merged_state
        depth += 1

        if is_terminal_state(current):
            return SequentialLSSResult(
                final_state=current,
                termination_reason=_terminal_reason_for_state(current),
                depth_reached=depth,
                iteration_summaries=summaries,
                log_events=log_events,
            )

        if progress_delta <= 0:
            weak_transition_streak += 1
            if experiment_mode and weak_transition_streak >= 2:
                return SequentialLSSResult(
                    final_state=current,
                    termination_reason="repeated_weak_transitions",
                    depth_reached=depth,
                    iteration_summaries=summaries,
                    log_events=log_events + ["stop_reason=repeated_weak_transitions"],
                )
            return SequentialLSSResult(
                final_state=current,
                termination_reason="no_useful_progress",
                depth_reached=depth,
                iteration_summaries=summaries,
                log_events=log_events + ["stop_reason=no_useful_progress"],
            )
        weak_transition_streak = 0

    return SequentialLSSResult(
        final_state=current,
        termination_reason="transition_budget_reached",
        depth_reached=depth,
        iteration_summaries=summaries,
        log_events=log_events + ["stop_reason=transition_budget_reached"],
    )


def solve(
    raw_problem: str,
    *,
    config: SolveConfig | None = None,
    client: UnifiedLLMClient | None = None,
) -> SolveResult:
    """Run PT/PCT/LSS search with mode policy and explicit fallback behavior."""

    cfg = config or SolveConfig()
    llm_client = client or StubLLMClient()
    _debug_runtime_print(
        f"[runner] solve start target_type={cfg.target_type} requested_mode={cfg.requested_mode} "
        f"client={llm_client.__class__.__name__}"
    )
    if cfg.debug_single_path:
        print(
            "debug_single_path enabled: "
            "retries={pt:0,pct:0,lss:0}, token_caps={pt:512,pct:512,lss:256}, "
            "beam_width=1, transition_budget=1, candidate_cap_per_state=1, fallback_disabled=true"
        )
    if cfg.experiment_state_first:
        print(
            "experiment_state_first enabled: "
            "pt=1, pct=1, lss<=3, validator=per_step, endgame=1, fallback_disabled=true"
        )
    if cfg.exploratory_search:
        print(
            "exploratory_search enabled: "
            f"max_initial_candidate_moves={cfg.max_initial_candidate_moves}, "
            f"max_branches_to_expand={cfg.max_branches_to_expand}, "
            f"max_search_depth={cfg.max_search_depth}, "
            f"enable_verbose_trace={'true' if cfg.enable_verbose_trace else 'false'}, "
            f"enable_phase6_synthesis={'true' if cfg.enable_phase6_synthesis else 'false'}"
        )

    if not _within_session_budget(cfg.session_max_wall_time_s):
        return _budget_exhausted_solve_result(raw_problem=raw_problem, target_type=cfg.target_type)

    initial = create_initial_state(raw_problem=raw_problem, target_type=cfg.target_type)
    mode_selection = _resolve_mode_selection(raw_problem, initial, cfg, budget_pressure=0.0)
    primary = _run_attempt(
        problem_text=raw_problem,
        cfg=cfg,
        llm_client=llm_client,
        mode_selection=mode_selection,
        fallback_used=False,
        fallback_reason="",
    )

    pressure = _estimate_budget_pressure(primary.depth_reached, primary.termination_reason, cfg)
    fallback = select_fallback(
        termination_reason=primary.termination_reason,
        best_state=primary.best_state,
        budget_pressure=pressure,
        malformed_output_count=_extract_malformed_count(primary.policy_trace),
        current_mode=mode_selection.mode,
        config=cfg.policy_config,
    )
    explicit_deep_requested = cfg.requested_mode.strip().lower() == SolveMode.DEEP.value
    if (
        explicit_deep_requested
        and fallback.trigger
        and fallback.reason == "no_valid_branches_survived"
        and pressure < cfg.policy_config.budget_pressure_fallback_threshold
    ):
        return primary
    if cfg.experiment_state_first:
        return primary
    if cfg.debug_single_path:
        return primary
    if not fallback.trigger or fallback.fallback_mode is None:
        return primary

    fallback_selection = ModeSelection(mode=fallback.fallback_mode, reason=f"fallback:{fallback.reason}")
    second = _run_attempt(
        problem_text=raw_problem,
        cfg=cfg,
        llm_client=llm_client,
        mode_selection=fallback_selection,
        fallback_used=True,
        fallback_reason=fallback.reason,
    )
    return second


def format_solve_result(result: SolveResult) -> str:
    """Render human-readable console output for local solve runs."""

    state = result.best_state
    selected_facts = state.derived_facts[:5]

    lines = [
        f"final status: {state.status.value}",
        f"solve mode: {result.solve_mode}",
        f"fallback used: {'yes' if result.fallback_used else 'no'}",
        f"fallback reason: {result.fallback_reason or 'NA'}",
        f"score: {state.score:.2f}",
        f"termination: {result.termination_reason}",
        f"depth reached: {result.depth_reached}",
        f"answer status: {result.answer_status}",
        f"answer source: {result.answer_source or 'NA'}",
        f"predicted answer: {result.predicted_answer if result.predicted_answer is not None else 'NA'}",
        "derived facts:",
    ]
    if selected_facts:
        lines.extend(f"- {fact}" for fact in selected_facts)
    else:
        lines.append("- (none)")

    lines.append("accepted trace:")
    if result.trace_summary:
        lines.extend(f"- {line}" for line in result.trace_summary)
    else:
        lines.append("- (none)")

    return "\n".join(lines)


def run() -> ReasoningState:
    """Backward-compatible bootstrap runner used by existing smoke tests."""

    return solve("Placeholder bootstrap problem.").best_state


def _run_attempt(
    *,
    problem_text: str,
    cfg: SolveConfig,
    llm_client: UnifiedLLMClient,
    mode_selection: ModeSelection,
    fallback_used: bool,
    fallback_reason: str,
) -> SolveResult:
    """Execute one mode-configured solve attempt."""

    mode_settings = mode_settings_for(mode_selection.mode, cfg.policy_config)
    effective_candidate_cap = 1 if cfg.debug_single_path else min(
        cfg.max_candidates, mode_settings.max_candidates_per_state
    )
    active_client = _client_for_mode(llm_client, mode_settings, debug_single_path=cfg.debug_single_path)
    cache_prefix = f"{active_client.__class__.__name__}:{mode_selection.mode.value}"
    _debug_runtime_print(
        f"[runner] attempt mode={mode_selection.mode.value} reason={mode_selection.reason} "
        f"active_client={active_client.__class__.__name__} cache_prefix={cache_prefix}"
    )
    attempt_context = RunAttemptContext()
    policy_trace = [f"mode={mode_selection.mode.value}", f"mode_reason={mode_selection.reason}"]

    state = create_initial_state(raw_problem=problem_text, target_type=cfg.target_type)
    if mode_settings.use_pt:
        state = _run_pt_with_cache(state, active_client, cache_prefix=cache_prefix, attempt_context=attempt_context)
        policy_trace.append("stage:pt=used")
        if cfg.experiment_state_first:
            policy_trace.append(
                "pt_quality="
                f"objects:{len(state.objects)} relations:{len(state.relations)} constraints:{len(state.constraints)} "
                f"unknowns:{len(state.unknowns_remaining)}"
            )
    else:
        policy_trace.append("stage:pt=skipped")

    if mode_settings.use_pct:
        state, pct_update = _run_pct_with_cache(
            state,
            active_client,
            cache_prefix=cache_prefix,
            attempt_context=attempt_context,
        )
        policy_trace.append("stage:pct=used")
        if cfg.experiment_state_first:
            policy_trace.append(
                "pct_additions="
                f"tags:{len(pct_update.strategy_tags)} goals:{len(pct_update.open_goals)} "
                f"equations:{len(pct_update.candidate_equations)} answer_candidate:{pct_update.answer_candidate}"
            )
        pct_handoff = _maybe_accept_pct_answer_candidate(
            state=state,
            pct_update=pct_update,
            solve_mode=mode_selection.mode.value,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            policy_trace=policy_trace,
            attempt_context=attempt_context,
        )
        if pct_handoff is not None:
            _persist_phase1_trace_artifact(
                problem_text=problem_text,
                mode=mode_selection.mode,
                fallback_used=fallback_used,
                enable_verbose_trace=cfg.enable_verbose_trace,
                attempt_context=attempt_context,
                result=pct_handoff,
            )
            return pct_handoff
    else:
        pct_update = PCTUpdate()
        policy_trace.append("stage:pct=skipped")

    plugins = active_domain_plugins(state)
    for plugin in plugins:
        plugin.annotate_state(state)

    if not mode_settings.use_lss:
        state.status = StateStatus.DEAD_END
        if "lss_skipped" not in state.strategy_tags:
            state.strategy_tags.append("lss_skipped")
        decision = select_answer_across_states([state])
        result = SolveResult(
            best_state=state,
            trace_summary=_summarize_trace(state),
            termination_reason="lss_stage_skipped",
            depth_reached=0,
            predicted_answer=decision.predicted_answer,
            answer_status=decision.answer_status,
            answer_source=_answer_source_from_status(decision.answer_status, decision.predicted_answer),
            supporting_state_ids=list(decision.supporting_state_ids),
            supporting_trace_count=decision.supporting_trace_count,
            solve_mode=mode_selection.mode.value,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            policy_trace=policy_trace + [f"malformed_outputs={attempt_context.malformed_output_count}"],
            verifier_rejections_by_level={"local": 0, "consistency": 0, "global": 0, "domain": 0},
        )
        _persist_phase1_trace_artifact(
            problem_text=problem_text,
            mode=mode_selection.mode,
            fallback_used=fallback_used,
            enable_verbose_trace=cfg.enable_verbose_trace,
            attempt_context=attempt_context,
            result=result,
        )
        return result

    proposer = DomainAwareProposer(
        client=active_client,
        plugins=plugins,
        max_candidates=effective_candidate_cap,
        cache_prefix=cache_prefix,
        allow_expensive_branching=mode_settings.allow_expensive_branching,
        attempt_context=attempt_context,
    )
    tracking_verifier = TrackingCompositeVerifier(build_default_verifier(), domain_plugins=plugins)
    transition_budget = 1 if cfg.debug_single_path else max(
        1,
        min(
            3 if cfg.experiment_state_first else cfg.lss_transition_budget,
            cfg.max_depth,
            mode_settings.max_depth,
        ),
    )
    lss_result = _run_sequential_lss_loop(
        initial_state=state,
        proposer=proposer,
        verifier=tracking_verifier,
        transition_budget=transition_budget,
        experiment_mode=cfg.experiment_state_first,
    )
    if cfg.experiment_state_first:
        policy_trace.extend([f"lss.{event}" for event in lss_result.log_events])
        if attempt_context.malformed_output_count > 0 and lss_result.depth_reached <= 1:
            policy_trace.append("stop_reason=parsing_failure")

    best_state = lss_result.final_state
    if cfg.exploratory_search:
        # PHASE2_SOFT_STRUCTURE / PHASE5_BRANCH_SEARCH: treat PT/PCT/LSS as proposal traces.
        latest_lss_text = attempt_context.lss_raw_outputs[-1] if attempt_context.lss_raw_outputs else ""
        exploratory_state = _run_shallow_branch_search(
            cfg=cfg,
            base_state=best_state,
            pt_text=attempt_context.pt_raw_output,
            pct_text=attempt_context.pct_raw_output,
            lss_text=latest_lss_text,
            attempt_context=attempt_context,
        )
        for fact in exploratory_state.accepted_facts:
            if fact and fact not in best_state.derived_facts:
                best_state.derived_facts.append(fact)
        best_state.normalize_in_place()
        if cfg.enable_phase6_synthesis:
            attempt_context.final_synthesis_text = _synthesize_final_attempt(
                state=best_state,
                exploratory=exploratory_state,
            )
        policy_trace.append(
            f"exploratory_search moves={len(exploratory_state.candidate_moves)} "
            f"depth={exploratory_state.depth} score={exploratory_state.score:.2f}"
        )
        policy_trace.append(
            f"exploratory_knobs initial_moves={cfg.max_initial_candidate_moves} "
            f"expand_branches={cfg.max_branches_to_expand} max_depth={cfg.max_search_depth} "
            f"verbose_trace={'on' if cfg.enable_verbose_trace else 'off'} "
            f"phase6_synthesis={'on' if cfg.enable_phase6_synthesis else 'off'}"
        )
        _debug_runtime_print(
            f"[runner][explore] final_branch_depth={exploratory_state.depth} "
            f"final_branch_score={exploratory_state.score:.2f}"
        )
    answer_decision = select_answer_across_states([best_state])
    trace_summary = _summarize_trace(best_state)
    if attempt_context.final_synthesis_text:
        trace_summary = trace_summary + ["phase6_synthesis_available"]
    endgame_output = _maybe_run_endgame_stage(
        client=active_client,
        state=best_state,
        predicted_answer=answer_decision.predicted_answer,
        trace_summary=trace_summary,
        policy_trace=policy_trace,
    )
    if cfg.experiment_state_first:
        endgame_ready, _ = _endgame_readiness(best_state)
        readiness_score = 1.0 if endgame_ready else 0.0
        policy_trace.append(
            f"endgame.readiness_score={readiness_score:.2f} final_answer={endgame_output.answer if endgame_output.answer is not None else 'NA'}"
        )
        if not endgame_ready:
            policy_trace.append("stop_reason=endgame_not_ready")
    if endgame_output.answer is not None:
        answer_decision = BeamAnswerDecision(
            predicted_answer=endgame_output.answer,
            answer_status="predicted",
            supporting_state_ids=list(answer_decision.supporting_state_ids),
            supporting_trace_count=answer_decision.supporting_trace_count,
        )
        policy_trace.append("answer_source=endgame_llm")
        trace_summary = trace_summary + [f"endgame_answer_found: answer={endgame_output.answer}"]
    final_llm_answer = _maybe_run_final_llm_solve(
        client=active_client,
        state=best_state,
        predicted_answer=answer_decision.predicted_answer,
        policy_trace=policy_trace,
    )
    if final_llm_answer is not None:
        answer_decision = BeamAnswerDecision(
            predicted_answer=final_llm_answer,
            answer_status="solved",
            supporting_state_ids=list(answer_decision.supporting_state_ids),
            supporting_trace_count=answer_decision.supporting_trace_count,
        )
        policy_trace.append("answer_source=final_llm")
        trace_summary = trace_summary + [f"final_llm_answer_found: answer={final_llm_answer}"]

    policy_trace.append(f"malformed_outputs={attempt_context.malformed_output_count}")
    if attempt_context.llm_fallback_reasons:
        policy_trace.append(f"llm_fallback_reasons={','.join(attempt_context.llm_fallback_reasons)}")
    result = SolveResult(
        best_state=best_state,
        trace_summary=trace_summary,
        termination_reason=lss_result.termination_reason,
        depth_reached=lss_result.depth_reached,
        predicted_answer=answer_decision.predicted_answer,
        answer_status=answer_decision.answer_status,
        answer_source=(
            "final_llm"
            if final_llm_answer is not None
            else (
                "endgame_llm"
                if endgame_output.answer is not None
                else _answer_source_from_status(answer_decision.answer_status, answer_decision.predicted_answer)
            )
        ),
        supporting_state_ids=list(answer_decision.supporting_state_ids),
        supporting_trace_count=answer_decision.supporting_trace_count,
        solve_mode=mode_selection.mode.value,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        policy_trace=policy_trace,
        iteration_summaries=lss_result.iteration_summaries,
        verifier_rejections_by_level=dict(tracking_verifier.rejections_by_level),
    )
    _persist_phase1_trace_artifact(
        problem_text=problem_text,
        mode=mode_selection.mode,
        fallback_used=fallback_used,
        enable_verbose_trace=cfg.enable_verbose_trace,
        attempt_context=attempt_context,
        result=result,
    )
    return result


def _resolve_mode_selection(
    problem_text: str,
    state: ReasoningState,
    cfg: SolveConfig,
    *,
    budget_pressure: float,
) -> ModeSelection:
    """Resolve explicit or policy-selected solve mode."""

    requested = cfg.requested_mode.strip().lower()
    if requested in {SolveMode.FAST.value, SolveMode.BALANCED.value, SolveMode.DEEP.value}:
        return ModeSelection(mode=SolveMode(requested), reason="explicit_config")
    return select_solve_mode(problem_text, state, cfg.policy_config, budget_pressure=budget_pressure)


def _client_for_mode(
    client: UnifiedLLMClient,
    mode_settings: SolveModeSettings,
    *,
    debug_single_path: bool = False,
) -> UnifiedLLMClient:
    """Apply mode-specific client overrides when supported."""

    if isinstance(client, VLLMClient):
        if debug_single_path or client._runtime.debug_single_path:
            return client.with_overrides(
                retries=0,
                pt_retries=0,
                pct_retries=0,
                lss_retries=0,
                pt_max_tokens=512,
                pct_max_tokens=512,
                lss_max_tokens=256,
                debug_single_path=True,
            )
        return client.with_overrides(retries=mode_settings.llm_retries)
    return client


def reset_runtime_state() -> None:
    """Reset process-local caches and session budget tracker (test helper)."""

    global _SESSION_STARTED_AT
    _SESSION_STARTED_AT = None
    _STAGE_CACHE.clear()


def _select_best_terminal(controller_result: ControllerResult) -> tuple[ReasoningState, BeamAnswerDecision]:
    """Select representative final state using answer-aware terminal logic."""

    candidates = terminal_states(controller_result.final_beam) or list(controller_result.final_beam)
    decision = select_answer_across_states(candidates)

    solved = [state for state in candidates if state.status == StateStatus.SOLVED]
    if decision.predicted_answer is not None:
        matched: list[ReasoningState] = []
        target = str(decision.predicted_answer)
        for state in solved:
            for fact in state.derived_facts:
                rhs = fact.split("=")[-1].strip()
                if rhs == target:
                    matched.append(state)
                    break
        if matched:
            return max(matched, key=lambda state: state.score), decision

    if solved:
        return max(solved, key=lambda state: state.score), decision
    return max(candidates, key=lambda state: state.score), decision


def is_endgame_ready(state: ReasoningState) -> bool:
    """Return whether the state is strong enough for endgame/final solve calls."""

    ready, _ = _endgame_readiness(state)
    return ready


def _endgame_readiness(state: ReasoningState) -> tuple[bool, list[str]]:
    """Evaluate endgame readiness and return blocking reasons when not ready."""

    reasons: list[str] = []
    has_meaningful_step = _has_meaningful_accepted_step(state)
    has_concrete_fact = _has_concrete_algebraic_or_numeric_fact(state)
    has_explicit_numeric_bound = _has_explicit_numeric_bound(state)
    has_derived_equation = _has_derived_equation_beyond_translations(state)

    if not state.accepted_steps and not state.derived_facts and not has_explicit_numeric_bound and not has_derived_equation:
        reasons.append("accepted_steps_empty_and_no_derived_facts")
    if state.accepted_steps and _last_step_is_weak_restatement(state):
        reasons.append("last_step_weak_restatement")

    has_multi_step_progress = _has_multi_step_progress(state)
    counting_bound_required = _looks_like_counting_or_bound_problem(state)
    only_vague_constraints = _has_only_vague_constraints(state)

    if counting_bound_required and not has_explicit_numeric_bound:
        reasons.append("missing_explicit_numeric_bound")
    if only_vague_constraints and not (has_meaningful_step or has_concrete_fact or has_explicit_numeric_bound):
        reasons.append("only_vague_qualitative_constraints")
    if not any(
        [
            has_meaningful_step,
            has_derived_equation,
            has_concrete_fact,
            has_multi_step_progress,
            has_explicit_numeric_bound,
        ]
    ):
        reasons.append("no_concrete_reduction")

    return not reasons, reasons


def _has_derived_equation_beyond_translations(state: ReasoningState) -> bool:
    """Check for equation-like state that goes beyond direct PT constraint translations."""

    direct_relations = {
        str(item).strip()
        for item in list(state.domain_constraints) + list(state.global_constraints)
        if str(item).strip()
    }
    for equation in state.current_equations:
        text = str(equation).strip()
        if not _looks_equation_like_text(text):
            continue
        if text not in direct_relations:
            return True
    return False


def _has_concrete_algebraic_or_numeric_fact(state: ReasoningState) -> bool:
    """Check whether derived facts include concrete algebraic or numeric consequences."""

    return any(_is_concrete_relation_text(str(fact).strip()) for fact in state.derived_facts if str(fact).strip())


def _has_meaningful_accepted_step(state: ReasoningState) -> bool:
    """Check whether at least one accepted step added concrete new information."""

    for step in state.accepted_steps:
        values: list[str] = []
        added_facts = step.updates.get("added_facts", [])
        added_constraints = step.updates.get("added_constraints", [])
        if isinstance(added_facts, list):
            values.extend(str(item).strip() for item in added_facts if str(item).strip())
        if isinstance(added_constraints, list):
            values.extend(str(item).strip() for item in added_constraints if str(item).strip())
        if any(_is_concrete_relation_text(value) or _contains_explicit_numeric_bound(value) for value in values):
            return True
    return False


def _has_multi_step_progress(state: ReasoningState) -> bool:
    """Require at least two accepted steps and one explicit derived relation or numeric bound."""

    if len(state.accepted_steps) < 2:
        return False
    for step in state.accepted_steps:
        added_facts = step.updates.get("added_facts", [])
        added_constraints = step.updates.get("added_constraints", [])
        values = []
        if isinstance(added_facts, list):
            values.extend(str(item).strip() for item in added_facts if str(item).strip())
        if isinstance(added_constraints, list):
            values.extend(str(item).strip() for item in added_constraints if str(item).strip())
        if any(_is_concrete_relation_text(value) or _contains_explicit_numeric_bound(value) for value in values):
            return True
    return False


def _has_explicit_numeric_bound(state: ReasoningState) -> bool:
    """Detect explicit numeric bounds in facts, constraints, or equations."""

    texts = (
        list(state.derived_facts)
        + list(state.domain_constraints)
        + list(state.global_constraints)
        + list(state.current_equations)
    )
    return any(_contains_explicit_numeric_bound(str(text).strip()) for text in texts if str(text).strip())


def _has_only_vague_constraints(state: ReasoningState) -> bool:
    """Detect states whose known relations are only vague qualitative statements."""

    texts = (
        list(state.derived_facts)
        + list(state.domain_constraints)
        + list(state.global_constraints)
        + list(state.current_equations)
    )
    nonempty = [str(text).strip() for text in texts if str(text).strip()]
    if not nonempty:
        return True
    return all(_is_vague_qualitative_text(text) for text in nonempty)


def _last_step_is_weak_restatement(state: ReasoningState) -> bool:
    """Check whether the latest accepted step only added weak qualitative restatement."""

    if not state.accepted_steps:
        return False
    step = state.accepted_steps[-1]
    values: list[str] = []
    added_facts = step.updates.get("added_facts", [])
    added_constraints = step.updates.get("added_constraints", [])
    if isinstance(added_facts, list):
        values.extend(str(item).strip() for item in added_facts if str(item).strip())
    if isinstance(added_constraints, list):
        values.extend(str(item).strip() for item in added_constraints if str(item).strip())
    if not values:
        return True
    return all(_is_vague_qualitative_text(value) for value in values)


def _looks_like_counting_or_bound_problem(state: ReasoningState) -> bool:
    """Heuristic detector for counting/bound style problems."""

    haystacks = [
        state.raw_problem.lower(),
        " ".join(str(item).lower() for item in state.open_goals),
        " ".join(str(item).lower() for item in state.strategy_tags),
    ]
    keywords = ("count", "number", "maximum", "minimum", "bound", "perimeter", "distinct", "range")
    return any(any(keyword in haystack for keyword in keywords) for haystack in haystacks)


def _looks_equation_like_text(text: str) -> bool:
    """Detect equation-like text with an explicit relation operator."""

    return any(token in text for token in ("=", "<=", ">=", "<", ">"))


def _contains_explicit_numeric_bound(text: str) -> bool:
    """Detect explicit numeric upper/lower bounds and ranges."""

    lowered = text.lower()
    if not re.search(r"\d", lowered):
        return False
    bound_terms = ("at most", "at least", "less than", "greater than", "range from", "<=", ">=", "<", ">", "between")
    if any(term in lowered for term in bound_terms):
        return True
    return False


def _is_concrete_relation_text(text: str) -> bool:
    """Detect concrete relation text usable for endgame reasoning."""

    lowered = text.lower()
    if _looks_equation_like_text(text):
        return True
    if _contains_explicit_numeric_bound(text):
        return True
    concrete_terms = ("even", "odd", "divisible", "multiple", "mod", "residue")
    return any(term in lowered for term in concrete_terms) and bool(re.search(r"\d", lowered))


def _is_vague_qualitative_text(text: str) -> bool:
    """Detect weak qualitative restatements that should not trigger endgame."""

    lowered = text.lower()
    if _is_concrete_relation_text(text):
        return False
    vague_terms = (
        "bounded",
        "limited by the range",
        "limited",
        "range",
        "qualitative",
        "roughly",
        "tightly bounded",
        "can be analyzed",
        "useful",
        "may help",
    )
    return any(term in lowered for term in vague_terms) or not _looks_equation_like_text(text)


def _maybe_run_endgame_stage(
    *,
    client: UnifiedLLMClient,
    state: ReasoningState,
    predicted_answer: int | None,
    trace_summary: list[str],
    policy_trace: list[str],
) -> EndgameSolveOutput:
    """Run a single endgame solve pass from reduced state when trigger conditions are met."""

    if predicted_answer is not None:
        _debug_runtime_print("[runner][endgame] endgame_ready=false reasons=predicted_answer_already_present")
        return EndgameSolveOutput()
    ready, reasons = _endgame_readiness(state)
    _debug_runtime_print(
        f"[runner][endgame] endgame_ready={'true' if ready else 'false'} reasons={','.join(reasons) if reasons else 'ready'}"
    )
    if not ready:
        return EndgameSolveOutput()

    generate_endgame = getattr(client, "generate_endgame", None)
    if not callable(generate_endgame):
        return EndgameSolveOutput()

    policy_trace.append("endgame_called")
    prompt = build_endgame_solve_prompt(
        raw_problem=state.raw_problem,
        pt_target=state.open_goals[0] if state.open_goals else "",
        pt_constraints=list(state.domain_constraints) + list(state.global_constraints),
        current_equations=list(state.current_equations),
        derived_facts=list(state.derived_facts),
        open_goals=list(state.open_goals),
        strategy_tags=list(state.strategy_tags),
        trace_summary=list(trace_summary),
    )
    if _debug_runtime_enabled():
        prompt_obj = parse_structured_json_object(prompt)
        _debug_runtime_print(
            f"[runner][endgame] prompt_context={json.dumps(prompt_obj.get('context', {}), sort_keys=True)}"
        )
    raw = generate_endgame(prompt)
    if not isinstance(raw, str) or not raw:
        policy_trace.append("endgame_no_answer")
        return EndgameSolveOutput()
    _debug_runtime_print(f"[runner][endgame] raw_preview={raw[:240]!r}")
    output = parse_endgame_solve_output(raw)
    if output.answer is not None and output.ready and not output.missing_requirements:
        policy_trace.append("endgame_answer_found")
    else:
        if output.missing_requirements:
            _debug_runtime_print(
                f"[runner][endgame] answer_rejected missing_requirements={output.missing_requirements!r}"
            )
        policy_trace.append("endgame_no_answer")
    return output


def _maybe_run_final_llm_solve(
    *,
    client: UnifiedLLMClient,
    state: ReasoningState,
    predicted_answer: int | None,
    policy_trace: list[str],
) -> int | None:
    """Run one final direct-answer LLM solve from reduced state when available."""

    if predicted_answer is not None:
        _debug_runtime_print("[runner][final_llm] endgame_ready=false reasons=predicted_answer_already_present")
        return None

    ready, reasons = _endgame_readiness(state)
    _debug_runtime_print(
        f"[runner][final_llm] endgame_ready={'true' if ready else 'false'} reasons={','.join(reasons) if reasons else 'ready'}"
    )
    if not ready:
        return None

    prompt = json.dumps(
        {
            "task": "solve_final",
            "context": {
                "raw_problem": state.raw_problem,
                "equations": list(state.current_equations),
                "facts": list(state.derived_facts),
                "constraints": list(state.domain_constraints),
            },
            "instructions": [
                "Solve the problem using the provided equations and facts.",
                "Return ONLY JSON.",
                "Do not include explanations.",
                'Output format: {"ready": boolean, "answer": integer|null, "confidence": "high|medium|low", "justification": [], "missing_requirements": []}',
            ],
        },
        sort_keys=True,
    )

    raw = ""
    generate = getattr(client, "generate", None)
    if callable(generate):
        policy_trace.append("final_llm_called")
        raw = generate(
            prompt,
            system_prompt="You are a math solver. Return only a JSON object with a single integer answer.",
        )
    else:
        generate_endgame = getattr(client, "generate_endgame", None)
        if not callable(generate_endgame):
            return None
        policy_trace.append("final_llm_called")
        raw = generate_endgame(prompt)

    if not isinstance(raw, str) or not raw:
        policy_trace.append("final_llm_no_answer")
        return None
    _debug_runtime_print(f"[runner][final_llm] raw_preview={raw[:240]!r}")
    output = parse_endgame_solve_output(raw)
    if output.answer is None or not output.ready or output.missing_requirements:
        if output.missing_requirements:
            _debug_runtime_print(
                f"[runner][final_llm] answer_rejected missing_requirements={output.missing_requirements!r}"
            )
        policy_trace.append("final_llm_no_answer")
        return None
    policy_trace.append("final_llm_answer_found")
    return output.answer


def _run_pt_with_cache(
    state: ReasoningState,
    client: UnifiedLLMClient,
    *,
    cache_prefix: str,
    attempt_context: RunAttemptContext,
) -> ReasoningState:
    """Run PT stage with cache keyed by raw problem hash."""

    key = f"{cache_prefix}:{_problem_hash(state.raw_problem)}:{state.target_type}"
    cached = None if _stage_cache_disabled() else _STAGE_CACHE.get("pt", key)
    if cached is not None:
        _debug_runtime_print(f"[runner][pt] cache_hit key={key[:16]}... len={len(cached)}")
    else:
        _debug_runtime_print(
            f"[runner][pt] cache_miss key={key[:16]}... calling {client.__class__.__name__}"
        )
        if _stage_cache_disabled():
            _debug_runtime_print("[runner][pt] cache_bypass enabled")
    raw = cached if cached is not None else client.generate_pt(
        build_pt_prompt(raw_problem=state.raw_problem, target_type=state.target_type)
    )
    attempt_context.pt_raw_output = raw
    if cached is None and not _stage_cache_disabled():
        _STAGE_CACHE.set("pt", key, raw)
        _debug_runtime_print(f"[runner][pt] cache_store key={key[:16]}... len={len(raw)}")
    _debug_runtime_print(f"[runner][pt] raw_preview={raw[:240]!r}")
    _record_llm_fallback_metadata(raw, attempt_context)
    return apply_pt_update(state, parse_pt_output(raw))


def _run_pct_with_cache(
    state: ReasoningState,
    client: UnifiedLLMClient,
    *,
    cache_prefix: str,
    attempt_context: RunAttemptContext,
) -> tuple[ReasoningState, PCTUpdate]:
    """Run PCT stage with cache keyed by normalized state hash."""

    key = f"{cache_prefix}:{_state_hash(state)}"
    prompt = build_pct_prompt(state)
    if _debug_runtime_enabled():
        prompt_obj = parse_structured_json_object(prompt)
        _debug_runtime_print(
            f"[runner][pct] prompt_context={json.dumps(prompt_obj.get('context', {}), sort_keys=True)}"
        )
    cached = None if _stage_cache_disabled() else _STAGE_CACHE.get("pct", key)
    if cached is not None:
        _debug_runtime_print(f"[runner][pct] cache_hit key={key[:16]}... len={len(cached)}")
    else:
        _debug_runtime_print(
            f"[runner][pct] cache_miss key={key[:16]}... calling {client.__class__.__name__}"
        )
        if _stage_cache_disabled():
            _debug_runtime_print("[runner][pct] cache_bypass enabled")
    raw = cached if cached is not None else client.generate_pct(prompt)
    attempt_context.pct_raw_output = raw
    if cached is None and not _stage_cache_disabled():
        _STAGE_CACHE.set("pct", key, raw)
        _debug_runtime_print(f"[runner][pct] cache_store key={key[:16]}... len={len(raw)}")
    _debug_runtime_print(f"[runner][pct] raw_preview={raw[:240]!r}")
    _record_llm_fallback_metadata(raw, attempt_context)
    update = parse_pct_output(raw)
    return apply_pct_update(state, update), update


def _maybe_accept_pct_answer_candidate(
    *,
    state: ReasoningState,
    pct_update: PCTUpdate,
    solve_mode: str,
    fallback_used: bool,
    fallback_reason: str,
    policy_trace: list[str],
    attempt_context: RunAttemptContext,
) -> SolveResult | None:
    """Attempt early answer selection from a PCT answer candidate before LSS."""

    if pct_update.answer_candidate is None:
        return None

    policy_trace.append("pct_answer_candidate_detected")
    candidate_state = state.clone()
    candidate_state.status = StateStatus.SOLVED
    decision = select_answer_across_states([candidate_state])

    if decision.predicted_answer != pct_update.answer_candidate:
        policy_trace.append("pct_answer_candidate_rejected")
        return None

    policy_trace.append("pct_answer_candidate_accepted")
    if attempt_context.llm_fallback_reasons:
        policy_trace.append(f"llm_fallback_reasons={','.join(attempt_context.llm_fallback_reasons)}")
    trace_summary = [f"pct_answer_candidate_accepted: answer={pct_update.answer_candidate}"]
    return SolveResult(
        best_state=candidate_state,
        trace_summary=trace_summary,
        termination_reason="pct_answer_candidate_accepted",
        depth_reached=0,
        predicted_answer=decision.predicted_answer,
        answer_status=decision.answer_status,
        answer_source="pct_answer_candidate",
        supporting_state_ids=list(decision.supporting_state_ids),
        supporting_trace_count=decision.supporting_trace_count,
        solve_mode=solve_mode,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        policy_trace=policy_trace + [f"malformed_outputs={attempt_context.malformed_output_count}"],
        verifier_rejections_by_level={"local": 0, "consistency": 0, "global": 0, "domain": 0},
    )


def _problem_hash(raw_problem: str) -> str:
    """Stable hash key for problem-level caches."""

    return hashlib.sha256(raw_problem.strip().encode("utf-8")).hexdigest()


def _state_hash(state: ReasoningState) -> str:
    """Stable hash key for state-level caches."""

    payload = {
        "raw_problem": state.raw_problem,
        "target_type": state.target_type,
        "symbolic_objects": state.symbolic_objects,
        "current_equations": state.current_equations,
        "derived_facts": state.derived_facts,
        "domain_constraints": state.domain_constraints,
        "global_constraints": state.global_constraints,
        "witness_parameters": state.witness_parameters,
        "strategy_tags": state.strategy_tags,
        "open_goals": state.open_goals,
        "branch_assignments": state.branch_assignments,
        "status": state.status.value,
        "normalized_form": state.normalized_form,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _within_session_budget(session_max_wall_time_s: float) -> bool:
    """Check and initialize optional process-wide session wall-time budget."""

    global _SESSION_STARTED_AT
    if session_max_wall_time_s <= 0:
        return True

    now = time.monotonic()
    if _SESSION_STARTED_AT is None:
        _SESSION_STARTED_AT = now
        return True
    return (now - _SESSION_STARTED_AT) < session_max_wall_time_s


def _estimate_budget_pressure(depth_reached: int, termination_reason: str, cfg: SolveConfig) -> float:
    """Estimate remaining budget pressure for fallback routing."""

    depth_pressure = 0.0 if cfg.max_depth <= 0 else min(depth_reached / cfg.max_depth, 1.0)
    if termination_reason == "budget_exhausted":
        return 1.0
    if 0 < cfg.max_wall_time_s <= 1.0:
        return max(depth_pressure, 0.9)
    return depth_pressure


def _extract_malformed_count(policy_trace: list[str]) -> int:
    """Parse malformed-output count from policy trace entries."""

    for entry in policy_trace:
        if not entry.startswith("malformed_outputs="):
            continue
        raw = entry.split("=", 1)[1]
        try:
            return int(raw)
        except ValueError:
            return 0
    return 0


def _record_llm_fallback_metadata(raw_text: str, attempt_context: RunAttemptContext) -> None:
    """Record fallback metadata emitted by LLM backends."""

    obj = parse_structured_json_object(raw_text)
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict):
        return
    reason = metadata.get("fallback_reason")
    if not isinstance(reason, str) or not reason:
        return
    _debug_runtime_print(f"[runner] recorded_llm_fallback reason={reason}")
    attempt_context.malformed_output_count += 1
    attempt_context.llm_fallback_reasons.append(reason)


def _budget_exhausted_solve_result(*, raw_problem: str, target_type: str) -> SolveResult:
    """Return deterministic result payload when session budget is exhausted."""

    state = create_initial_state(raw_problem=raw_problem, target_type=target_type)
    state.status = StateStatus.DEAD_END
    state.strategy_tags.append("budget_exhausted")
    decision = select_answer_across_states([state])
    return SolveResult(
        best_state=state,
        trace_summary=[],
        termination_reason="session_budget_exhausted",
        depth_reached=0,
        predicted_answer=decision.predicted_answer,
        answer_status=decision.answer_status,
        answer_source=_answer_source_from_status(decision.answer_status, decision.predicted_answer),
        supporting_state_ids=list(decision.supporting_state_ids),
        supporting_trace_count=decision.supporting_trace_count,
        solve_mode="balanced",
        fallback_used=False,
        fallback_reason="",
        policy_trace=["session_budget_exhausted"],
        verifier_rejections_by_level={"local": 0, "consistency": 0, "global": 0, "domain": 0},
    )


def _summarize_trace(state: ReasoningState) -> list[str]:
    """Build compact accepted-step summaries for console and tests."""

    summaries: list[str] = []
    for step in state.accepted_steps:
        title = step.updates.get("title")
        added_facts = step.updates.get("added_facts", [])
        branch_label = step.updates.get("branch_label")
        suffix_parts: list[str] = []
        if branch_label:
            suffix_parts.append(f"branch={branch_label}")
        if isinstance(added_facts, list) and added_facts:
            suffix_parts.append(f"facts={','.join(str(item) for item in added_facts)}")
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        if isinstance(title, str) and title:
            summaries.append(f"{step.action}:{title}: {step.rationale}{suffix}")
            continue
        summaries.append(f"{step.action}: {step.rationale}{suffix}")
    return summaries


def _answer_source_from_status(answer_status: str, predicted_answer: int | None) -> str:
    """Map current answer resolution to a lightweight answer-source tag."""

    if predicted_answer is None:
        return ""
    if answer_status in {"unique_integer", "consensus_integer"}:
        return "beam_answering"
    if answer_status == "predicted":
        return "endgame_llm"
    return ""
