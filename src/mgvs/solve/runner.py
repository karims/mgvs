"""Local end-to-end runner wiring PT/PCT/LSS, verification, and search."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace

from mgvs.actions.models import CandidateAction
from mgvs.config import RuntimeBudgetConfig, SolveModeSettings, SolvePolicyConfig
from mgvs.domains import active_domain_plugins
from mgvs.domains.base import DomainPlugin
from mgvs.llm.base import UnifiedLLMClient
from mgvs.llm.parser import (
    PCTUpdate,
    apply_pct_update,
    apply_pt_update,
    parse_lss_output,
    parse_pct_output,
    parse_pt_output,
    parse_structured_json_object,
)
from mgvs.llm.prompts import build_lss_prompt, build_pct_prompt, build_pt_prompt
from mgvs.llm.stub import StubLLMClient
from mgvs.llm.vllm_client import VLLMClient
from mgvs.search.controller import (
    ControllerConfig,
    ControllerResult,
    IterationSummary,
    StateCanonicalizer,
    run_search,
)
from mgvs.search.termination import terminal_states
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
    beam_width: int = 3
    max_candidates: int = 3
    max_wall_time_s: float = 20.0
    session_max_wall_time_s: float = 0.0
    requested_mode: str = "auto"
    policy_config: SolvePolicyConfig = field(default_factory=SolvePolicyConfig.default)

    @classmethod
    def from_env(cls, *, target_type: str = "unspecified") -> "SolveConfig":
        """Build solve config from runtime budget environment variables."""

        budget = RuntimeBudgetConfig.from_env()
        return cls(
            target_type=target_type,
            max_depth=budget.max_depth,
            beam_width=budget.beam_width,
            max_candidates=budget.candidate_action_cap_per_state,
            max_wall_time_s=budget.per_problem_max_wall_time_s,
            session_max_wall_time_s=budget.session_max_wall_time_s,
            requested_mode="auto",
            policy_config=SolvePolicyConfig.from_env(),
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

    def is_action_valid(self, state: ReasoningState, action: CandidateAction) -> bool:
        result = self._verifier.verify_action(state, action)
        self._record_rejections(result)
        if not result.passed:
            return False

        for plugin in self._domain_plugins:
            domain_result = plugin.validate_action(state, action)
            if not domain_result.passed:
                self.rejections_by_level["domain"] = self.rejections_by_level.get("domain", 0) + 1
                return False
        return True

    def is_state_valid(self, state: ReasoningState) -> bool:
        result = self._verifier.verify_state(state)
        self._record_rejections(result)
        if not result.passed:
            return False

        for plugin in self._domain_plugins:
            domain_result = plugin.validate_state(state)
            if not domain_result.passed:
                self.rejections_by_level["domain"] = self.rejections_by_level.get("domain", 0) + 1
                return False
        return True

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


def _stage_cache_disabled() -> bool:
    """Return whether stage cache should be bypassed for debugging."""

    return os.environ.get("MGVS_DISABLE_STAGE_CACHE") == "1"


@dataclass
class RunAttemptContext:
    """Mutable execution context for one solve attempt."""

    malformed_output_count: int = 0
    llm_fallback_reasons: list[str] = field(default_factory=list)


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
        return actions


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
    active_client = _client_for_mode(llm_client, mode_settings)
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
        return SolveResult(
            best_state=state,
            trace_summary=_summarize_trace(state),
            termination_reason="lss_stage_skipped",
            depth_reached=0,
            predicted_answer=decision.predicted_answer,
            answer_status=decision.answer_status,
            supporting_state_ids=list(decision.supporting_state_ids),
            supporting_trace_count=decision.supporting_trace_count,
            solve_mode=mode_selection.mode.value,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            policy_trace=policy_trace + [f"malformed_outputs={attempt_context.malformed_output_count}"],
            verifier_rejections_by_level={"local": 0, "consistency": 0, "global": 0, "domain": 0},
        )

    proposer = DomainAwareProposer(
        client=active_client,
        plugins=plugins,
        max_candidates=min(cfg.max_candidates, mode_settings.max_candidates_per_state),
        cache_prefix=cache_prefix,
        allow_expensive_branching=mode_settings.allow_expensive_branching,
        attempt_context=attempt_context,
    )
    tracking_verifier = TrackingCompositeVerifier(build_default_verifier(), domain_plugins=plugins)
    controller_result = run_search(
        state,
        proposer,
        verifier=tracking_verifier,
        canonicalizer=DefaultStateCanonicalizer(),
        config=ControllerConfig(
            max_depth=min(cfg.max_depth, mode_settings.max_depth),
            beam_width=min(cfg.beam_width, mode_settings.beam_width),
            max_wall_time_s=cfg.max_wall_time_s,
            candidate_cap_per_state=min(cfg.max_candidates, mode_settings.max_candidates_per_state),
        ),
    )

    best_state, answer_decision = _select_best_terminal(controller_result)
    policy_trace.append(f"malformed_outputs={attempt_context.malformed_output_count}")
    if attempt_context.llm_fallback_reasons:
        policy_trace.append(f"llm_fallback_reasons={','.join(attempt_context.llm_fallback_reasons)}")
    return SolveResult(
        best_state=best_state,
        trace_summary=_summarize_trace(best_state),
        termination_reason=controller_result.termination_reason,
        depth_reached=controller_result.depth_reached,
        predicted_answer=answer_decision.predicted_answer,
        answer_status=answer_decision.answer_status,
        supporting_state_ids=list(answer_decision.supporting_state_ids),
        supporting_trace_count=answer_decision.supporting_trace_count,
        solve_mode=mode_selection.mode.value,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        policy_trace=policy_trace,
        iteration_summaries=controller_result.iteration_summaries,
        verifier_rejections_by_level=dict(tracking_verifier.rejections_by_level),
    )


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


def _client_for_mode(client: UnifiedLLMClient, mode_settings: SolveModeSettings) -> UnifiedLLMClient:
    """Apply mode-specific client overrides when supported."""

    if isinstance(client, VLLMClient):
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
