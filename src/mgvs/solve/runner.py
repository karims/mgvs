"""Local end-to-end runner wiring PT/PCT/LSS, verification, and search."""

from __future__ import annotations

from dataclasses import dataclass, field

from mgvs.actions.models import CandidateAction
from mgvs.domains import active_domain_plugins
from mgvs.domains.base import DomainPlugin
from mgvs.llm.base import UnifiedLLMClient
from mgvs.llm.stub import StubLLMClient, StubPipelineProposer, run_pct_stage, run_pt_stage
from mgvs.llm.vllm_client import VLLMClient
from mgvs.search.controller import (
    ControllerConfig,
    ControllerResult,
    IterationSummary,
    StateCanonicalizer,
    run_search,
)
from mgvs.state.models import ReasoningState, create_initial_state
from mgvs.types import StateStatus
from mgvs.verify.base import CombinedVerificationResult, CompositeVerifier
from mgvs.verify.consistency import V0StateConsistencyVerifier
from mgvs.verify.global_ import V0GlobalCompatibilityVerifier
from mgvs.verify.local import V0LocalValidityVerifier

TERMINAL_STATUSES: set[StateStatus] = {
    StateStatus.SOLVED,
    StateStatus.CONTRADICTION,
    StateStatus.DEAD_END,
    StateStatus.PARAMETRIC,
}


@dataclass(frozen=True)
class SolveConfig:
    """Configuration for local solve orchestration."""

    target_type: str = "unspecified"
    max_depth: int = 4
    beam_width: int = 3
    max_candidates: int = 3


@dataclass(frozen=True)
class SolveResult:
    """Result payload from end-to-end local solve run."""

    best_state: ReasoningState
    trace_summary: list[str]
    termination_reason: str
    depth_reached: int
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
    if normalized == "vllm":
        return VLLMClient.from_env()
    return StubLLMClient()


class DomainAwareProposer:
    """Wraps a proposer and applies domain plugin action enrichment."""

    def __init__(self, base_proposer: StubPipelineProposer, plugins: list[DomainPlugin]) -> None:
        self._base_proposer = base_proposer
        self._plugins = plugins

    def propose(self, state: ReasoningState, depth: int) -> list[CandidateAction]:
        actions = self._base_proposer.propose(state, depth)
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
    """Run PT -> PCT -> LSS/controller and return best state and trace summary."""

    cfg = config or SolveConfig()
    llm_client = client or StubLLMClient()

    state = create_initial_state(raw_problem=raw_problem, target_type=cfg.target_type)
    state = run_pt_stage(state, llm_client)
    state = run_pct_stage(state, llm_client)
    plugins = active_domain_plugins(state)
    for plugin in plugins:
        plugin.annotate_state(state)

    base_proposer = StubPipelineProposer(llm_client, max_candidates=cfg.max_candidates)
    proposer = DomainAwareProposer(base_proposer, plugins)
    tracking_verifier = TrackingCompositeVerifier(build_default_verifier(), domain_plugins=plugins)
    controller_result = run_search(
        state,
        proposer,
        verifier=tracking_verifier,
        canonicalizer=DefaultStateCanonicalizer(),
        config=ControllerConfig(max_depth=cfg.max_depth, beam_width=cfg.beam_width),
    )

    best_state = _select_best_terminal(controller_result)
    return SolveResult(
        best_state=best_state,
        trace_summary=_summarize_trace(best_state),
        termination_reason=controller_result.termination_reason,
        depth_reached=controller_result.depth_reached,
        iteration_summaries=controller_result.iteration_summaries,
        verifier_rejections_by_level=dict(tracking_verifier.rejections_by_level),
    )


def format_solve_result(result: SolveResult) -> str:
    """Render human-readable console output for local solve runs."""

    state = result.best_state
    selected_facts = state.derived_facts[:5]

    lines = [
        f"final status: {state.status.value}",
        f"score: {state.score:.2f}",
        f"termination: {result.termination_reason}",
        f"depth reached: {result.depth_reached}",
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


def _select_best_terminal(controller_result: ControllerResult) -> ReasoningState:
    """Select highest scoring terminal state when available."""

    terminals = [state for state in controller_result.final_beam if state.status in TERMINAL_STATUSES]
    if terminals:
        return max(terminals, key=lambda state: state.score)
    return controller_result.best_state


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
