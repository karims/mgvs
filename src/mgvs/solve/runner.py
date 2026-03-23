"""Local end-to-end runner wiring PT/PCT/LSS, verification, and search."""

from __future__ import annotations

from dataclasses import dataclass

from mgvs.llm.base import UnifiedLLMClient
from mgvs.llm.stub import StubLLMClient, StubPipelineProposer, run_pct_stage, run_pt_stage
from mgvs.search.controller import ControllerConfig, ControllerResult, StateCanonicalizer, run_search
from mgvs.state.models import ReasoningState, create_initial_state
from mgvs.types import StateStatus
from mgvs.verify.base import CompositeVerifier
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


def build_default_verifier() -> CompositeVerifier:
    """Construct the default three-level composite verifier."""

    return CompositeVerifier(
        local_verifier=V0LocalValidityVerifier(),
        consistency_verifier=V0StateConsistencyVerifier(),
        global_verifier=V0GlobalCompatibilityVerifier(),
    )


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

    proposer = StubPipelineProposer(llm_client, max_candidates=cfg.max_candidates)
    controller_result = run_search(
        state,
        proposer,
        verifier=build_default_verifier(),
        canonicalizer=DefaultStateCanonicalizer(),
        config=ControllerConfig(max_depth=cfg.max_depth, beam_width=cfg.beam_width),
    )

    best_state = _select_best_terminal(controller_result)
    return SolveResult(
        best_state=best_state,
        trace_summary=_summarize_trace(best_state),
        termination_reason=controller_result.termination_reason,
        depth_reached=controller_result.depth_reached,
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
