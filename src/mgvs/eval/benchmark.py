"""Benchmark loader and evaluation runner for offline MGVS experiments."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass

from mgvs.eval.metrics import EvaluationMetrics, ProblemEvaluation, compute_metrics
from mgvs.solve.runner import SolveConfig, SolveResult, select_llm_client, solve


@dataclass(frozen=True)
class BenchmarkProblem:
    """A single benchmark row parsed from CSV input."""

    problem_id: str
    problem_text: str
    expected_answer: float | None


@dataclass(frozen=True)
class EvaluationReport:
    """Full evaluation output including per-problem records and metrics."""

    records: list[ProblemEvaluation]
    metrics: EvaluationMetrics


def load_benchmark_csv(
    path: str,
    *,
    problem_col: str,
    answer_col: str,
    id_col: str | None = None,
    limit: int | None = None,
) -> list[BenchmarkProblem]:
    """Load benchmark problems from CSV with configurable column names."""

    problems: list[BenchmarkProblem] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            problem_text = str(row.get(problem_col, "")).strip()
            if not problem_text:
                continue

            raw_id = row.get(id_col, "") if id_col else ""
            problem_id = str(raw_id).strip() or str(index)

            expected_raw = str(row.get(answer_col, "")).strip()
            expected = _parse_numeric(expected_raw)
            problems.append(
                BenchmarkProblem(
                    problem_id=problem_id,
                    problem_text=problem_text,
                    expected_answer=expected,
                )
            )

            if limit is not None and len(problems) >= limit:
                break

    return problems


def run_evaluation(
    problems: list[BenchmarkProblem],
    *,
    solve_config: SolveConfig | None = None,
    backend: str = "stub",
    export_traces_path: str | None = None,
) -> EvaluationReport:
    """Evaluate solver over a benchmark set and return per-problem records."""

    client = select_llm_client(backend)
    cfg = solve_config or SolveConfig()
    records: list[ProblemEvaluation] = []

    for problem in problems:
        result = solve(problem.problem_text, config=cfg, client=client)
        predicted = extract_numeric_prediction(result)

        numeric_correct = False
        if predicted is not None and problem.expected_answer is not None:
            numeric_correct = abs(predicted - problem.expected_answer) <= 1e-9

        avg_fanout = _average_branch_fanout(result)

        records.append(
            ProblemEvaluation(
                problem_id=problem.problem_id,
                status=result.best_state.status.value,
                expected_answer=problem.expected_answer,
                predicted_answer=predicted,
                numeric_correct=numeric_correct,
                depth_reached=result.depth_reached,
                accepted_steps=len(result.best_state.accepted_steps),
                average_branch_fanout=avg_fanout,
                verifier_rejections_by_level=dict(result.verifier_rejections_by_level),
            )
        )

        if export_traces_path:
            _export_trace(problem_id=problem.problem_id, result=result, output_path=export_traces_path)

    return EvaluationReport(records=records, metrics=compute_metrics(records))


def format_evaluation_report(report: EvaluationReport) -> str:
    """Render a compact human-readable metrics summary."""

    metrics = report.metrics
    lines = [
        f"total problems: {metrics.total_problems}",
        f"solved count: {metrics.solved_count}",
        f"correct numeric count: {metrics.correct_numeric_count}",
        f"contradiction count: {metrics.contradiction_count}",
        f"dead_end count: {metrics.dead_end_count}",
        f"parametric count: {metrics.parametric_count}",
        f"average search depth: {metrics.average_search_depth:.2f}",
        f"average accepted steps: {metrics.average_accepted_steps:.2f}",
        f"average branch fanout: {metrics.average_branch_fanout:.2f}",
        f"verifier rejections by level: {json.dumps(metrics.verifier_rejections_by_level, sort_keys=True)}",
    ]
    return "\n".join(lines)


def extract_numeric_prediction(result: SolveResult) -> float | None:
    """Extract a numeric prediction from final state facts if available."""

    # Prefer explicit equation-style derived facts, e.g. "x = 1".
    for fact in reversed(result.best_state.derived_facts):
        rhs = fact.split("=")[-1].strip()
        parsed = _parse_numeric(rhs)
        if parsed is not None:
            return parsed

    # Fallback: scan all derived facts and normalized form for any numeric token.
    haystacks = list(result.best_state.derived_facts)
    if result.best_state.normalized_form:
        haystacks.append(result.best_state.normalized_form)

    for text in reversed(haystacks):
        for token in re.findall(r"[-+]?\d*\.?\d+", text):
            parsed = _parse_numeric(token)
            if parsed is not None:
                return parsed

    return None


def _parse_numeric(value: str) -> float | None:
    """Parse numeric string safely; return None if unavailable."""

    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _average_branch_fanout(result: SolveResult) -> float:
    """Compute average branch fanout from accepted trace metadata."""

    fanouts: list[float] = []
    for step in result.best_state.accepted_steps:
        if step.action != "branch":
            continue
        raw = step.updates.get("branch_fanout")
        try:
            fanout = float(raw)
        except (TypeError, ValueError):
            fanout = 1.0
        fanouts.append(fanout)

    if not fanouts:
        return 0.0
    return sum(fanouts) / len(fanouts)


def _export_trace(*, problem_id: str, result: SolveResult, output_path: str) -> None:
    """Write per-problem trace export as JSON or JSONL."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    payload = {
        "problem_id": problem_id,
        "status": result.best_state.status.value,
        "score": result.best_state.score,
        "termination_reason": result.termination_reason,
        "depth_reached": result.depth_reached,
        "trace_summary": list(result.trace_summary),
        "accepted_steps": [
            {
                "action": step.action,
                "rationale": step.rationale,
                "updates": step.updates,
            }
            for step in result.best_state.accepted_steps
        ],
        "verifier_rejections_by_level": dict(result.verifier_rejections_by_level),
    }

    if output_path.endswith(".jsonl"):
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return

    if output_path.endswith(".json"):
        existing: list[dict[str, object]] = []
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as handle:
                try:
                    loaded = json.load(handle)
                except json.JSONDecodeError:
                    loaded = []
            if isinstance(loaded, list):
                existing = loaded
        existing.append(payload)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2, sort_keys=True)
        return

    raise ValueError("export_traces_path must end with .json or .jsonl")
