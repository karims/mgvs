"""Benchmark result comparison utilities for iterative MGVS tuning."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from mgvs.eval.review import load_records


@dataclass(frozen=True)
class RunSummary:
    """Aggregate summary for one results file."""

    label: str
    path: str
    total_problems: int
    correct_numeric_count: int
    solved_count: int
    status_distribution: dict[str, int]
    fallback_used_count: int
    fallback_used_rate: float
    average_depth: float
    average_branches: float
    average_runtime: float | None


@dataclass(frozen=True)
class ProblemDiff:
    """Per-problem baseline/candidate comparison record."""

    problem_id: str
    baseline_status: str
    candidate_status: str
    baseline_numeric_correct: bool
    candidate_numeric_correct: bool
    change: str


@dataclass(frozen=True)
class CandidateComparison:
    """Comparison payload for one candidate against baseline."""

    candidate: RunSummary
    improved_count: int
    regressed_count: int
    unchanged_count: int
    per_problem: list[ProblemDiff]


@dataclass(frozen=True)
class ComparisonReport:
    """Top-level comparison report for one baseline and one+ candidates."""

    baseline: RunSummary
    comparisons: list[CandidateComparison]


def compare_results_files(*, baseline_path: str, candidate_paths: list[str]) -> ComparisonReport:
    """Compare baseline results file against one or more candidate files."""

    baseline_records = load_records(baseline_path)
    baseline = summarize_records(label="baseline", path=baseline_path, records=baseline_records)

    comparisons: list[CandidateComparison] = []
    for index, candidate_path in enumerate(candidate_paths):
        candidate_records = load_records(candidate_path)
        candidate = summarize_records(
            label=f"candidate_{index + 1}",
            path=candidate_path,
            records=candidate_records,
        )
        comparisons.append(
            compare_problem_outcomes(
                baseline_records=baseline_records,
                candidate_records=candidate_records,
                candidate_summary=candidate,
            )
        )

    return ComparisonReport(baseline=baseline, comparisons=comparisons)


def summarize_records(*, label: str, path: str, records: list[dict[str, object]]) -> RunSummary:
    """Compute aggregate metrics for raw benchmark result records."""

    total = len(records)
    correct = 0
    solved = 0
    fallback = 0
    status_distribution: dict[str, int] = {}

    depth_values: list[float] = []
    branch_values: list[float] = []
    runtime_values: list[float] = []

    for record in records:
        status = str(record.get("status", "unknown"))
        status_distribution[status] = status_distribution.get(status, 0) + 1

        if bool(record.get("numeric_correct", False)):
            correct += 1
        if status == "solved":
            solved += 1
        if bool(record.get("fallback_used", False)):
            fallback += 1

        depth_values.append(_as_float(record.get("depth_reached"), default=0.0))
        branch_values.append(_as_float(record.get("average_branch_fanout"), default=0.0))

        runtime_value = _extract_runtime(record)
        if runtime_value is not None:
            runtime_values.append(runtime_value)

    avg_depth = 0.0 if total == 0 else sum(depth_values) / total
    avg_branches = 0.0 if total == 0 else sum(branch_values) / total
    avg_runtime = None if not runtime_values else (sum(runtime_values) / len(runtime_values))

    return RunSummary(
        label=label,
        path=path,
        total_problems=total,
        correct_numeric_count=correct,
        solved_count=solved,
        status_distribution=status_distribution,
        fallback_used_count=fallback,
        fallback_used_rate=(0.0 if total == 0 else fallback / total),
        average_depth=avg_depth,
        average_branches=avg_branches,
        average_runtime=avg_runtime,
    )


def compare_problem_outcomes(
    *,
    baseline_records: list[dict[str, object]],
    candidate_records: list[dict[str, object]],
    candidate_summary: RunSummary,
) -> CandidateComparison:
    """Compute per-problem and aggregate change labels for one candidate."""

    baseline_by_id = _records_by_problem_id(baseline_records)
    candidate_by_id = _records_by_problem_id(candidate_records)

    all_ids = sorted(set(baseline_by_id) | set(candidate_by_id))
    diffs: list[ProblemDiff] = []
    improved = 0
    regressed = 0
    unchanged = 0

    for problem_id in all_ids:
        base = baseline_by_id.get(problem_id, {})
        cand = candidate_by_id.get(problem_id, {})

        base_status = str(base.get("status", "missing"))
        cand_status = str(cand.get("status", "missing"))
        base_correct = bool(base.get("numeric_correct", False))
        cand_correct = bool(cand.get("numeric_correct", False))

        change = _classify_change(
            baseline_status=base_status,
            candidate_status=cand_status,
            baseline_correct=base_correct,
            candidate_correct=cand_correct,
        )

        if change == "improved":
            improved += 1
        elif change == "regressed":
            regressed += 1
        else:
            unchanged += 1

        diffs.append(
            ProblemDiff(
                problem_id=problem_id,
                baseline_status=base_status,
                candidate_status=cand_status,
                baseline_numeric_correct=base_correct,
                candidate_numeric_correct=cand_correct,
                change=change,
            )
        )

    return CandidateComparison(
        candidate=candidate_summary,
        improved_count=improved,
        regressed_count=regressed,
        unchanged_count=unchanged,
        per_problem=diffs,
    )


def format_comparison_report(report: ComparisonReport) -> str:
    """Render a compact human-readable comparison summary."""

    lines = [
        "baseline:",
        f"- path: {report.baseline.path}",
        f"- total problems: {report.baseline.total_problems}",
        f"- correct numeric answers: {report.baseline.correct_numeric_count}",
        f"- solved count: {report.baseline.solved_count}",
        f"- fallback usage: {report.baseline.fallback_used_count} ({report.baseline.fallback_used_rate:.2f})",
        f"- average depth: {report.baseline.average_depth:.2f}",
        f"- average branches: {report.baseline.average_branches:.2f}",
    ]
    if report.baseline.average_runtime is not None:
        lines.append(f"- average runtime: {report.baseline.average_runtime:.4f}s")

    for comparison in report.comparisons:
        lines.extend(
            [
                "candidate:",
                f"- path: {comparison.candidate.path}",
                f"- correct numeric answers: {comparison.candidate.correct_numeric_count}",
                f"- solved count: {comparison.candidate.solved_count}",
                f"- fallback usage: {comparison.candidate.fallback_used_count} ({comparison.candidate.fallback_used_rate:.2f})",
                f"- average depth: {comparison.candidate.average_depth:.2f}",
                f"- average branches: {comparison.candidate.average_branches:.2f}",
                f"- improved: {comparison.improved_count}",
                f"- regressed: {comparison.regressed_count}",
                f"- unchanged: {comparison.unchanged_count}",
            ]
        )
        if comparison.candidate.average_runtime is not None:
            lines.append(f"- average runtime: {comparison.candidate.average_runtime:.4f}s")

    return "\n".join(lines)


def write_comparison_report(report: ComparisonReport, output_path: str) -> None:
    """Write comparison report as a JSON artifact."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "baseline": _summary_payload(report.baseline),
        "comparisons": [
            {
                "candidate": _summary_payload(item.candidate),
                "improved_count": item.improved_count,
                "regressed_count": item.regressed_count,
                "unchanged_count": item.unchanged_count,
                "per_problem": [
                    {
                        "problem_id": diff.problem_id,
                        "baseline_status": diff.baseline_status,
                        "candidate_status": diff.candidate_status,
                        "baseline_numeric_correct": diff.baseline_numeric_correct,
                        "candidate_numeric_correct": diff.candidate_numeric_correct,
                        "change": diff.change,
                    }
                    for diff in item.per_problem
                ],
            }
            for item in report.comparisons
        ],
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _summary_payload(summary: RunSummary) -> dict[str, object]:
    return {
        "label": summary.label,
        "path": summary.path,
        "total_problems": summary.total_problems,
        "correct_numeric_count": summary.correct_numeric_count,
        "solved_count": summary.solved_count,
        "status_distribution": summary.status_distribution,
        "fallback_used_count": summary.fallback_used_count,
        "fallback_used_rate": summary.fallback_used_rate,
        "average_depth": summary.average_depth,
        "average_branches": summary.average_branches,
        "average_runtime": summary.average_runtime,
    }


def _as_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_runtime(record: dict[str, object]) -> float | None:
    for key in ("runtime_seconds", "runtime_sec", "solve_runtime_seconds"):
        if key not in record:
            continue
        value = _as_float(record.get(key), default=-1.0)
        if value >= 0.0:
            return value
    return None


def _records_by_problem_id(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for idx, record in enumerate(records):
        raw_problem_id = str(record.get("problem_id", "")).strip()
        problem_id = raw_problem_id if raw_problem_id else f"row_{idx}"
        indexed[problem_id] = record
    return indexed


def _classify_change(
    *,
    baseline_status: str,
    candidate_status: str,
    baseline_correct: bool,
    candidate_correct: bool,
) -> str:
    if candidate_correct and not baseline_correct:
        return "improved"
    if baseline_correct and not candidate_correct:
        return "regressed"

    baseline_rank = _status_rank(baseline_status)
    candidate_rank = _status_rank(candidate_status)

    if candidate_rank > baseline_rank:
        return "improved"
    if candidate_rank < baseline_rank:
        return "regressed"
    return "unchanged"


def _status_rank(status: str) -> int:
    ranks = {
        "solved": 4,
        "active": 3,
        "parametric": 2,
        "dead_end": 1,
        "contradiction": 0,
        "missing": -1,
    }
    return ranks.get(status, 0)
