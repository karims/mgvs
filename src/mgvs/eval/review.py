"""Trace review and failure-taxonomy reporting utilities for MGVS benchmark runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


CATEGORIES: tuple[str, ...] = (
    "correct",
    "wrong_answer",
    "pt_failure",
    "pct_failure",
    "lss_failure",
    "verifier_too_strict",
    "verifier_too_weak",
    "branch_explosion",
    "budget_exhausted",
    "answer_extraction_failure",
    "fallback_used_correct",
    "fallback_used_wrong",
    "parametric",
    "contradiction",
    "dead_end",
)


@dataclass(frozen=True)
class ReviewedRun:
    """Classification output for one problem run."""

    problem_id: str
    category: str
    numeric_correct: bool
    solve_mode: str
    fallback_used: bool
    depth_reached: int
    average_branch_fanout: float


@dataclass(frozen=True)
class ReviewReport:
    """Aggregate review report plus per-problem classifications."""

    reviewed_runs: list[ReviewedRun]
    counts_by_category: dict[str, int]
    correctness_by_mode: dict[str, dict[str, int]]
    fallback_frequency: float
    average_depth_by_category: dict[str, float]
    average_branch_by_category: dict[str, float]


def load_records(path: str) -> list[dict[str, object]]:
    """Load JSON or JSONL records from local path."""

    if path.endswith(".jsonl"):
        items: list[dict[str, object]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
        return items

    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    return []


def review_results(
    results_records: list[dict[str, object]],
    trace_records: list[dict[str, object]],
) -> ReviewReport:
    """Classify each run and compute compact taxonomy stats."""

    traces_by_id = {
        str(record.get("problem_id", "")): record
        for record in trace_records
        if str(record.get("problem_id", ""))
    }

    reviewed: list[ReviewedRun] = []
    for result in results_records:
        problem_id = str(result.get("problem_id", ""))
        trace = traces_by_id.get(problem_id, {})
        category = classify_run(result, trace)
        reviewed.append(
            ReviewedRun(
                problem_id=problem_id,
                category=category,
                numeric_correct=bool(result.get("numeric_correct", False)),
                solve_mode=str(result.get("solve_mode", "unknown")),
                fallback_used=bool(result.get("fallback_used", False)),
                depth_reached=int(result.get("depth_reached", 0) or 0),
                average_branch_fanout=float(result.get("average_branch_fanout", 0.0) or 0.0),
            )
        )

    counts = {cat: 0 for cat in CATEGORIES}
    for item in reviewed:
        counts[item.category] = counts.get(item.category, 0) + 1

    by_mode: dict[str, dict[str, int]] = {}
    fallback_count = 0
    for item in reviewed:
        bucket = by_mode.setdefault(item.solve_mode, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if item.numeric_correct:
            bucket["correct"] += 1
        if item.fallback_used:
            fallback_count += 1

    fallback_frequency = 0.0 if not reviewed else fallback_count / len(reviewed)

    depth_by_cat: dict[str, list[float]] = {}
    branch_by_cat: dict[str, list[float]] = {}
    for item in reviewed:
        depth_by_cat.setdefault(item.category, []).append(float(item.depth_reached))
        branch_by_cat.setdefault(item.category, []).append(float(item.average_branch_fanout))

    avg_depth = {
        cat: (sum(vals) / len(vals) if vals else 0.0)
        for cat, vals in depth_by_cat.items()
    }
    avg_branch = {
        cat: (sum(vals) / len(vals) if vals else 0.0)
        for cat, vals in branch_by_cat.items()
    }

    return ReviewReport(
        reviewed_runs=reviewed,
        counts_by_category=counts,
        correctness_by_mode=by_mode,
        fallback_frequency=fallback_frequency,
        average_depth_by_category=avg_depth,
        average_branch_by_category=avg_branch,
    )


def classify_run(result: dict[str, object], trace: dict[str, object]) -> str:
    """Heuristic classification of run outcome into failure/success taxonomy."""

    status = str(result.get("status", ""))
    numeric_correct = bool(result.get("numeric_correct", False))
    fallback_used = bool(result.get("fallback_used", False))
    answer_status = str(result.get("answer_status", ""))
    termination_reason = str(result.get("termination_reason", ""))
    avg_branch = float(result.get("average_branch_fanout", 0.0) or 0.0)

    verifier = result.get("verifier_rejections_by_level", {})
    verifier_counts = verifier if isinstance(verifier, dict) else {}
    verifier_sum = sum(int(value) for value in verifier_counts.values())

    policy_trace = result.get("policy_trace", [])
    policy_entries = policy_trace if isinstance(policy_trace, list) else []
    stage_pt_used = "stage:pt=used" in policy_entries
    stage_pct_used = "stage:pct=used" in policy_entries
    malformed_count = _extract_malformed_count(policy_entries)

    accepted_steps = result.get("accepted_steps", trace.get("accepted_steps", []))
    if isinstance(accepted_steps, list):
        accepted_count = len(accepted_steps)
    else:
        accepted_count = int(accepted_steps or 0)

    if fallback_used and numeric_correct:
        return "fallback_used_correct"
    if fallback_used and not numeric_correct:
        return "fallback_used_wrong"

    if termination_reason in {"budget_exhausted", "session_budget_exhausted"}:
        return "budget_exhausted"

    if status == "parametric":
        return "parametric"
    if status == "contradiction":
        return "contradiction"
    if status == "dead_end":
        return "dead_end"

    if numeric_correct:
        return "correct"

    if answer_status == "missing_answer" and status == "solved":
        return "answer_extraction_failure"

    if avg_branch >= 2.5:
        return "branch_explosion"

    if verifier_sum >= 5 and accepted_count == 0:
        return "verifier_too_strict"

    if status == "solved" and not numeric_correct and verifier_sum == 0:
        return "verifier_too_weak"

    if malformed_count >= 2 and accepted_count == 0:
        return "lss_failure"

    if stage_pt_used and accepted_count == 0 and status == "active":
        return "pt_failure"

    if stage_pct_used and not result.get("strategy_tags") and status == "active":
        return "pct_failure"

    if accepted_count == 0:
        return "lss_failure"

    return "wrong_answer"


def format_review_report(report: ReviewReport) -> str:
    """Create compact human-readable review summary."""

    lines = ["category counts:"]
    for category in CATEGORIES:
        count = report.counts_by_category.get(category, 0)
        if count:
            lines.append(f"- {category}: {count}")

    lines.append("correctness by mode:")
    for mode, stats in sorted(report.correctness_by_mode.items()):
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)
        lines.append(f"- {mode}: {correct}/{total} correct")

    lines.append(f"fallback frequency: {report.fallback_frequency:.2f}")
    lines.append("average depth by category:")
    for category, value in sorted(report.average_depth_by_category.items()):
        lines.append(f"- {category}: {value:.2f}")

    lines.append("average branch fanout by category:")
    for category, value in sorted(report.average_branch_by_category.items()):
        lines.append(f"- {category}: {value:.2f}")

    return "\n".join(lines)


def write_review_report(report: ReviewReport, output_path: str) -> None:
    """Write review report payload to a JSON file."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "counts_by_category": report.counts_by_category,
        "correctness_by_mode": report.correctness_by_mode,
        "fallback_frequency": report.fallback_frequency,
        "average_depth_by_category": report.average_depth_by_category,
        "average_branch_by_category": report.average_branch_by_category,
        "reviewed_runs": [
            {
                "problem_id": item.problem_id,
                "category": item.category,
                "numeric_correct": item.numeric_correct,
                "solve_mode": item.solve_mode,
                "fallback_used": item.fallback_used,
                "depth_reached": item.depth_reached,
                "average_branch_fanout": item.average_branch_fanout,
            }
            for item in report.reviewed_runs
        ],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _extract_malformed_count(policy_entries: list[object]) -> int:
    """Parse malformed count from policy trace entries."""

    for entry in policy_entries:
        if not isinstance(entry, str):
            continue
        if not entry.startswith("malformed_outputs="):
            continue
        raw = entry.split("=", 1)[1]
        try:
            return int(raw)
        except ValueError:
            return 0
    return 0
