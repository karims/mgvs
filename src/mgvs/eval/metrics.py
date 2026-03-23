"""Metric aggregation helpers for MGVS benchmark evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProblemEvaluation:
    """Per-problem evaluation record."""

    problem_id: str
    status: str
    expected_answer: float | None
    predicted_answer: float | None
    numeric_correct: bool
    depth_reached: int
    accepted_steps: int
    average_branch_fanout: float
    solve_mode: str = "balanced"
    fallback_used: bool = False
    termination_reason: str = ""
    answer_status: str = "missing_answer"
    policy_trace: list[str] = field(default_factory=list)
    strategy_tags: list[str] = field(default_factory=list)
    verifier_rejections_by_level: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate evaluation metrics across all problems."""

    total_problems: int
    solved_count: int
    correct_numeric_count: int
    contradiction_count: int
    dead_end_count: int
    parametric_count: int
    average_search_depth: float
    average_accepted_steps: float
    average_branch_fanout: float
    verifier_rejections_by_level: dict[str, int]


def compute_metrics(records: list[ProblemEvaluation]) -> EvaluationMetrics:
    """Compute aggregate benchmark metrics from per-problem records."""

    total = len(records)
    solved = sum(1 for rec in records if rec.status == "solved")
    correct_numeric = sum(1 for rec in records if rec.numeric_correct)
    contradiction = sum(1 for rec in records if rec.status == "contradiction")
    dead_end = sum(1 for rec in records if rec.status == "dead_end")
    parametric = sum(1 for rec in records if rec.status == "parametric")

    if total == 0:
        avg_depth = 0.0
        avg_steps = 0.0
        avg_fanout = 0.0
    else:
        avg_depth = sum(rec.depth_reached for rec in records) / total
        avg_steps = sum(rec.accepted_steps for rec in records) / total
        avg_fanout = sum(rec.average_branch_fanout for rec in records) / total

    rejection_totals: dict[str, int] = {}
    for rec in records:
        for level, count in rec.verifier_rejections_by_level.items():
            rejection_totals[level] = rejection_totals.get(level, 0) + int(count)

    return EvaluationMetrics(
        total_problems=total,
        solved_count=solved,
        correct_numeric_count=correct_numeric,
        contradiction_count=contradiction,
        dead_end_count=dead_end,
        parametric_count=parametric,
        average_search_depth=avg_depth,
        average_accepted_steps=avg_steps,
        average_branch_fanout=avg_fanout,
        verifier_rejections_by_level=rejection_totals,
    )
