"""Phase 14 tests for review taxonomy and reporting pipeline."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mgvs.cli.main import main
from mgvs.eval.review import format_review_report, load_records, review_results


class TestReviewPhase14(unittest.TestCase):
    """Covers synthetic taxonomy classification and CLI review flow."""

    def _write_jsonl(self, path: Path, items: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    def test_review_classification_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            results_path = tmp / "results.jsonl"
            traces_path = tmp / "traces.jsonl"

            self._write_jsonl(
                results_path,
                [
                    {
                        "problem_id": "p1",
                        "status": "solved",
                        "numeric_correct": True,
                        "solve_mode": "fast",
                        "fallback_used": False,
                        "depth_reached": 1,
                        "average_branch_fanout": 0.0,
                        "termination_reason": "high_priority_solved",
                        "answer_status": "unique_integer",
                        "accepted_steps": 1,
                        "policy_trace": ["stage:pt=used", "malformed_outputs=0"],
                        "strategy_tags": ["s"],
                        "verifier_rejections_by_level": {"local": 0},
                    },
                    {
                        "problem_id": "p2",
                        "status": "solved",
                        "numeric_correct": False,
                        "solve_mode": "balanced",
                        "fallback_used": True,
                        "depth_reached": 3,
                        "average_branch_fanout": 1.2,
                        "termination_reason": "no_valid_next_states",
                        "answer_status": "missing_answer",
                        "accepted_steps": 0,
                        "policy_trace": ["stage:pt=used", "malformed_outputs=2"],
                        "strategy_tags": ["s"],
                        "verifier_rejections_by_level": {"local": 0},
                    },
                    {
                        "problem_id": "p3",
                        "status": "dead_end",
                        "numeric_correct": False,
                        "solve_mode": "deep",
                        "fallback_used": False,
                        "depth_reached": 4,
                        "average_branch_fanout": 3.0,
                        "termination_reason": "budget_exhausted",
                        "answer_status": "missing_answer",
                        "accepted_steps": 0,
                        "policy_trace": ["stage:pt=used", "malformed_outputs=0"],
                        "strategy_tags": [],
                        "verifier_rejections_by_level": {"local": 3, "consistency": 3},
                    },
                ],
            )

            self._write_jsonl(
                traces_path,
                [
                    {"problem_id": "p1", "accepted_steps": [{"a": 1}]},
                    {"problem_id": "p2", "accepted_steps": []},
                    {"problem_id": "p3", "accepted_steps": []},
                ],
            )

            report = review_results(load_records(str(results_path)), load_records(str(traces_path)))
            text = format_review_report(report)

            self.assertEqual(report.counts_by_category["correct"], 1)
            self.assertEqual(report.counts_by_category["fallback_used_wrong"], 1)
            self.assertEqual(report.counts_by_category["budget_exhausted"], 1)
            self.assertIn("correctness by mode", text)
            self.assertGreater(report.fallback_frequency, 0.0)

    def test_cli_review_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            results = tmp / "results.jsonl"
            traces = tmp / "traces.jsonl"
            output = tmp / "review.json"

            self._write_jsonl(
                results,
                [
                    {
                        "problem_id": "q1",
                        "status": "solved",
                        "numeric_correct": True,
                        "solve_mode": "fast",
                        "fallback_used": False,
                        "depth_reached": 1,
                        "average_branch_fanout": 0.0,
                        "termination_reason": "high_priority_solved",
                        "answer_status": "unique_integer",
                        "accepted_steps": 1,
                        "policy_trace": ["stage:pt=used", "malformed_outputs=0"],
                        "strategy_tags": ["s"],
                        "verifier_rejections_by_level": {"local": 0},
                    }
                ],
            )
            self._write_jsonl(traces, [{"problem_id": "q1", "accepted_steps": [{"x": 1}]}])

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(
                    [
                        "review",
                        "--results",
                        str(results),
                        "--traces",
                        str(traces),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("counts_by_category", payload)
            self.assertIn("reviewed_runs", payload)


if __name__ == "__main__":
    unittest.main()
