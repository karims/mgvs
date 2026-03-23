"""Phase 17 tests for benchmark comparison workflow and CLI integration."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mgvs.cli.main import main
from mgvs.eval.compare import compare_results_files, format_comparison_report


class TestComparePhase17(unittest.TestCase):
    """Covers aggregate and per-problem result-file comparison behavior."""

    def _write_jsonl(self, path: Path, items: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    def test_compare_results_with_improvement_and_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            baseline = tmp / "baseline.jsonl"
            candidate = tmp / "candidate.jsonl"

            self._write_jsonl(
                baseline,
                [
                    {
                        "problem_id": "p1",
                        "status": "solved",
                        "numeric_correct": False,
                        "fallback_used": False,
                        "depth_reached": 3,
                        "average_branch_fanout": 1.2,
                        "runtime_seconds": 1.2,
                    },
                    {
                        "problem_id": "p2",
                        "status": "solved",
                        "numeric_correct": True,
                        "fallback_used": False,
                        "depth_reached": 2,
                        "average_branch_fanout": 0.5,
                        "runtime_seconds": 0.8,
                    },
                    {
                        "problem_id": "p3",
                        "status": "dead_end",
                        "numeric_correct": False,
                        "fallback_used": True,
                        "depth_reached": 4,
                        "average_branch_fanout": 2.0,
                    },
                ],
            )

            self._write_jsonl(
                candidate,
                [
                    {
                        "problem_id": "p1",
                        "status": "solved",
                        "numeric_correct": True,
                        "fallback_used": False,
                        "depth_reached": 2,
                        "average_branch_fanout": 1.0,
                        "runtime_seconds": 1.0,
                    },
                    {
                        "problem_id": "p2",
                        "status": "dead_end",
                        "numeric_correct": False,
                        "fallback_used": True,
                        "depth_reached": 3,
                        "average_branch_fanout": 0.8,
                        "runtime_seconds": 0.9,
                    },
                    {
                        "problem_id": "p3",
                        "status": "dead_end",
                        "numeric_correct": False,
                        "fallback_used": True,
                        "depth_reached": 4,
                        "average_branch_fanout": 1.8,
                    },
                ],
            )

            report = compare_results_files(
                baseline_path=str(baseline),
                candidate_paths=[str(candidate)],
            )

            self.assertEqual(report.baseline.correct_numeric_count, 1)
            self.assertEqual(report.comparisons[0].candidate.correct_numeric_count, 1)
            self.assertEqual(report.comparisons[0].improved_count, 1)
            self.assertEqual(report.comparisons[0].regressed_count, 1)
            self.assertEqual(report.comparisons[0].unchanged_count, 1)

            text = format_comparison_report(report)
            self.assertIn("improved: 1", text)
            self.assertIn("regressed: 1", text)
            self.assertIn("average runtime", text)

    def test_cli_compare_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            baseline = tmp / "a.json"
            candidate = tmp / "b.json"
            output = tmp / "compare.json"

            baseline.write_text(
                json.dumps(
                    [
                        {
                            "problem_id": "q1",
                            "status": "solved",
                            "numeric_correct": False,
                            "fallback_used": False,
                            "depth_reached": 2,
                            "average_branch_fanout": 1.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    [
                        {
                            "problem_id": "q1",
                            "status": "solved",
                            "numeric_correct": True,
                            "fallback_used": False,
                            "depth_reached": 1,
                            "average_branch_fanout": 0.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(
                    [
                        "compare",
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(output.exists())

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("baseline", payload)
            self.assertIn("comparisons", payload)
            self.assertEqual(payload["comparisons"][0]["improved_count"], 1)
            self.assertIn("candidate:", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
