"""Phase 9 tests for benchmark loading, evaluation, and CLI integration."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mgvs.cli.main import main
from mgvs.eval.benchmark import (
    extract_numeric_prediction,
    format_evaluation_report,
    load_benchmark_csv,
    run_evaluation,
)
from mgvs.solve.runner import SolveConfig, solve


class TestEvalPhase9(unittest.TestCase):
    """Covers CSV benchmark workflow and aggregate metric reporting."""

    def setUp(self) -> None:
        self.fixture_path = Path("tests/fixtures/synthetic_eval.csv")

    def test_loader_with_configurable_columns(self) -> None:
        problems = load_benchmark_csv(
            str(self.fixture_path),
            problem_col="problem_text",
            answer_col="expected_answer",
            id_col="uid",
        )

        self.assertEqual(len(problems), 4)
        self.assertEqual(problems[0].problem_id, "p1")
        self.assertEqual(problems[0].expected_answer, 1.0)

    def test_run_evaluation_collects_metrics(self) -> None:
        problems = load_benchmark_csv(
            str(self.fixture_path),
            problem_col="problem_text",
            answer_col="expected_answer",
            id_col="uid",
        )

        report = run_evaluation(
            problems,
            solve_config=SolveConfig(target_type="competition"),
            backend="stub",
        )

        self.assertEqual(report.metrics.total_problems, 4)
        self.assertEqual(report.metrics.solved_count, 2)
        self.assertEqual(report.metrics.correct_numeric_count, 2)
        self.assertEqual(report.metrics.contradiction_count, 1)
        self.assertEqual(report.metrics.parametric_count, 1)

    def test_missing_numeric_prediction_is_safe(self) -> None:
        result = solve(
            "Parametric demo: represent family of solutions",
            config=SolveConfig(target_type="competition"),
        )

        prediction = extract_numeric_prediction(result)
        self.assertIsNone(prediction)

    def test_trace_export_jsonl(self) -> None:
        problems = load_benchmark_csv(
            str(self.fixture_path),
            problem_col="problem_text",
            answer_col="expected_answer",
            id_col="uid",
            limit=2,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "traces.jsonl"
            run_evaluation(
                problems,
                solve_config=SolveConfig(target_type="competition"),
                backend="stub",
                export_traces_path=str(out),
            )

            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            payload = json.loads(lines[0])
            self.assertIn("problem_id", payload)
            self.assertIn("trace_summary", payload)

    def test_cli_eval_command(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(
                [
                    "eval",
                    "--input",
                    str(self.fixture_path),
                    "--problem-col",
                    "problem_text",
                    "--answer-col",
                    "expected_answer",
                    "--id-col",
                    "uid",
                    "--backend",
                    "stub",
                    "--limit",
                    "2",
                ]
            )

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("total problems: 2", output)
        self.assertIn("correct numeric count", output)


if __name__ == "__main__":
    unittest.main()
