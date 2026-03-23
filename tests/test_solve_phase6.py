"""Phase 6 tests for end-to-end local solve runner and CLI wiring."""

import io
import unittest
from contextlib import redirect_stdout

from mgvs.cli.main import main
from mgvs.solve.runner import SolveConfig, solve
from mgvs.types import StateStatus


class TestSolvePhase6(unittest.TestCase):
    """Validates architecture-in-motion using deterministic stub pipeline."""

    def test_solve_happy_path(self) -> None:
        result = solve("Solve x + 1 = 2", config=SolveConfig(target_type="equation"))

        self.assertEqual(result.best_state.status, StateStatus.SOLVED)
        self.assertIn("x = 1", result.best_state.derived_facts)
        self.assertTrue(any("subtract_one_both_sides" in line for line in result.trace_summary))

    def test_solve_branch_path(self) -> None:
        result = solve("Branch demo: analyze two sign cases", config=SolveConfig())

        self.assertEqual(result.best_state.status, StateStatus.SOLVED)
        self.assertTrue(any("split_positive_negative" in line for line in result.trace_summary))

    def test_solve_contradiction_path(self) -> None:
        result = solve("Contradiction demo: inconsistent constraints", config=SolveConfig())

        self.assertEqual(result.best_state.status, StateStatus.CONTRADICTION)
        self.assertTrue(any("prune_contradiction" in line for line in result.trace_summary))

    def test_solve_parametric_path(self) -> None:
        result = solve("Parametric demo: represent family of solutions", config=SolveConfig())

        self.assertEqual(result.best_state.status, StateStatus.PARAMETRIC)
        self.assertTrue(any("introduce_parameterized_witness" in line for line in result.trace_summary))

    def test_cli_solve_command_outputs_summary(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["solve", "--problem", "Solve x + 1 = 2"])

        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("final status: solved", output)
        self.assertIn("accepted trace:", output)


if __name__ == "__main__":
    unittest.main()
