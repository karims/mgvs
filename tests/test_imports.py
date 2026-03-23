"""Import smoke tests to ensure package skeleton is wired correctly."""

import unittest

from mgvs.solve.runner import run
from mgvs.types import StateStatus


class TestImports(unittest.TestCase):
    """Basic package import and execution checks."""

    def test_run_returns_state(self) -> None:
        state = run()
        self.assertEqual(state.status, StateStatus.ACTIVE)
        self.assertEqual(state.target_type, "unspecified")


if __name__ == "__main__":
    unittest.main()
