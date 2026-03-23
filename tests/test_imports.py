"""Import smoke tests to ensure package skeleton is wired correctly."""

import unittest

from mgvs.solve.runner import run


class TestImports(unittest.TestCase):
    """Basic package import and execution checks."""

    def test_run_returns_state(self) -> None:
        state = run()
        self.assertEqual(state.state_id, "initial")


if __name__ == "__main__":
    unittest.main()
