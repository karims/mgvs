"""Smoke test for the CLI bootstrap entrypoint."""

import io
import unittest
from contextlib import redirect_stdout

from mgvs.cli.main import main


class TestCLI(unittest.TestCase):
    """CLI behavior checks for bootstrap scaffolding."""

    def test_main_returns_zero(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([])
        self.assertEqual(code, 0)
        self.assertIn("mgvs bootstrap ready", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
