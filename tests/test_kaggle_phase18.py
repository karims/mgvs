"""Phase 18 tests for Kaggle thin-runner assets."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class TestKagglePhase18(unittest.TestCase):
    """Validates key offline bundle/notebook assets exist and are wired."""

    def test_runtime_config_exists_and_has_required_keys(self) -> None:
        path = Path("kaggle/runtime_config.json")
        self.assertTrue(path.exists())

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("target_type", payload)
        self.assertIn("max_depth", payload)
        self.assertIn("beam_width", payload)
        self.assertIn("max_candidates", payload)
        self.assertIn("submission", payload)

    def test_notebook_contains_kaggle_input_install_flow(self) -> None:
        path = Path("kaggle/submission_notebook.ipynb")
        self.assertTrue(path.exists())

        notebook = json.loads(path.read_text(encoding="utf-8"))
        source_blob = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if isinstance(cell, dict)
        )

        self.assertIn("/kaggle/input", source_blob)
        self.assertIn("runtime_config.json", source_blob)
        self.assertIn("submission.csv", source_blob)

    def test_build_script_bundles_config_and_examples(self) -> None:
        path = Path("scripts/build_kaggle_bundle.sh")
        text = path.read_text(encoding="utf-8")

        self.assertIn("config/runtime_config.json", text)
        self.assertIn("examples/reference_numeric.csv", text)
        self.assertIn("submission_notebook.ipynb", text)


if __name__ == "__main__":
    unittest.main()
