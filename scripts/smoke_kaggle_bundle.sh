#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${ROOT_DIR}/dist/kaggle_bundle"
WHEEL_DIR="${BUNDLE_DIR}/wheels"
IMPORT_DIR="$(mktemp -d)"
KAGGLE_SIM="$(mktemp -d)"

cleanup() {
  rm -rf "${IMPORT_DIR}" "${KAGGLE_SIM}"
}
trap cleanup EXIT

"${ROOT_DIR}/scripts/build_kaggle_bundle.sh"

test -d "${BUNDLE_DIR}"
test -f "${BUNDLE_DIR}/kaggle/README.md"
test -f "${BUNDLE_DIR}/kaggle/submission_notebook.ipynb"
test -f "${BUNDLE_DIR}/config/runtime_config.json"
test -f "${BUNDLE_DIR}/examples/handpicked_problems.json"
test -f "${BUNDLE_DIR}/examples/reference_numeric.csv"

DATASET_SLUG="mgvs-kaggle-bundle"
SIM_DATASET_DIR="${KAGGLE_SIM}/kaggle/input/${DATASET_SLUG}/kaggle_bundle"
mkdir -p "${SIM_DATASET_DIR}"
cp -R "${BUNDLE_DIR}/." "${SIM_DATASET_DIR}/"

WHEEL_PATH="$(ls "${WHEEL_DIR}"/*.whl | head -n 1)"
python -m pip install --no-deps --target "${IMPORT_DIR}" "${WHEEL_PATH}" >/dev/null

mkdir -p "${KAGGLE_SIM}/kaggle/input/competition"
cat > "${KAGGLE_SIM}/kaggle/input/competition/test.csv" <<CSV
id,problem
1,If x = 1, compute x.
2,Parametric demo: represent family of solutions
CSV

PYTHONPATH="${IMPORT_DIR}" KAGGLE_SIM_ROOT="${KAGGLE_SIM}/kaggle" python - <<'PY'
import csv
import json
import os
from pathlib import Path

from mgvs.solve.runner import SolveConfig, solve

root = Path(os.environ["KAGGLE_SIM_ROOT"])
dataset_slug = "mgvs-kaggle-bundle"
bundle_root = root / "input" / dataset_slug / "kaggle_bundle"
config_path = bundle_root / "config" / "runtime_config.json"
config_payload = json.loads(config_path.read_text(encoding="utf-8"))

cfg = SolveConfig(
    target_type=str(config_payload.get("target_type", "competition")),
    max_depth=int(config_payload.get("max_depth", 4)),
    beam_width=int(config_payload.get("beam_width", 3)),
    max_candidates=int(config_payload.get("max_candidates", 3)),
)

input_path = root / "input" / "competition" / "test.csv"
output_path = root / "working" / "submission.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
with input_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        problem = str(row.get("problem", "")).strip()
        result = solve(problem, config=cfg)
        prediction = ""
        if result.predicted_answer is not None:
            prediction = str(result.predicted_answer)
        rows.append((str(row.get("id", "")), prediction))

rows.sort(key=lambda item: item[0])
with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["id", "answer"])
    for row in rows:
        writer.writerow(row)

print(output_path)
PY

test -f "${KAGGLE_SIM}/kaggle/working/submission.csv"
line_count="$(wc -l < "${KAGGLE_SIM}/kaggle/working/submission.csv")"
test "${line_count}" -eq 3

printf 'Kaggle bundle smoke test passed.\n'
