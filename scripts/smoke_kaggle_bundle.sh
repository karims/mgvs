#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${ROOT_DIR}/dist/kaggle_bundle"
WHEEL_DIR="${BUNDLE_DIR}/wheels"
IMPORT_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${IMPORT_DIR}"
}
trap cleanup EXIT

"${ROOT_DIR}/scripts/build_kaggle_bundle.sh"

test -d "${BUNDLE_DIR}"
test -f "${BUNDLE_DIR}/kaggle/README.md"
test -f "${BUNDLE_DIR}/kaggle/submission_notebook.ipynb"
test -f "${BUNDLE_DIR}/examples/handpicked_problems.json"

WHEEL_PATH="$(ls "${WHEEL_DIR}"/*.whl | head -n 1)"
python -m pip install --no-deps --target "${IMPORT_DIR}" "${WHEEL_PATH}" >/dev/null
PYTHONPATH="${IMPORT_DIR}" python -c "import mgvs; print(mgvs.__version__)"

printf 'Kaggle bundle smoke test passed.\n'
