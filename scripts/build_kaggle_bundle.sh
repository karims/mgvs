#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
BUNDLE_DIR="${DIST_DIR}/kaggle_bundle"
WHEEL_DIR="${BUNDLE_DIR}/wheels"
ZIP_PATH="${DIST_DIR}/mgvs_kaggle_bundle.zip"

rm -rf "${BUNDLE_DIR}" "${ZIP_PATH}"
mkdir -p "${WHEEL_DIR}" "${BUNDLE_DIR}/examples" "${BUNDLE_DIR}/kaggle"

python -m pip wheel "${ROOT_DIR}" --no-deps --no-build-isolation --wheel-dir "${WHEEL_DIR}" >/dev/null

cp "${ROOT_DIR}/examples/handpicked_problems.json" "${BUNDLE_DIR}/examples/"
cp "${ROOT_DIR}/kaggle/README.md" "${BUNDLE_DIR}/kaggle/"
cp "${ROOT_DIR}/kaggle/submission_notebook.ipynb" "${BUNDLE_DIR}/kaggle/"

cat > "${BUNDLE_DIR}/MANIFEST.txt" <<MANIFEST
mgvs Kaggle bundle contents:
- wheels/*.whl
- examples/handpicked_problems.json
- kaggle/README.md
- kaggle/submission_notebook.ipynb
MANIFEST

(
  cd "${DIST_DIR}"
  zip -rq "${ZIP_PATH}" "kaggle_bundle"
)

printf 'Created bundle directory: %s\n' "${BUNDLE_DIR}"
printf 'Created Kaggle upload zip: %s\n' "${ZIP_PATH}"
