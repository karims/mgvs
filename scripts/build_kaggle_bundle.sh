#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
BUNDLE_DIR="${DIST_DIR}/kaggle_bundle"
WHEEL_DIR="${BUNDLE_DIR}/wheels"
CONFIG_DIR="${BUNDLE_DIR}/config"
EXAMPLES_DIR="${BUNDLE_DIR}/examples"
KAGGLE_DIR="${BUNDLE_DIR}/kaggle"
ZIP_PATH="${DIST_DIR}/mgvs_kaggle_bundle.zip"

rm -rf "${BUNDLE_DIR}" "${ZIP_PATH}"
mkdir -p "${WHEEL_DIR}" "${CONFIG_DIR}" "${EXAMPLES_DIR}" "${KAGGLE_DIR}"

python -m pip wheel "${ROOT_DIR}" --no-deps --no-build-isolation --wheel-dir "${WHEEL_DIR}" >/dev/null

cp "${ROOT_DIR}/kaggle/runtime_config.json" "${CONFIG_DIR}/runtime_config.json"
cp "${ROOT_DIR}/examples/handpicked_problems.json" "${EXAMPLES_DIR}/"
cp "${ROOT_DIR}/examples/reference_numeric.csv" "${EXAMPLES_DIR}/"
cp "${ROOT_DIR}/kaggle/README.md" "${KAGGLE_DIR}/"
cp "${ROOT_DIR}/kaggle/submission_notebook.ipynb" "${KAGGLE_DIR}/"

cat > "${BUNDLE_DIR}/MANIFEST.txt" <<MANIFEST
mgvs Kaggle bundle contents:
- wheels/*.whl (installable package, includes mgvs code/prompt builders/parsers)
- config/runtime_config.json (default offline solve settings)
- examples/handpicked_problems.json (stub smoke examples)
- examples/reference_numeric.csv (tiny numeric eval sample)
- kaggle/README.md (upload + notebook usage docs)
- kaggle/submission_notebook.ipynb (thin submission scaffold)
MANIFEST

(
  cd "${DIST_DIR}"
  zip -rq "${ZIP_PATH}" "kaggle_bundle"
)

printf 'Created bundle directory: %s\n' "${BUNDLE_DIR}"
printf 'Created Kaggle upload zip: %s\n' "${ZIP_PATH}"
printf 'Bundle manifest: %s\n' "${BUNDLE_DIR}/MANIFEST.txt"
