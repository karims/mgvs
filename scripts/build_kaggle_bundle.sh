#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist
bundle_path="dist/mgvs_kaggle_bundle.zip"
rm -f "${bundle_path}"
zip -r "${bundle_path}" kaggle README.md src examples >/dev/null
printf 'Created %s\n' "${bundle_path}"
