# MGVS Kaggle Offline Runner

This folder provides a thin, reproducible Kaggle workflow where all solver logic stays in the `mgvs` package.

## Bundle Build and Upload Flow

1. Build the Kaggle bundle locally:

```bash
scripts/build_kaggle_bundle.sh
```

2. Upload `dist/mgvs_kaggle_bundle.zip` as a Kaggle Dataset.

3. Attach that dataset to your Kaggle notebook.

Expected dataset layout after upload:
- `kaggle_bundle/wheels/*.whl`
- `kaggle_bundle/config/runtime_config.json`
- `kaggle_bundle/examples/handpicked_problems.json`
- `kaggle_bundle/examples/reference_numeric.csv`
- `kaggle_bundle/kaggle/submission_notebook.ipynb`
- `kaggle_bundle/kaggle/README.md`

## Notebook Attachment and Install Flow

Inside the notebook:

1. Set your dataset slug (`MGVS_DATASET_SLUG`).
2. Install the wheel from `/kaggle/input/<slug>/kaggle_bundle/wheels/*.whl`.
3. Load defaults from `/kaggle/input/<slug>/kaggle_bundle/config/runtime_config.json`.
4. Read competition input CSV, run solver, write deterministic `submission.csv`.

No internet dependency is required.

## Local Kaggle Dry-Run Validation

Run:

```bash
scripts/smoke_kaggle_bundle.sh
```

This script validates:
- bundle was built and contains required files
- wheel installs from bundle path
- Kaggle-like path (`/kaggle/input/...`) flow works in a local simulation
- a deterministic `submission.csv` is produced from synthetic `test.csv`

## Offline Assumptions

- Notebook runtime has no internet access.
- Code is installed only from attached dataset wheel.
- Core solve behavior is package-driven (`mgvs.solve.runner.solve`).
- Notebook is only for I/O and submission assembly.

## Common Failure Points

- Wrong dataset slug:
  - Symptom: no wheel found in `/kaggle/input/.../wheels`.
  - Fix: set `MGVS_DATASET_SLUG` correctly.

- Missing wheel in uploaded dataset:
  - Symptom: install step fails.
  - Fix: rebuild with `scripts/build_kaggle_bundle.sh` and re-upload.

- Wrong competition columns:
  - Symptom: empty predictions or key errors.
  - Fix: set `ID_COL`, `PROBLEM_COL`, and `ANSWER_COL` to match competition schema.

- Invalid runtime config values:
  - Symptom: solve config creation fails or strange runtime behavior.
  - Fix: validate numeric fields in `runtime_config.json`.

- Using vLLM in offline notebook without endpoint:
  - Symptom: generation calls fail/time out.
  - Fix: keep `backend` as `stub` unless a reachable local endpoint is available.
