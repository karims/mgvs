# MGVS Kaggle Offline Runner

This folder contains assets for running `mgvs` in a Kaggle notebook without internet access.

## 1) Build and Upload Bundle

From your local repo:

```bash
scripts/build_kaggle_bundle.sh
```

Upload `dist/mgvs_kaggle_bundle.zip` as a Kaggle Dataset.

Recommended dataset structure after upload:
- `kaggle_bundle/wheels/*.whl`
- `kaggle_bundle/examples/handpicked_problems.json`
- `kaggle_bundle/kaggle/submission_notebook.ipynb`

## 2) Notebook Setup (Offline)

Inside a Kaggle notebook, attach the uploaded dataset and install from the local wheel path.

```python
import glob
import subprocess
import sys

WHEEL_PATH = sorted(glob.glob("/kaggle/input/<your-dataset-slug>/kaggle_bundle/wheels/*.whl"))[0]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", WHEEL_PATH])
```

No GitHub clone and no internet dependency is required.

## 3) Running Solver on Competition Input

Expected pattern:
1. Read competition CSV from `/kaggle/input/<competition>/test.csv`.
2. For each row, call `mgvs.solve.runner.solve(problem_text, ...)`.
3. Convert solver result to a prediction string/number.
4. Write `submission.csv` with competition-required columns.

The starter notebook scaffold in `submission_notebook.ipynb` already includes this flow.

## 4) Runtime Config via Environment Variables

For optional vLLM/OpenAI-compatible backend settings:
- `MGVS_VLLM_BASE_URL`
- `MGVS_VLLM_API_KEY`
- `MGVS_VLLM_MODEL_NAME`
- `MGVS_VLLM_TEMPERATURE`
- `MGVS_VLLM_MAX_TOKENS`
- `MGVS_VLLM_TIMEOUT`

Offline-first default is the deterministic stub backend.

## 5) Offline-First Assumptions

- The notebook does not fetch code from the internet.
- The package is installed from a prebuilt wheel in `/kaggle/input/...`.
- Core logic remains in the Python package; notebook only handles I/O and submission assembly.
