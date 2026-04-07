# mgvs

LLM-guided structured search over a canonical reasoning state with deterministic multi-level verification.

`mgvs` is a research/competition codebase for Markov-Guided Verified Search: a solver architecture that keeps a self-contained reasoning state, asks an LLM for bounded next actions instead of full proofs, verifies those actions, and searches over the resulting state space with beam selection and deduplication.

## What is in the repo

- `src/mgvs/state`: canonical reasoning-state models and trace records
- `src/mgvs/actions`: structured action schema, application, and simple scoring
- `src/mgvs/verify`: local, consistency, and global verification layers
- `src/mgvs/search`: controller loop, beam pruning, deduplication, and termination
- `src/mgvs/llm`: PT/PCT/LSS prompt builders, parsers, stub backend, and vLLM client
- `src/mgvs/solve`: end-to-end runner, answering, solve policy, caching, and runtime controls
- `src/mgvs/eval`: benchmark evaluation, metrics, review taxonomy, and result comparison
- `src/mgvs/domains`: lightweight algebra, polynomial, and number-theory plugins
- `kaggle/`: offline Kaggle bundle assets and thin notebook scaffold

## Pipeline

The solver is organized around three LLM stages:

- `PT` (problem translation): extract objects, equations, constraints, witnesses, and goals
- `PCT` (concept/tactic proposal): propose strategy tags and high-level next goals
- `LSS` (local step synthesis): emit bounded candidate actions for the search controller

Those actions are then verified, applied to cloned reasoning states, canonicalized, deduplicated, and beam-pruned until the search terminates.

## Install

```bash
python3 -m pip install -e .
```

The package exposes a console script:

```bash
mgvs
```

## Quickstart

Run the local stubbed solver on a single problem:

```bash
mgvs solve --problem "Solve x + 1 = 2"
```

Run a small benchmark CSV:

```bash
mgvs eval \
  --input examples/reference_numeric.csv \
  --problem-col problem \
  --answer-col answer \
  --id-col id
```

Review exported benchmark traces:

```bash
mgvs review \
  --results out/results.jsonl \
  --traces out/traces.jsonl \
  --output out/review.json
```

Compare two benchmark runs:

```bash
mgvs compare \
  --baseline out/baseline.jsonl \
  --candidate out/candidate.jsonl \
  --output out/compare.json
```

## CLI commands

- `mgvs solve`: run the end-to-end solver for one problem
- `mgvs eval`: run a benchmark from a local CSV and compute aggregate metrics
- `mgvs review`: classify benchmark outcomes into a compact failure taxonomy
- `mgvs compare`: compare baseline vs candidate result files for iterative tuning

Both `solve` and `eval` support `--backend stub` and `--backend vllm`.

## Backends

### Stub backend

The default backend is deterministic and intended for local architecture testing:

```bash
mgvs solve --backend stub --problem "Solve x + 1 = 2"
```

### vLLM / OpenAI-compatible backend

The vLLM client expects an OpenAI-compatible `/chat/completions` endpoint and reads configuration from environment variables:

- `MGVS_VLLM_BASE_URL`
- `MGVS_VLLM_API_KEY`
- `MGVS_VLLM_MODEL_NAME`
- `MGVS_VLLM_TEMPERATURE`
- `MGVS_VLLM_MAX_TOKENS`
- `MGVS_VLLM_TIMEOUT`

Example:

```bash
export MGVS_VLLM_BASE_URL="http://localhost:8000/v1"
export MGVS_VLLM_MODEL_NAME="your-model"
mgvs solve --backend vllm --problem "Solve x + 1 = 2"
```

## Debugging LLM output

For raw LLM tracing and parser diagnostics:

```bash
export MGVS_DEBUG_LLM=1
export MGVS_DEBUG_PARSER=1
export MGVS_DEBUG_RUNTIME=1
export MGVS_DISABLE_STAGE_CACHE=1
```

These flags enable:

- full response-object logging inside the vLLM client
- raw PT/PCT/LSS content logging before parse failure
- parser-level LSS skip diagnostics
- runner-level backend selection and PT/PCT/LSS cache hit/miss tracing
- cache bypass for fresh calls during debugging

`MGVS_DISABLE_STAGE_CACHE=1` is especially useful when you want to avoid replaying previously cached fallback outputs.

## Evaluation artifacts

`mgvs eval` can export both results and traces:

```bash
mgvs eval \
  --input examples/reference_numeric.csv \
  --problem-col problem \
  --answer-col answer \
  --id-col id \
  --export-results out/results.jsonl \
  --export-traces out/traces.jsonl
```

Those files feed directly into:

- `mgvs review` for failure taxonomy summaries
- `mgvs compare` for baseline-vs-candidate regression checks

## Kaggle workflow

The repo includes an offline-first Kaggle path:

```bash
scripts/build_kaggle_bundle.sh
scripts/smoke_kaggle_bundle.sh
```

This produces a bundle containing:

- the installable wheel
- runtime config
- lightweight examples
- the submission notebook scaffold

See [kaggle/README.md](/Users/ksnaik/StudioProjects/mgvs/kaggle/README.md) for the upload flow, notebook attachment flow, offline assumptions, and common failure points.

## Development

Run the full test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -q
```

There is also a smoke script:

```bash
scripts/smoke_test.sh
```

## Current status

This is still an experimental structured-search system, not a finished competition solver. The architecture is in place: reasoning-state modeling, structured action search, verifier stack, benchmark tooling, review/compare workflow, and Kaggle packaging. The main active work is improving prompt contracts, parser robustness, and backend reliability for real model-driven runs.
