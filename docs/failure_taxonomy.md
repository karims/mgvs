# MGVS Failure Taxonomy (Phase 14 Scaffold)

This document defines the compact taxonomy used by `mgvs review`.

## Categories

- `correct`: prediction matches expected numeric answer.
- `wrong_answer`: answer extracted but incorrect.
- `pt_failure`: likely failure during problem translation setup.
- `pct_failure`: likely failure during concept/tactic stage.
- `lss_failure`: likely failure in local-step synthesis.
- `verifier_too_strict`: verifier rejections dominate and no progress survives.
- `verifier_too_weak`: solved-but-wrong outcomes with little/no verifier friction.
- `branch_explosion`: branching fanout is too high for budget.
- `budget_exhausted`: run ended due to wall-time/budget pressure.
- `answer_extraction_failure`: solved state exists but no usable numeric answer extracted.
- `fallback_used_correct`: fallback path triggered and produced a correct answer.
- `fallback_used_wrong`: fallback path triggered but answer is still incorrect/missing.
- `parametric`: terminal parametric state.
- `contradiction`: terminal contradiction state.
- `dead_end`: terminal dead-end state.

## Notes

- Classification is heuristic and metadata-driven.
- The taxonomy is intended for triage, not formal proof of root cause.
- Update this file when classifier rules change.
