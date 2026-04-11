# MGVS State/Pipeline Map

## Main orchestration entrypoint

- Main solver entrypoint: `src/mgvs/solve/runner.py`
  - `solve(raw_problem, *, config=None, client=None) -> SolveResult`
  - Main per-attempt pipeline: `_run_attempt(...)`
- CLI entrypoint: `src/mgvs/cli/main.py`
  - `main(...)`
  - `mgvs solve --problem "..."` calls `solve(...)`

## Current stage order

Current stage order in `src/mgvs/solve/runner.py` is:

1. Initial state creation
   - `create_initial_state(...)` in `src/mgvs/state/models.py`
2. PT stage
   - `_run_pt_with_cache(...)`
   - prompt: `build_pt_prompt(...)`
   - parse: `parse_pt_output(...)`
   - merge: `apply_pt_update(...)`
3. PCT stage
   - `_run_pct_with_cache(...)`
   - prompt: `build_pct_prompt(...)`
   - parse: `parse_pct_output(...)`
   - merge: `apply_pct_update(...)`
4. PCT answer-candidate handoff
   - `_maybe_accept_pct_answer_candidate(...)`
   - may terminate early with a `SolveResult`
5. Domain plugin annotation
   - `active_domain_plugins(...)`
   - `plugin.annotate_state(...)`
6. LSS/search controller loop
   - `DomainAwareProposer.propose(...)`
   - prompt: `build_lss_prompt(...)`
   - parse: `parse_lss_output(...)`
   - duplicate/no-op filtering and local action scoring happen before controller expansion
   - controller: `run_search(...)`
7. Terminal state / beam answer selection
   - `_select_best_terminal(...)`
   - `select_answer_across_states(...)`
8. Endgame stage
   - `_maybe_run_endgame_stage(...)`
   - readiness gate: `is_endgame_ready(...)` / `_endgame_readiness(...)`
   - prompt: `build_endgame_solve_prompt(...)`
   - parse: `parse_endgame_solve_output(...)`
9. Final direct-answer LLM stage
   - `_maybe_run_final_llm_solve(...)`
   - prompt is built inline in `runner.py`
   - parse: `parse_endgame_solve_output(...)`
10. Optional fallback second attempt
   - in `solve(...)`, unless `debug_single_path` is enabled

## Prompt locations

### PT prompt
- File: `src/mgvs/llm/prompts.py`
- Function: `build_pt_prompt(raw_problem: str, target_type: str) -> str`

### PCT prompt
- File: `src/mgvs/llm/prompts.py`
- Function: `build_pct_prompt(state: ReasoningState, *, max_tactics: int = DEFAULT_PCT_MAX_TACTICS) -> str`

### LSS prompt
- File: `src/mgvs/llm/prompts.py`
- Function: `build_lss_prompt(state: ReasoningState, max_candidates: int) -> str`

### Endgame prompt
- File: `src/mgvs/llm/prompts.py`
- Function: `build_endgame_solve_prompt(...) -> str`

### Final direct-answer prompt
- File: `src/mgvs/solve/runner.py`
- Function: `_maybe_run_final_llm_solve(...)`
- Note: this prompt is not stored in `prompts.py`; it is assembled inline as JSON.

## Where prompt templates are stored

- Structured PT/PCT/LSS/endgame prompt contracts are stored inline in:
  - `src/mgvs/llm/prompts.py`
- There is no separate prompt-template directory such as `src/mgvs/llm/prompts/lss.py` in the current repo.
- One additional final-answer prompt exists inline in:
  - `src/mgvs/solve/runner.py`

## JSON parsing locations

Primary structured parsing lives in:

- File: `src/mgvs/llm/parser.py`
  - `parse_structured_json_object(text: str) -> dict[str, Any]`
  - `_load_json_object(text: str) -> dict[str, Any]`
  - `parse_pt_output(text: str) -> PTUpdate`
  - `parse_pct_output(text: str) -> PCTUpdate`
  - `parse_lss_output(text: str) -> list[CandidateAction]`
  - `parse_endgame_solve_output(text: str) -> EndgameSolveOutput`

LLM client also does pre-parse JSON recovery checks in:

- File: `src/mgvs/llm/vllm_client.py`
  - `_generate_once(...)`
  - uses `parse_structured_json_object(content)` before returning structured stage output

## Where state is merged / updated

### PT merge
- File: `src/mgvs/llm/parser.py`
- Function: `apply_pt_update(state: ReasoningState, update: PTUpdate) -> ReasoningState`
- Updates:
  - `symbolic_objects`
  - `current_equations`
  - `domain_constraints`
  - `global_constraints`
  - `witness_parameters`
  - `open_goals`
  - `derived_facts`

### PCT merge
- File: `src/mgvs/llm/parser.py`
- Function: `apply_pct_update(state: ReasoningState, update: PCTUpdate) -> ReasoningState`
- Updates:
  - `strategy_tags`
  - `open_goals`
  - `current_equations` from `candidate_equations`
  - `derived_facts` gets `answer_candidate = <int>` when present

### LSS/action application
- File: `src/mgvs/actions/apply.py`
- Function: `apply_action(state, action, *, score_config=None) -> list[ReasoningState]`
- Helpers:
  - `_apply_common_updates(...)`
  - `_append_trace(...)`
  - `_apply_score(...)`
  - `_resolve_prune_status(...)`
- Updates may affect:
  - `derived_facts`
  - `domain_constraints` / `global_constraints`
  - `symbolic_objects`
  - `normalized_form`
  - `status`
  - `accepted_steps`
  - `score`
  - `branch_assignments` for branch actions

## Transition validation / pruning

### Search-loop orchestration
- File: `src/mgvs/search/controller.py`
- Function: `run_search(...) -> ControllerResult`
- Flow:
  - propose actions
  - `is_action_valid(...)`
  - `apply_action(...)`
  - canonicalize child states
  - `is_state_valid(...)`
  - deduplicate
  - beam prune
  - terminate when needed

### Verifier stack
- File: `src/mgvs/verify/base.py`
  - `CompositeVerifier`
  - `VerificationResult`
  - `CombinedVerificationResult`
- File: `src/mgvs/solve/runner.py`
  - `TrackingCompositeVerifier`
  - `build_default_verifier()`

### Local validation
- File: `src/mgvs/verify/local.py`
- Class: `V0LocalValidityVerifier`
- Rejects malformed or empty actions and malformed state equations.

### Consistency validation
- File: `src/mgvs/verify/consistency.py`
- Class: `V0StateConsistencyVerifier`
- Handles contradictory status/action combinations, invalid branch/prune patterns, witness conflicts, and simple contradictory constraints.

### Global validation
- File: `src/mgvs/verify/global_.py`
- Class: `V0GlobalCompatibilityVerifier`
- Handles witness/global-constraint compatibility checks.

### Beam pruning / dedup
- File: `src/mgvs/search/beam.py`
  - `deduplicate_states(...)`
  - `select_beam(...)`
- Canonical form currently comes from:
  - `DefaultStateCanonicalizer.canonicalize(...)` in `src/mgvs/solve/runner.py`

### Termination / pruning states
- File: `src/mgvs/search/termination.py`
  - `is_terminal_state(...)`
  - `terminal_states(...)`
  - `should_terminate(...)`
- Terminal statuses:
  - `solved`
  - `contradiction`
  - `dead_end`
  - `parametric`

## Endgame / final answer stages

### Endgame stage
- File: `src/mgvs/solve/runner.py`
- Function: `_maybe_run_endgame_stage(...)`
- Readiness gate:
  - `is_endgame_ready(...)`
  - `_endgame_readiness(...)`
- Prompt:
  - `build_endgame_solve_prompt(...)` in `src/mgvs/llm/prompts.py`
- Parse:
  - `parse_endgame_solve_output(...)` in `src/mgvs/llm/parser.py`

### Final direct-answer stage
- File: `src/mgvs/solve/runner.py`
- Function: `_maybe_run_final_llm_solve(...)`
- Prompt: inline JSON payload in that function
- Parse:
  - `parse_endgame_solve_output(...)`
- Note: this is a separate post-endgame direct-answer call, not just the endgame stage.

### Beam/state answer extraction
- File: `src/mgvs/solve/answering.py`
- Functions/classes:
  - `extract_state_answer(...)`
  - `select_answer_across_states(...)`
  - `StateAnswerCandidate`
  - `BeamAnswerDecision`

## LLM backend implementation paths

### Base client interface
- File: `src/mgvs/llm/base.py`
- Main abstraction: unified LLM client interface used by runner

### vLLM/OpenAI-compatible client
- File: `src/mgvs/llm/vllm_client.py`
- Class: `VLLMClient`
- Stage methods:
  - `generate_pt(...)`
  - `generate_pct(...)`
  - `generate_lss(...)`
  - `generate_endgame(...)`
- Structured-stage response extraction happens in:
  - `_generate_once(...)`
  - `_extract_message_text(...)`
- Structured field precedence for PT/PCT/LSS/endgame currently prefers:
  - `reasoning_content`
  - `reasoning`
  - `content`

### Stub backend
- File: `src/mgvs/llm/stub.py`
- Class: `StubLLMClient`
- Stage methods:
  - `generate_pt(...)`
  - `generate_pct(...)`
  - `generate_lss(...)`
  - `generate_endgame(...)`

## State schema / models

### Canonical reasoning state
- File: `src/mgvs/state/models.py`
- Class: `ReasoningState`
- Key fields:
  - `raw_problem`
  - `target_type`
  - `symbolic_objects`
  - `current_equations`
  - `derived_facts`
  - `domain_constraints`
  - `global_constraints`
  - `witness_parameters`
  - `strategy_tags`
  - `open_goals`
  - `branch_assignments`
  - `status`
  - `score`
  - `normalized_form`
  - `accepted_steps`

### Trace records
- File: `src/mgvs/state/trace.py`
- Class: `TraceStep`

### Action schema
- File: `src/mgvs/actions/models.py`
- Class: `CandidateAction`
- Enum: `ActionType`

### Parser-side structured update models
- File: `src/mgvs/llm/parser.py`
- Dataclasses:
  - `PTUpdate`
  - `PCTUpdate`
  - `EndgameSolveOutput`

### Pydantic usage
- No active Pydantic model layer was found in `src/`.
- The current implementation uses standard-library `dataclasses`, enums, and plain dict/list payloads.

## Short architecture notes

- PT/PCT/LSS/endgame structured prompts are centralized in `src/mgvs/llm/prompts.py`.
- Structured JSON parsing is centralized in `src/mgvs/llm/parser.py`.
- Search orchestration is split between:
  - `src/mgvs/solve/runner.py` for stage sequencing and solve policy
  - `src/mgvs/search/controller.py` for generic propose/verify/apply/beam loop
- Transition validity is enforced by:
  - `TrackingCompositeVerifier` in `runner.py`
  - `CompositeVerifier` in `verify/base.py`
  - local/consistency/global verifiers under `src/mgvs/verify/`
- State mutation happens in three main places:
  - `apply_pt_update(...)`
  - `apply_pct_update(...)`
  - `apply_action(...)`

## Stages that do not exist separately

- There is no separate on-disk prompt-template directory for PT/PCT/LSS such as `src/mgvs/llm/prompts/lss.py`.
- There is no separate standalone “final answer parser”; final direct-answer parsing currently reuses `parse_endgame_solve_output(...)`.
- There is no Pydantic-based schema layer in the current codebase.
