# Verified training-attempt memory

New standard generated-model and sequence-evolution goals enable
`verified_training_attempts.v1` by default. An unsuccessful development episode
can leave useful attempted code and training diagnostics for the next expert
discussion. This is separate from accepted-model experience and independent
acceptance feedback.

## Execution contract

- The controller matches each numbered attempted source and diagnostic to an
  actual paired `evaluate_model_code` request/result, even if results arrive out
  of order. Scope, training inputs, capability and execution fingerprints must
  agree.
- At most two attempts from each of the last two development episodes enter the
  prompt. A source over 12,000 characters is explicitly omitted; its fingerprint
  and diagnostic remain. It is never silently truncated into runnable code.
- The projection includes training failures or allowlisted training metrics.
  It reads no independent acceptance, future-score or fitted-state file.
- An invalid tool trace supplies no attempted source. Changed previously bound
  evidence fails validation. Uncheckpointed attempts are not repair authority.
- Reuse episodes receive no development-attempt projection. Failed attempts
  continue to consume their original resources; this policy creates no extra
  attempts or budget.
- The host writes `attempt_memory.json`, `attempt_prompt.json` and, for a
  development episode, `attempt_evidence.json`. Independent audit reconstructs
  the projection and verifies checkpoint and execution-prompt bindings.

## Configuration

`scripts/rsi/run_science_goal.py create` and
`scripts/rsi/run_science_evolution.py create` accept
`--attempt-memory verified_training_attempts.v1` or `--attempt-memory none`.
The resolved policy is saved in the immutable goal protocol. Existing protocols
without this field mean `none`; do not modify an active/frozen run to enable it.
Create a separate goal for a changed policy.

The standard web generated-model drivers use the same preparation entry point.
Classifier and REAP experiment preparations keep their original defaults.
Source-withheld `metrics_only` and `independent_restart` interventions cannot
enable this source-bearing memory.

## Validation and experimental scope

`research_runs/rsi/20261018_attempt_memory_validation_v1/regression_summary.json`
records 1,067 passed tests, zero failures/errors/skips. The two recorded test
invocations cover the complete previous 1,054-test suite and 13 new tests.
The first attempted regression command used a nonexistent file path and ran no
tests; that log is retained separately and is not included in the pass count.

The integration test runs two simulated host episodes and verifies that a
failed source and its error enter the second prompt without changing accounting.
Tampering, source withholding, unavailable traces, history bounds and exclusion
of independent data are also covered. These tests do not establish that a real
language model repaired code successfully.

The separate frozen GLM-5.2 follow-up
`20261018_training_memory_models_glm52_v1` allocates ParPgb and McbA to two native
expert adapters. Each has at most four episodes, 96 physical model calls,
48 tools and 3,600 seconds; each episode has ceilings of 24 calls, 12 tools and
900 seconds. Every allocation and failure remains in the record. It measures
operational compatibility and observed use of prior attempts. The earlier
matrix remains unchanged; a before/after comparison is not a causal estimate
of memory benefit. Final outcomes require independent terminal replay.

For a causal memory claim, freeze a separate matched with/without-memory design
before inference, with repeated seeds and equal task data, native profiles and
total budgets. Report both failure recovery and independently evaluated protein
performance. Successful debugging alone is not recursive scientific progress.
