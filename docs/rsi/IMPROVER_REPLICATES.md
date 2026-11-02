# Independent improvement-policy revisions

`scripts/rsi/run_improver_replicates.py` prepares three independent policy-revision
attempts and 54 offspring goals. This experiment tests whether revised model-code
proposal instructions help subsequent modelling. It does not promote a policy
automatically, change the evaluator, or establish recursive improvement by merely
submitting new text.

## Frozen design

- Three revision attempts use the identical projection of six completed public
  REAP development histories (P450 BM3 FL62 and SaSrtA, partitions 4, 5 and 6).
- Each revision has one submission opportunity, at most 8 model calls, 4 tools
  and 420 seconds. All three revisions are attempted before offspring execution.
  An unsuccessful revision stops the batch before offspring; it is not replaced.
- Each revision supplies 18 allocations: two proteins, partitions 7, 8 and 9,
  and three conditions: fixed policy, revised policy, and fixed policy with the
  additional batch-level revision allowance. Offspring begin without a fitted
  predictor and receive identical historical development information.
- Each ordinary offspring goal has 6 episodes, 96 model calls, 48 tools and
  2,700 seconds. The extra-cost control distributes the revision allowance over
  its six goals, charging the revision once per batch rather than once per goal.
- The three groups are interleaved and their ordering rotates across allocation
  slots. Runtime concurrency is one. All requested allocations remain in the
  denominator, including budget exhaustion and execution failures.
- All 54 allocations must terminate and pass independent development/reuse
  replay before any new future metric is calculated. A global barrier binds
  goal, protocol and result hashes to those audits. A failed audit stops scoring.

The protocol uses shared retrospective proteins and partitions. Three API
revision attempts are not three independent biological replicates or successive
generations of an evolving policy. Comparison of resource allowances is distinct
from comparison of actual token use or financial cost. A provider alias is not
an immutable model checkpoint. Positive or negative conclusions require the
completed comparison and its audit; successful revision submission is not enough.

## Historical evidence compatibility

The development-history importer verifies the original policy, assessment,
publication state and immutable model package. If the active plugin has changed,
it additionally verifies the archived implementation and its original plugin
fingerprint. It never instantiates the new evaluator for an old acceptance,
executes archived source, rewrites that acceptance, or imports future metrics.
Independent audit separately reconstructs the projected evidence and numerical
development results.

The first preparation attempt on 18 October 2026 stopped at the plugin version
check before any external model call. That original directory and log remain
under `research_runs/rsi/20261018_improver_replicates_glm52_v1`. After adding the
archive-aware reader, the second offline preparation exposed the old four-file
fingerprint schema. That failure remains in the v2 directory. Both the legacy
four-file and current five-file schemas are now explicitly checked without
executing archived source; unknown schemas are rejected. All 989 regression
tests passed. A new frozen setup was launched
under `research_runs/rsi/20261018_improver_replicates_glm52_v3`; its actual
`supervision.json` and batch `report.json` determine live status.

## Execution and recovery

Run `prepare` in an isolated source snapshot. Give `--output`, an `--audit-root`
outside that snapshot, explicit `--profile` and `--revision-profile`, source
`--admission`, `--features`, `--baseline`, and repeated `--history-goal` arguments.
Preparation and all three independent preflights require no external model calls.

`run --output ... --credential-toml ...` accepts only a prepared, unattempted
batch. It obtains an exclusive lease, verifies frozen code and plans, and records
each worker's PID, start ticks and boot ID. It retains unexpected failures.
Do not restart an attempted batch or alter its protocol to complete it. Inspect
its recorded process identity and evidence; use a separately declared allocation
for a changed implementation or research protocol.

Secrets are resolved only at execution time and are not stored in policy text,
recipes, model artifacts or experiment reports. The public development-history
paths above are the only histories used by this experiment.
