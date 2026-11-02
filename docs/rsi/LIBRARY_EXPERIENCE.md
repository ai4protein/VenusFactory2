# Cross-task Scientist model experience

Scientist goals can now receive frozen implementation experience from the owner
model library. This supplements the existing within-goal evolution: later
generations still inherit their own accepted predecessors, while a new research
task can adapt implementations verified on earlier tasks. Harness and LLM
bindings remain independent of this mechanism.

## Information conditions

An explicit selection declares one of three conditions:

| Condition | Information available to the receiving Scientist |
| --- | --- |
| `none` | An empty experience packet |
| `summary` | Donor scope and a deterministic structural description of imports, routines and numerical library calls |
| `source` | The same description plus the accepted fit/predict implementation |

The structural summary is derived from Python syntax; it is not an inferred
mechanistic explanation. It excludes literal values. All three conditions use
the same `read_model_library_experience` tool and require an actual successful
read in a completed development episode. Context sizes differ and should be
reported with model usage; equal call limits are not equal token consumption.

The model-facing packet does not include fitted model state, observation tables,
acceptance metrics, origin paths or provider configuration. Source code remains
an implementation precedent: the recipient must fit its own candidate, pass its
own numerical acceptance test, register it and demonstrate fresh-session reuse.
No transfer operation creates or counts a new model.

The separate `read_model_experience` tool retains the current goal's local
generation history. Thus an external `none` or `summary` condition does not
silently remove or expose source from the within-goal lineage. Library experience
is absent from the fresh-session inference phase.

## Admission and verification

Before the first model request, the host replays each selected library entry's
publication proof and training validation. It freezes the selected IDs, canonical
origins, source identities, donor cohort hashes and receiving task admission.
The goal definition and protocol both bind the manifest and public projection.
Reopening and publication recheck these bindings and their provenance.

Donor dependencies are traversed iteratively. Each ancestor is verified before
the descendant that uses its experience.
An operation-local cache avoids replaying shared ancestors repeatedly during one
audit and is discarded afterwards. Cycles, changed origins, invalid numerical
evidence and retired canonical models fail verification. There is no persistent
cache that could hide a later origin change.

For sequence-model transfer, all cohorts of direct **and ancestral** donors must be exactly sequence-disjoint
from the receiving validation and future cohorts. This includes source goals'
future cohorts, whether or not they have been scored. It prevents direct and
indirect reuse of those exact evaluation sequences through the experience chain.
It does not establish homology independence, semantic novelty or biological
generalization. Related-family studies still need scientifically justified
grouping and independent evaluation.

The projection admits at most five selected donors and 200,000 encoded bytes.
These limits bound one episode's context, not the size of the shared model
collection. Source summaries and source packets use the same donor selection.

## Individual goals

Create a selection JSON with the existing owner-library path and verified entry
IDs returned by `scientist_model_library.py list --verify`:

```json
{
  "condition": "source",
  "library_root": "/home/tanyang/workspace/VenusFactory2/research_runs/rsi/scientist_model_library_v1",
  "entry_ids": ["REPLACE_WITH_A_VERIFIED_64_CHARACTER_ENTRY_ID"]
}
```

Both `run_science_goal.py create` and `run_science_evolution.py create` accept
`--experience-selection /absolute/path/selection.json` alongside
`--task-definition`. The goal and donor library must share an owner directory.
Credentials are unnecessary for creation. Keep the existing run/resume
credential and harness configuration explicit; creating a goal never makes a
model request.

For example, with a public profile and an admitted task:

```sh
PYTHONPATH=src python scripts/rsi/run_science_evolution.py create \
  --output research_runs/rsi/NEW_transfer_goal \
  --profile /absolute/path/profile.json \
  --task-definition /absolute/path/task.json \
  --experience-selection /absolute/path/selection.json \
  --objective 'Develop and independently validate a model for this protein task' \
  --generations 2 --max-cycles 8 --model-calls 192 --tool-calls 96 \
  --wall-seconds 7200
```

Use a frozen research repository for experiments that need to survive subsequent
code changes. Source changes are not silently applied to an already admitted goal.

## Continuous campaigns

An allocation may add an `experience` policy to the existing plan fields:

```json
{
  "condition": "source",
  "selection": "prior_campaign",
  "max_entries": 3
}
```

The scheduler considers verified publications from earlier allocations, newest
allocation first, with deterministic entry ordering. It takes the first eligible
donors without ranking on target performance. Interface incompatibility and
cohort overlap produce recorded exclusions. Invalid or changed evidence stops
admission rather than being skipped. Later allocations and unverified entries
cannot become donors. A first allocation can therefore have an empty packet,
while subsequent tasks accumulate the campaign's actual verified implementations.

For a matched transfer experiment, use an explicit policy containing
`condition`, `selection: "explicit"` and `entry_ids`, and hold the donor IDs,
task, model profile and resource limits fixed across conditions. The owner
library comes from the campaign definition; allocations cannot redirect it.
The chosen request and eligibility review are written under `admitted_inputs`
before goal creation and are not reselected on resume.

## Validation and interpretation

Tests cover direct and ancestral cohort overlap, source changes, frozen
admission, required tool reads, independent within-goal/library contexts, and a
synthetic 1,005-node dependency chain with cycle rejection. The latter is a
scalability test and contributes zero scientific models.

The final regression on 23 October 2026 passed 205 tests across experience,
campaign scheduling, task admission, model publication, numerical sequence
models, goal routing and web goal integration. It does not establish visual UI
quality or external-model scientific performance.

A separate local transport probe replays the three existing historical model
entries and delivers all three information conditions through the actual MCP
server to the published avGFP task with 600 training observations. It checks the
frozen capability and returned content and verifies that the library JSON tree
is unchanged. This uses no external model service and does not establish that
transfer improves model performance.

Evidence is retained under
`research_runs/rsi/20261023_library_experience_changes_v1/`. The first source
integration also found and fixed a missing `generated_sequence.v1` entry in the
official capability bundle builder; its regression now uses the real builder,
MCP development and fresh-session reuse rather than assembling a test bundle
manually. GLM-backed transfer and recursive-improvement efficacy remain to be
tested after model access recovers.


## Prepared GFP transfer pilot

A fixed 24-allocation pilot is prepared at
`research_runs/rsi/20261023_library_transfer_glm52_setup_v1/`:
four GFP families, three external experience conditions and native Kimi/DeepSeek
Scientist harnesses, each with the existing GLM-5.2 binding and two generations.
All conditions use the same three historical donors, task partitions and call
limits. The study protocol fixes feasibility and future-data outcomes before
execution. One run per cell is a pilot; it does not support a powered efficacy
claim or a Scientist-versus-Agent comparison.

Local preflight verified 738 copied source files, both runtime bindings and all
12 distinct task/experience combinations. The 24 allocations remain queued,
with zero external model calls, new autonomous models or future scores. Runtime
binaries are referenced from their existing installations and hashed during
preflight; they are not claimed immutable. GLM access remains unresolved after
the latest authorized request returned HTTP 403.

Each allocation allows up to 192 model calls, 96 tool calls and 7,200 seconds.
These are limits, not consumed resources. This pilot can yield at most 48 lineage
outputs before independent acceptance and deduplication; it cannot by itself
fulfil the broader 1,001-model objective. The detailed protocol is
`study_protocol.json`; `preflight_report_v2.json` records the successful check.
The initial check script omitted its output parent directory; that failed check
and its log are retained separately, and the frozen production source was not
modified by its repair.


## Contextual model implementations

Numeric protein-condition tasks now support explicit study-disjoint source transfer through `overlap_policy: "context_study_disjoint.v1"`. The host binds reviewed study/group evidence, canonicalizes DOI identities, excludes any direct or ancestral donor from the receiving study, and requires compatible ordered numeric inputs and output semantics. Each recipient refits, passes its own numerical acceptance and completes fresh-session reuse. No fitted model state is transferred. Historical selections retain their existing rules.

The related regression passed 218 tests. The source-refit/baseline-feedback interaction was subsequently verified in 39 tests across two overlapping suites. All six real plant source versions reverified and were correctly excluded as same-study donors. Configuration, evidence and scientific interpretation: Contextual source transfer (local research archive).
