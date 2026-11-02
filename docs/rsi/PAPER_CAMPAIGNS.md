# Multi-paper reproduction and capability accumulation

`paper_campaign.v1` is a persistent Scientist driver. The expert team executes an
admitted computational scope for each paper, reviews its evidence, registers a
content-addressed workflow, then retrieves it in fresh sessions and performs new
computation. This extends the existing autonomous model-development drivers; a
registered reproduction recipe is not an invented model or evidence of recursive
performance improvement.

## Available scopes

The executable catalog is `src/agent/rsi/campaigns/paper_campaign.py`. It currently covers
eight publications and fourteen admitted allocations:

| Publication / domain | Executable scope | Explicit omissions |
| --- | --- | --- |
| PHL7 / PET hydrolase | All 24 published OPES output reanalyses; separately, CPU or CUDA WT HREX validation | Production-length new dynamics and experimental degradation |
| htFuncLib / GFP design | Author EpiNNet training, independent prediction replay and library reconstruction | New Rosetta jobs and fluorescence screening |
| ALDE / directed evolution | GB1 and TrpB, three acquisition strategies, either 5 or 70 seeds per strategy | Other model families and wet-lab ParPgb evolution |
| McbA / enzyme engineering | All 25 released notebook tasks and reference discrepancies | Upstream zero-shot generation, cell-free expression and conversion assays |
| TurboPETase | Three released checkpoints, CPU or CUDA | Original training and PET degradation experiments |
| RFantibody / VHH design | HA or RSV site 3, 64 backbones and 256 sequence records per target | Other targets, full published sampling and binding assays |
| CataPro / enzyme discovery | Full admitted 50-fit training allocation, or 30-checkpoint / 95,507-row cohort replay | Wet-lab enzyme discovery and mutation assays; training is excluded from checkpoint replay |
| NucB / nuclease engineering | Four raw-count generations, 16 tables, 55,760-sequence landscape, 44 library hit-rate rows and 28 initial-rate fits | CNN/VAE training, sequence design, diversity analysis, unseeded WT-denominator figure and wet-lab replication |

The ongoing five-target RFantibody benchmark is a separate frozen experiment;
this catalog does not alter its allocation or imply that it has completed.

Presets: `cpu_papers` (PHL7 + GFP), `enzyme_learning` (partial ALDE + McbA +
CataPro training), `seven_papers`, `nuclease` (NucB), and `eight_papers`. Existing
presets retain their original allocation. The CLI also composes any distinct admitted
scopes with repeated `--paper-case`. The web UI exposes the five presets and
shows per-paper criteria, included steps and excluded steps.

## Run

From the research repository, use a new output directory beneath
`research_runs/rsi`. Supply an explicit harness/model profile and credential
source; credentials are never stored in recipes.

```sh
PYTHONPATH=src python scripts/rsi/run_paper_campaign.py create \
  --output research_runs/rsi/my_paper_campaign \
  --profile /absolute/path/to/profile.json \
  --case cpu_papers \
  --objective 'Reproduce the declared PHL7 and GFP scopes and verify reuse' \
  --max-cycles 8 --model-calls 192 --tool-calls 96 --wall-seconds 21600

PYTHONPATH=src python scripts/rsi/run_paper_campaign.py run \
  --output research_runs/rsi/my_paper_campaign \
  --credential-toml /absolute/path/to/provider-config.toml
```

Use `status`, `pause` and `resume` with the same output. Scientific results and
code snapshots are immutable evidence: create a new run for changed code or
scope, rather than rewriting the old run. The two-paper GLM5.2 validation is
isolated under `research_runs/rsi/20261018_paper_campaign_glm52_v1`.

## Completion and failure semantics

- All requested scopes must pass reproduction, registration and fresh-session
  reuse. One successful paper cannot complete the campaign.
- Scheduling chooses the unfinished scope with the fewest attempts. Exceptions
  retain the reserved budget; failed papers cannot starve the others. A restart
  retains the chosen paper and episode history.
- Each recipe binds the paper DOI, allocation, included/excluded steps, source
  hashes, domain verification, authentic tool receipt and distinct sessions.
- Partial computational reproduction is a legitimate, explicit outcome.
  `full_paper_reproduced` remains false. New executions of a deterministic
  workflow are not independent biological replicates.
- A changed source report, receipt, allocation, recipe or paper scope fails
  validation. The system does not silently relax a paper after failure.
- Capability versions retain their original goal evidence. The optional library
  shares verified recipes between goals within an owner namespace; the web
  driver restricts this namespace to the same conversation.

## Adding another paper

1. Resolve the original paper, author code, license and data; declare the
   computational steps and exclusions. Prefer a meaningful complete stage to an
   undocumented reduction of sample size.
2. If an existing plugin supports the required workflow, add a distinct
   allocation in `reproduction_goal_contract.py` and a scope in `paper_campaign.py`.
3. For a new workflow, implement its pinned runner, strict parameter schema and
   native artifact verifier in the existing case-plugin / case-evidence broker.
   Validate source/data provenance, held-out boundaries and numerical outputs.
   Add tests for incomplete, forged and changed evidence before admission.
4. Run the new scope through the campaign, including fresh-session reuse.
   Report actual completed stages and discrepancies; catalog inclusion alone is
   not experimental success.
5. For a claim of new model development, use `generated_model_evolution.v1` and
   independently evaluate changed model code, selection criteria, fresh reuse
   and held-out/cross-protein performance. Workflow accumulation alone does not
   establish that claim.

The system executes admitted paper workflows. Automatic conversion of any
arbitrary PDF into a validated new plugin is not implemented by this extension.

## References

The catalog stores DOI links for every paper. CataPro metadata was checked
against [the original article](https://www.nature.com/articles/s41467-025-58038-4),
and htFuncLib against [the original article](https://www.nature.com/articles/s41467-023-38099-z)
and [author repository](https://github.com/Fleishman-Lab/htFuncLib).


## Verified allocations and ongoing expansion (18 October 2026)

The PHL7 + GFP campaign completed all four initial/reuse episodes and passed
independent terminal audit: 54 physical model calls, 29 tools and 2,639.1 seconds.
It predates immutable scope admission. A separate final-code GFP allocation
validated that admission contract and completed training, registration and new
training in fresh sessions: 23 physical calls, 12 tools and 492.1 seconds.
Both preserve partial computational scope and exclude wet-lab replication.

The new `20261018_enzyme_paper_campaign_glm52_v1` allocation uses ALDE's declared
30-trajectory partial scope, all 25 McbA notebook tasks, and three TurboPETase
checkpoints on CPU. It requires fresh-session recomputation for each paper.
Consult its `result.json` and goal checkpoints for actual completion; startup
does not count as a successful reproduction. All source snapshots were frozen
after 989 regression tests passed.

Independent audit records are under `20261018_paper_campaign_audit_v1` and
`20261018_paper_campaign_admission_audit_v1`. The separate improvement-policy
comparison is described in `IMPROVER_REPLICATES.md`.


## Cross-goal capability library

The web driver automatically uses the current conversation's `.paper_capabilities`
directory. A CLI operator can opt in during `create` with
`--capability-library research_runs/rsi/my_library`. All participating goal
directories must be beneath that library's parent; choose a dedicated owner
directory when multiple users share a host. The web API does not accept a user
supplied library path or search other conversations.

The library publishes a recipe only after its originating campaign completes
and native checks revalidate the initial computation and fresh-session
recomputation. Entries bind the exact DOI/scope/allocation, immutable recipe,
original goal and protocol hashes. Historical source snapshots are checked as
data; importing a recipe never executes archived implementation code.

When a new goal starts, exact matching entries are revalidated and pinned in
its immutable admission contract. Its local registry records the original
entry identity. These imported checks establish historical reproduction and
registration only: the new goal remains unfinished until fresh sessions request
new computation and review new artifacts. The UI explicitly distinguishes
imported workflows awaiting computation from verified reuse. Completing an
imported workflow preserves the original capability ID, rather than counting
each use as a new discovery.

Changed provenance, altered recipes, traversal/symlink paths, changed scope or
unavailable originating evidence reject import or reuse. Failed goals cannot
publish. Existing campaigns without a library declaration retain their prior
behavior. Publication can be retried without model calls using the `publish`
CLI action on a completed library-enabled campaign.

This is accumulation of validated scientific workflows. It does not establish
new model architecture generation, biological improvement or autonomous
admission of arbitrary papers; those require their separate evaluations.


## Cross-goal GLM validation (18 October 2026)

`20261018_cross_goal_recipe_glm52_v1` imported the already audited GFP EpiNNet
recipe from `20261018_paper_campaign_admission_glm52_v2` through the new owner
library. Seeding that historical entry was a controller operation. The new
Scientist goal began unfinished, retrieved the imported recipe, requested new
author-workflow training, reviewed new evidence and completed one reuse episode:
13 physical model calls and 7 tool calls with GLM5.2. An initially malformed
review was rejected and corrected within the same allocation; that record is
retained. Goal execution accounted for 406.7 seconds; this excludes setup and
post-goal library publication/audit.

The read-only audit in `20261018_cross_goal_recipe_audit_v1/audit.json` replays
native numerical checks and verifies all 663 frozen implementation files,
original admission, unchanged donor and recipient evidence, new computation in
a distinct directory, and two new sessions disjoint from all four donor
sessions. Reuse preserves the original library ID and adds zero new capability
entries. No external model calls are made by the audit.

The same revision passed 998 regression checks across RSI, harness and dual-mode
skill integration, TypeScript checking and the production frontend build.
Synthetic desktop/mobile UI checks verify the imported/pending label and lack
of horizontal overflow; they are interface checks, not scientific experiments.
Validation records are in `20261018_cross_goal_library_validation_v1`.

This establishes cross-goal acquisition and recomputation of a validated
workflow. It does not establish invention of EpiNNet, higher protein fitness,
new biological replication, arbitrary-paper admission or recursive performance
improvement.


## Bounded web preparation

Historical receipt, code and protocol hashes are bound during goal creation.
Numerical artifact replay runs in the detached worker before model execution,
completion assessment and publication. This preserves the native verification
gate while avoiding the web service's 60-second creation timeout when several
large historical workflows are imported.

The final revision passed 999 regression checks, including a test that allows
metadata-only admission but forces worker evaluation to reject unavailable
native artifacts. Real GFP history preparation took 4.306
seconds in `20261018_cross_goal_recipe_glm52_v2`. That final-code allocation
completed one cross-goal reuse episode with 14 physical model calls, 7 tool calls
and 418.4 seconds of goal accounting. Its separate read-only native audit passed:
663 frozen files remained unchanged, two new sessions were disjoint from the
donor's four sessions, and the existing capability ID was preserved. The earlier
v1 success remains evidence for its own implementation. Both runs reconstruct
the same published GFP workflow; they do not establish a causal RSI benefit.


## NucB raw-count extension (2026-10-18)

`nucb.raw_analysis.v1` executes all four author generations from the pinned raw
counts, builds the landscape, and recomputes library hit rates and purified-protein
initial velocities. It accepts no subset, alternate-source or verify-existing
parameter. Registering or reusing it requires native artifact checks; reuse starts
another computation rather than accepting the earlier report.

The author commit is `702284847c4558567d8288169a86dc147cfaab32` from
[google-deepmind/nuclease_design](https://github.com/google-deepmind/nuclease_design).
Source and data manifests are hash-pinned. The native verifier compares all 17
computed tables (16 generation tables plus the landscape) against the published
reference, then recomputes every analysis artifact from the run's reconstructed
inputs. This includes NumPy OLS verification of all 28 author SciPy line fits.
Failed and partial outputs remain available for diagnosis.

The registered scope is `nucb_raw_analysis`; choose `--case nuclease` to exercise
reproduction, review, registration and fresh-session computation. The existing
`nucb` model-development goal remains a separate activity-classification task.
This extension does not regenerate the author CNN or demonstrate that the agent
invented the statistical method. The known G1 diversity discrepancy is outside
this scope, not corrected or erased.

The frozen GLM5.2 experiment is recorded under
`research_runs/rsi/20261018_nucb_paper_campaign_glm52_v1` once launched. Consult its
`result.json` and independent terminal audit before claiming success. Catalog
admission and passing software tests alone are not completed paper reproduction.

## NucB raw-to-analysis ordering repair

The first raw-data campaign (`20261018_nucb_paper_campaign_glm52_v1`) remains
unsuccessful. All four generations and 17 processed/landscape comparisons
passed, but the subsequent hit-rate bootstrap differed from the notebook.
Its second attempt was interrupted by the engineering controller after the
deterministic cause was confirmed; its failed/interrupted episodes and reserved
cost remain in the record. This was an implementation intervention, not an
autonomous agent repair.

The author bootstrap samples row positions. Raw reconstruction changed the
positions of 58,086 of 58,141 generation-level records, despite numerically
equivalent tables. The analysis now matches the published generation/sequence
identity order before resampling, retaining every reconstructed measurement.
Missing, replaced or duplicate identities are rejected; reference measurements
are never substituted. The input hashes and identity-order receipt are bound
to the analysis protocol and checked again by independent replay.

Production-code analysis and independent replay from copies of the first
run's reconstructed tables matched all 44 published hit-rate rows and all 28
initial-rate fits. This diagnostic is recorded under
`20261018_nucb_order_validation_v1`; it is not fresh raw-count computation and
does not change the failed campaign. A separate frozen
`20261018_nucb_paper_campaign_glm52_v2` is prepared to rerun the full raw workflow
and fresh-session reuse. Its actual completion must be established from its
terminal result and separate audit.
