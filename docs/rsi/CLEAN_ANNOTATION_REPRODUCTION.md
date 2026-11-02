# CLEAN enzyme annotation reproduction

This plugin extends the research system from protein engineering to enzyme functional annotation. Its catalog entry declares an executable scope; it is not evidence that a run succeeded. Run it in a dedicated research checkout.

## Source and admitted scope

- Paper: *Enzyme function prediction using contrastive learning*, Science (2023), DOI [10.1126/science.adf2465](https://doi.org/10.1126/science.adf2465).
- Author source: [tttianhao/CLEAN](https://github.com/tttianhao/CLEAN), commit `f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15`. This is an explicit public implementation version, not a claim that it exactly matches the publication-time code.
- Asset manifest SHA256: `b673aaf0eb9af1b70f03061a2589e857bf0f2182a2405f08c4928f9d51acc4c8`.
- Fixed test allocation: new 392 records, Price 149, halogenase 37. All 578 records receive fresh ESM-1b layer-33 mean representations with no truncation. Both released split70 and split100 heads yield 1,156 prediction records.
- The released projected training-member tensors cover 79,172 and 241,025 EC memberships, respectively. Their ordered EC means are reconstructed against all 74,487 and 227,362 released training records.
- CPU reconstruction environment: Python 3.12; torch 2.8.0+cu128 (CUDA disabled), fair-esm 2.0.0, NumPy 1.26.4, pandas 2.2.1, scikit-learn 1.4.2. This differs from the publication environment and is disclosed in each protocol.

Original contrastive training, cross-validation, p-value inference, new enzyme mining and biochemical validation are excluded. Replaying author weights is not autonomous model invention. The source license and pretrained artifacts remain in separately licensed research assets and are not deployed with the service.

## Execution and acceptance

Plugin `clean.annotation.v1` accepts no subset, checkpoint override or cached-output shortcut. Scope `clean_all_cohorts` and suite `enzyme_annotation` use the same full allocation. The prior `seven_papers` and `eight_papers` suites retain their previous allocations.

The runner checks source, datasets, released heads, training-member tensors and both ESM checkpoint hashes before inference. It records per-sequence identities and hashes, fresh embeddings, every EC distance, label choices, the exact protocol, source copies and artifact digests.

The independent validator reconstructs both projection networks, EC centers, distances, the first-gradient max-separation rule and full-union label metrics in NumPy. It checks every record, overlap with released training sequences, and the Price saved-prediction comparison. The audit begins at the bound ESM means: it does not claim a second independent transformer execution. The numerical distance tolerance is 5e-5; output CSV rounding is checked separately.

Author metrics use the cohort's truth label space. Predictions of EC classes absent from that space can be omitted from those metrics. Reports therefore also retain full-union micro precision, recall, F1 and exact match; an extra predicted class counts as a false positive. Weighted F1 alone cannot expose all such errors because classes with zero true support carry zero weight.

The goal follows the existing Scientist protocol: discuss the admitted scope, request a domain job, inspect and review the evidence, register the verified recipe, then retrieve it in a fresh episode and perform a second full computation. Execution, registration and reuse all call the native artifact validator. A successful numerical computation cannot override a failed agent-session audit.

## Validation runs

- Controller baseline: `research_runs/rsi/20261018_clean_full_cohort_baseline_v1`. It has an independent frozen source copy. This is engineering validation, not an autonomous-agent result.
- Integration regression: `research_runs/rsi/20261018_clean_integration_validation_v1`. The first invocation used a nonexistent test-directory alias; its failure log remains. The corrected invocation uses the existing RSI, harness and agent test paths.
- The subsequent GLM campaign must wait for successful baseline computation, current native replay, complete regression and source-integrity checks. Only a terminal audit can establish registration plus fresh-session recomputation.

## Reproduction commands

From the repository root, with the admitted research assets and isolated environment available:

```bash
PYTHONPATH=src research_runs/rsi/envs/clean/bin/python \
  scripts/rsi/run_clean_annotation.py \
  --output research_runs/rsi/clean_unique_new_run
```

For a persistent Scientist goal, create a fresh goal with `scripts/rsi/run_paper_campaign.py create --case enzyme_annotation`, a chosen harness/model profile and at least two episodes. Use the existing credential configuration only when executing the goal; never place tokens in the protocol or logs. The catalog and job recipe are independent of the selected harness and LLM.

## Scientific interpretation

Keep three questions separate: whether a published computation can be reconstructed, whether the agent can execute/register/reuse that workflow, and whether learning from earlier work improves a later task under fair budgets. This plugin addresses the first two. The separate generated-model, failure-memory, cross-protein and improver experiments are needed for the third. Adding CLEAN to the catalog does not establish recursive improvement.

This software uses or is derived from the CLEAN software developed by the Department of Chemical and Biomolecular Engineering at the University of Illinois Urbana-Champaign. Consult the asset license for research-use and publication requirements; no external notification has been sent.

## Completed controller baseline (2026-10-18)

All 578 sequences and 1,156 predictions completed in 1244.39 seconds. The independent audit passed all six model/cohort allocations with maximum distance error 2.53611489e-05, and verified that all 296 frozen source files and the historical report were unchanged. This is a controller-initiated engineering baseline; the GLM Scientist registration/recomputation goal is prepared but has not run because recent provider requests returned HTTP 403.

| Model/cohort | Rows | Full-union micro F1 | Full-union exact match | Author truth-space exact match | Train sequence overlaps |
|---|---:|---:|---:|---:|---:|
| split70.new | 392 | 0.496933 | 0.469388 | 0.522959 | 4 |
| split70.price | 149 | 0.379009 | 0.302013 | 0.429530 | 0 |
| split70.halogenase | 37 | 0.805195 | 0.783784 | 0.810811 | 0 |
| split100.new | 392 | 0.497934 | 0.461735 | 0.510204 | 5 |
| split100.price | 149 | 0.426426 | 0.335570 | 0.476510 | 0 |
| split100.halogenase | 37 | 0.743590 | 0.756757 | 0.783784 | 0 |

On the released Price file, split100 reproduces all 149 label sets; split70 reproduces 92/149. The reference CSV does not declare its training head, so this is an observed agreement rather than proof of how the original file was generated. No test-data tuning was performed. The new cohort has four/five exact sequence overlaps with the released split70/split100 training sets, respectively, despite zero identifier overlaps. These observations remain in every report and must not be presented as an independent unseen-protein evaluation.


## Full split10 training capability (2026-10-22)

`clean.training.v1` adds the distinct `clean_full_training` scope and
`enzyme_annotation_training` paper suite. It uses the existing case broker,
Scientist review, capability registration and fresh-session reuse contracts.
The registered workflow and its verifier do not depend on a particular harness
or LLM. The released-weight annotation scope above retains its own allocation.

Each invocation computes 22,525 new ESM-1b representations: 7,757 split10 training
sequences, 14,190 released augmentations and all 578 evaluation records. It then
runs the full author triplet schedule (2,000 epochs) and SupConH schedule (1,500
epochs), evaluates both trained heads on all three cohorts, and independently
replays all 1,156 predictions. Parameters must be `{}`: smaller subsets, shorter
training, old-output verification and resume options are not admitted by this
capability. Failed outputs and source snapshots remain evidence. A new attempt
requires its own directory and allocation.

The controller, stage wrappers and numerical verifier use the fixed CLEAN
environment. The broker strips inherited credentials and supplies the same
hash seed and CPU thread settings to the outer wrapper and its stages. Numerical
acceptance runs in a separate process in that environment, keeping the host's
different NumPy/pandas versions out of the scientific calculation. Cancellation
terminates the verifier process group; progress polling does not wait for the
numerical replay. Unverified stage/record/epoch progress cannot complete a job.

Acceptance binds the current implementation and archived sources, complete
stage arguments, input protocol, artifacts, fresh input journal and full epoch
traces. The native evaluator then reconstructs numerical predictions and
metrics. Reusing the registered recipe requires a newly executed complete
pipeline; a synthetic workflow test or an existing training report cannot
satisfy that condition. A reproduction job has a three-day wall limit. A
Scientist goal must budget for initial execution, review and a second execution.
The catalog supplies a compatible suggested budget.

The earlier controller baseline completed all stages at
`20261018_clean_split10_training_inputs_v1`,
`20261018_clean_split10_author_training_v1` and
`20261018_clean_split10_evaluation_v1`. Its read-only audit is
`20261022_clean_completed_audit_v1`. The new broker integration is separately
allocated at `20261022_clean_training_plugin_integration_v1`; consult its
`status.json` and broker evidence for actual progress and terminal acceptance.
It makes zero external model calls and cannot establish autonomous registration
or an agent-generated model. The latest regression artifacts are under
`20261022_clean_training_plugin_changes_v1`.

This scope reproduces author training on one released split. It excludes the
publication's five-fold evaluation, independent second transformer/training
replicates, new enzyme assays, novel-model invention and causal RSI efficacy.
The post hoc cohort audit found no exact training-sequence overlap, but does
not establish homology-independent generalization. Full-union scores retain
unseen EC labels and all released evaluation rows; no score-based exclusions
or tuning are permitted by this capability.

For a new direct research execution, use a unique output directory:

```bash
PYTHONHASHSEED=1234 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
research_runs/rsi/envs/clean/bin/python \
  scripts/rsi/run_clean_training_plugin.py \
  --output research_runs/rsi/clean_training_unique_new_run
```

The paper-campaign interface accepts `--case enzyme_annotation_training`, or
the explicit `--paper-case clean_full_training` scope. A real autonomous trial
still needs a working model endpoint and separate terminal review; catalog
availability alone is not success.
