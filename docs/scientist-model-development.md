# Scientist: model development and recursive improvement

Scientist develops models from a research goal, supplied data and a resource
budget through PI → planning → implementation → SC scientific acceptance → revision. A delivery
contains fitted models, reproducible code, measured evaluation and a research
report. Registration follows a separate admission Review.

The PI derives a research direction from the goal and available materials. Plan
steps keep their scientific acceptance criteria while their approach can change
in response to experiments. Completed steps retain their verified evidence.

## Scientific acceptance and revision

SC receives a fresh evidence packet containing the objective, fixed acceptance
criteria, execution receipts, measured validation trajectories and candidate
scope. PI and implementation endorsements are excluded. Each submitted model
requires an explicit usefulness verdict and evidence-backed checks for goal fit,
scientific value, evaluation validity, and scope and limitations. Constant
predictions cannot establish input-specific predictive value; a documented
reference purpose is assessed separately. There is no universal baseline-win rule.

A negative verdict or missing evidence prevents technical Review and registration,
persists an actionable request for PI, and continues the research loop. Accepted
verdicts are bound to the selected experiment, runtime, evaluation and manifest
hashes. Completion rechecks the saved evidence and verdict; changing a model or
receipt cannot inherit the previous acceptance. Independent technical Review uses
its own fresh context and does not receive SC's endorsement. Successful deliveries
include the SC assessment and technical test report. Closed historical protocols
remain unchanged; resumed topics use the new acceptance policy.

## Algorithm freedom

The implementation role can write its own representations, objectives, estimators,
adaptation rules and training code in the session sandbox. A platform PLM trainer
is optional. `get_research_strategy_guide` explains requirements for supervised,
few-shot, semi-supervised, meta-learning, transfer and custom methods. These are
development strategies, not an automatic claim that the selected method is novel.

Meta-learning evaluations require multiple training tasks and disjoint validation
and test tasks, with task IDs recorded for each observation. Legitimate support
observations can be part of a query input; its own target cannot. Semi-supervised
experiments require unlabeled observations with no target field or overlap with
labeled/held-out records. Review must inspect the actual implementation, meaningful
baselines, support/query design, ablations and uncertainty. These checks do not
prove that user-supplied grouping labels encode true protein homology.

## Inspecting inputs and saving progress

The shared file reader previews large text, sequence and table files in bounded
UTF-8 windows. Use the returned next_offset_bytes to continue reading, or analyze
the full dataset with isolated Python. A preview includes the full file size and
marks truncation; its slice hash is distinct from a whole-file hash.

Save useful partial results with submit_research_checkpoint. It is exposed
directly and also accepts the common call_tool route. When an audited executor
stops before saving a checkpoint, one bounded continuation can use the remaining
model-call, tool-call and time budget. It receives the verified dataset metadata,
measured generations and current execution feedback. Tool calls written as prose
remain text; only actual tool invocations create execution receipts.

## Generated model interface

The source module exposes two functions:

- `fit(training, seed)` receives a dictionary. Its `training["train"]` value is
  a list of observations, each containing an `inputs` dictionary and a `target`.
  Return a fitted estimator or a dictionary containing learned parameters and
  fitted preprocessing objects.
- `predict(model, inputs)` receives one input dictionary and returns an output
  dictionary such as `{"value": 1.2}`. The output key follows `prediction_key`;
  regression values must be finite numbers and classification outputs scalar labels.

These entrypoints are module-level functions; helper classes can implement the
underlying algorithm. Table features retain their original string representation.
Encode sequence and categorical features, convert numeric strings where needed,
and use the same transformation during fitting and prediction. Fit learned
transformations on the training partition and retain them with the model.

Dataset preparation returns a placeholder row/query example and a compact profile
of the training partition: stored types, empty fields, numeric-string counts,
distinct-value counts and length ranges. Profiles cover at most 16 fields and
explicitly report omitted fields. Verified resume context restores the same
profile; held-out observations and targets do not contribute to it.

A failed field lookup identifies the nested training-row access and the direct
prediction-input access. Array-shape errors explain the estimator's feature-matrix
requirement. Other estimator errors retain their own cause without unrelated
input-format instructions.

The platform persists fitted state, records dependencies and runs predictions in
isolated processes. Failed fits return the final exception and applicable input
contract; full execution logs remain in the training receipts. Before a worker is
started, Python compilation and interface checks collect diagnostics with line
numbers without importing or executing the submitted module. Rejected sources
retain a host-owned source snapshot and a failed preflight receipt. Editing the
workspace file does not overwrite that evidence, and rejected attempts count
toward the training-attempt budget. Focused repair receives the current source's
diagnostics, while completed generations still require actual fitting, validation
and independent Review.

## Failed attempts and measured replacements

A failed fit retains its source, logs and host-owned receipt under a stable attempt
ID. Scientist can revise the same file or implement a replacement in a new file.
A replacement resolves that attempt's execution blocker only after successful
fitting and fresh-process validation on the same frozen data, prediction field,
experiment role and parent lineage. An older model, a different dataset or an
ablation of another branch does not resolve it.

The host records the link between failed and replacement attempts and verifies
the receipts, frozen evaluation and runtime files. The Critic receives this link;
the failed attempt remains available as negative evidence. Resolving an execution
blocker does not approve the model: the selected candidate still needs independent
Review, reproduction and a delivery package. Historical attempts without the
required binding remain unchanged.

## Reusable reference controls

Call `get_research_baseline` without a reference ID to list four fixed controls:
training mean, amino-acid-composition ridge regression, random forest and support-vector
regression. Select an ID to obtain its complete `fit`/`predict` source, source hashes
and interface. Add `output_file="src/reference.py"` to save the selected source
directly in the research workspace; different existing contents are never overwritten.
Set `sequence_column` to the actual input field and `output_field` to the evaluation's
prediction key. Reading or saving a reference does not fit or register a model.

These adapters reuse the repository's classical controls: canonical amino-acid composition
and log sequence length, with target scaling learned from training data. They do not
represent assay covariates or mutation positions. They apply to regression datasets with
canonical protein sequences; other tasks need an appropriate custom baseline.

Save and fit a selected reference as `variant="baseline"`. Use the same frozen splits
to develop and measure a justified candidate. Keep the reference identity and code provenance
in the report; a supplied reference is not a Scientist-created method. The implementation
role remains free to build other representations and algorithms.

## Experiment protocol

1. Inspect data, provenance, assay conditions and available compute. Define fixed
   train/validation/test groups before optimizing a model.
2. For an uploaded table, call `prepare_research_dataset` with its measured
   target, feature columns and biological grouping. Write the model module
   described below, then call `train_research_model` to fit and evaluate a
   baseline. An existing fitted runtime implementing `run(inputs)` can enter
   through `evaluate_research_model` with `variant="baseline"`.
3. Use its validation feedback to revise a hypothesis, representation, algorithm
   or fitted model. Evaluate the child with `parent_experiment_id`.
   Set `evaluation_file=""` to reuse the parent's frozen evaluation table. If an
   explicit table is supplied, its inputs, targets and splits must match the
   parent. Training-time transformations may change; evaluation targets may not.
4. Repeat matched experiments and relevant ablations/seeds within the budget.
   Negative results are retained; superior accuracy is not mandatory for delivery.
5. Freeze the selected generation. Save `review/proposal.json` with the
   candidate name, description, source, runtime files, interface examples and
   `review_spec`. Set `review_spec.development_experiment_id` to the selected
   measured generation and submit with
   `propose_research_capability(proposal_file="review/proposal.json")`.
   Export the selected experiment's evaluation with `restore_research_evaluation(experiment_id, output_file)` and use the returned
   path in `review_spec.evaluation_file` and the evidence list. Exporting creates
   a new file and never overwrites different existing contents. The optional
   metric claim may be omitted; the host computes the reported score.
   After a revision request, edit the same draft and resubmit. Each attempt and
   admission receipt is retained separately; unchanged attribution and interface
   fields do not need to be re-entered. Inline submission remains supported.
6. Independent admission replays interfaces and final test predictions, recomputes
   metrics, reruns training without the test split, and checks scientific scope.
7. Approved models receive a private delivery under `workspace/deliveries/` and
   enter the configured registry. Online models retain the existing private
   registration policy; this change does not grant public access to user models.

The host snapshots runtime source and fitted artifacts outside the writable
session workspace, scores predictions itself and persists immutable receipts. Training-input
JSON is checked against the selected generation's frozen training view. Independent
Review receives its hash, fields and counts, rather than the complete measured
table; the unchanged private input remains available for actual refitting.
Identical evaluation tables share one immutable, content-addressed snapshot
within a study. Lineages require identical evaluation records. Older receipts
without a snapshot remain unchanged and require the original matching table. Renaming a model, editing only
comments, forging a displayed receipt or changing validation labels cannot pass
as a measured improvement. The public workspace copy is informational only.

Validation feedback uses MAE/RMSE or accuracy plus a training-mean/majority
reference. The final Review additionally checks a submitted metric claim against
recomputed test predictions. Dataset preflight, training and admission use one
partition validator in `research_evaluation.py`; measured generations and final
admission also share its scoring functions.
Train supports 5–20,000 rows, validation 2–512 and test 3–512. Every supplied
unlabeled pool is checked for targets, duplicate identities and overlap with
labeled partitions, including when used alongside a supervised control. Admission
applies the selected generation's strategy prerequisites. These checks run before
model evaluation; they do not rewrite existing experiment or admission receipts.

Developer-supplied splits are not a concealed
benchmark: the report explicitly discloses this. Repeated use of a supplied test
set does not establish unbiased generalization, causal RSI benefit or statistical
significance. Publication-grade claims still need sealed external evaluation,
matched budgets, repeated independent tasks and suitable ablations.

## Measurement access during autonomous model development

The protected CSV/TSV preparation tool supports ordinary group-disjoint splits,
whole-task episodes and a separate unlabeled pool through the same interface:

- Set `task_column` and `support_size` (1–64) to prepare task episodes. Tasks linked
  by a shared biological group stay in the same partition. Deterministic support
  selection leaves the remaining observations as queries; no task is silently
  discarded. The training partition must contain multiple tasks.
- Query rows have `task_id` and `inputs["support"]`, a list of same-task
  `{id, inputs, target}` observations. Fitting receives only training tasks.
  Validation and final predictions receive their own fixed support observations,
  never the query target. Support/query identities, inputs and task bindings are
  checked again during evaluation and admission.
- Set `unlabeled_file` and optionally `unlabeled_id_column` for a second table with
  the same selected feature columns and no measured-target column. Its selected
  fields enter `training["unlabeled"]`; unused columns stay private. Exact input
  and ID overlaps with all labeled support/query records are rejected.

Use optional `split_column` to retain a paper's existing `train`, `validation`
and `test` assignments. Every row needs one of these values (case-insensitive),
and all three partitions must be present. The partition column must be separate
from targets, inputs, groups, IDs and task identities. Biological groups and
connected task/group components cannot cross partitions. The host still validates
sample counts, duplicates, exact-input overlap and hidden-label boundaries.

With a supplied partition column, validation/test fractions do not assign rows;
the seed still selects the fixed support observations within each task. This
allows a new, explicitly documented source-training curation to retain earlier
held-out tasks and support/query observations. Admit a new study for revised
source data; an existing protected input cannot be repartitioned to reveal its
held-out observations. Preparation and restored Scientist context both identify
the supplied partition column. Without it, seeded grouping behaves as before.

These options can be combined. An unlabeled pool contains base features; task
query inputs additionally carry support observations. The model author chooses
how to use either resource. A supervised control can ignore them while preserving
the same frozen evaluation queries. Related sequences still require biologically
appropriate group annotations; schema checks do not infer homology.

Prepared evaluations remain bounded to 20 MiB and 21,024 source observations;
expanding support lists cannot bypass the size limit. Original uploads remain
private. Changing support/query roles or the labeled partition after preparation
is rejected. Resume verifies both source tables and restores their interface
metadata without including held-out observations in model context.

New model topics keep uploaded measurement tables in host-owned storage outside
the Agent workspace. Original uploads are copied and hashed in bounded chunks;
local uploads retain their existing size policy, while online uploads retain the
10 MiB limit. Dataset preparation accepts a documented CSV/TSV cohort up to
20 MiB and 21,024 rows. Larger originals remain stored privately.
Planning receives column names and inferred types. Preparing
a CSV/TSV table fixes the group split and exposes only its training rows; the
workspace evaluation file is an opaque reference resolved by the host.

Training, revisions, restored evaluations and independent reproduction use the
same frozen split. Generated code receives training observations, while the host
scores validation predictions. Independent Review additionally measures the final
test and receives aggregate evidence. Raw evaluation labels are excluded from
review prompts, runtime mounts and release archives. The original upload remains
available to its owner.

Changing the split, supplying an unregistered evaluation JSON or reusing an
Agent-visible legacy evaluation cannot acquire this isolation status. Existing
research records remain unchanged. Tables already exposed in an earlier run need
a fresh topic; retrospective masking cannot erase the previous model context.
Measurement tables in formats other than CSV/TSV remain protected and require
conversion before dataset preparation. A small JSON document containing only recognized
source metadata, such as a dataset title, source URL and license, remains readable;
measurement records and evaluation partitions stay protected.

The isolation tests exercise actual sandbox reads, initial and later uploads,
fitting, revision, restoration, independent reproduction, ZIP delivery and private
capability reuse. These are software tests. Public-data evaluation still requires
appropriate biological grouping and independent studies to support generalization.

## A coherent project and selected releases

The workspace convention is `src/` for reusable code, `training/` for training
reproduction, `artifacts/` for fitted state, `experiments/` for trials and
ablations, `reports/` for results, and `data/` for inputs. Existing uploaded paths
are preserved. The host owns `deliveries/`; the developer updates the project
README instead of creating another deployable model for each attempt.

The `scientist.artifact.v1` contract applies to learned models and reusable
methods/tools. The host verifies the saved independent Review receipt and source
hashes before packaging. Models additionally require a measured development
trajectory, fitted artifacts and training reproduction. Methods need interface,
boundary and independent checks but do not invent weights or predictive scores.

Each `artifact.zip` contains:

```text
README.md                      Use and reproduction commands
capability.json                Identity, author, version, scope and review hash
MODEL_CARD.md / METHOD_CARD.md  Intended use and limitations
predict.py                     JSON entry point with input/output validation
runtime/                       Selected code and fitted state, relative paths preserved
interface.json                 Input/output schema
examples/                      Example input and output
tests/                         Runnable interface and boundary checks
evaluation/                    Test report and independent review
documentation/                 Approved supporting Markdown/RST, when supplied
requirements.txt               Recorded dependency versions
checksums.json                 SHA-256 of package members
```

Raw evaluation data, rejected implementations and temporary files are excluded.
Files shows a README, card, test report and one ZIP per release. The UI shows
selected releases first; experiment records and previous versions remain
expandable. Project download expands the selected releases under `deliverables/`
alongside the research report and evidence. Drafts export their working project;
`include_working_files=true` explicitly includes working files for an admitted
project as well. No historical workspace files are deleted by this selection.

Keep a stable capability name and increment the version for a changed runtime.
One topic, artifact kind and protein/property/scenario scope define a release
family. Registration selects the last admitted version of that family for normal
discovery. Earlier immutable IDs still work for recorded reproducible calls.
Different topics or biological scopes remain separate; this is not automatic
semantic deduplication of unrelated studies. Existing catalog entries without
family metadata retain their previous visibility until explicitly re-registered.

The goal API exposes `research.development`, the full `research.deliveries`,
`research.selected_deliveries` and `research.delivery_pending`. Completion
revalidates selected delivery hashes and registration identities. Legacy goals
retain their original completion protocol.

## Scaling toward 10,000+ models

Count deployable, independently scoped and reviewed models, not experiments,
renamed copies or prompt variants. Each model needs an author, version, protein,
property, scenario, interface, resources, provenance, private/public visibility,
test report and reusable fitted artifact. Registry search loads selected models
on demand; the development protocol does not place the entire registry into the
LLM context. Ten thousand successful autonomous deliveries have not been tested.

## Validation

`tests/harness/test_research_development.py` executes actual CPU sandbox code for
small-data nonlinear fitting, semi-supervised label spreading and cross-task
initialization/adaptation. It checks immutable lineage, leakage rejection,
retraining, independent-review contracts, packaging and fresh Agent invocation.
Role/reviewer doubles in protocol tests are explicitly controlled fixtures;
they are not evidence of LLM autonomy or biological effectiveness. A separate
bounded free-model run records real request/tool evidence when available.

References: [MAML](https://proceedings.mlr.press/v70/finn17a.html),
[scikit-learn semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html).
