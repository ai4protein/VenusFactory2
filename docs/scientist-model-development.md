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

SC also receives a bounded history of host-recorded validation before a model
proposal exists. The history preserves a baseline, recent measured revisions,
exact-prediction matches and receipt identities. A missing submission is distinct
from missing evaluation. Study-level objections return to PI even when no model
assessment was possible; they do not discard earlier experiments or bypass the
separate model-admission checks.
For study feedback before submission, SC must cite experiment IDs from that
history. Unknown, missing or duplicate citations trigger the same bounded output
repair as a malformed review. Measured progress precedes the current failure
summary; repeated failed-tool envelopes and general Agent discovery instructions
are omitted, while successful observations and failure summaries remain. Citation
checks bind the opinion to a study; they do not by themselves prove its scientific
interpretation correct.

New interim reviews also receive a host-owned measurement state and return typed
revision requests. A request claiming absent validation, partition records, a
baseline or a revision is checked against that state. Reviewers may still question
the scientific adequacy of existing evidence. Conflicting requests receive one
bounded correction attempt; an unresolved conflict preserves the measurements
and records `review_evidence_inconsistent` without model acceptance. PI receives
verified state and revision categories separately from reviewer commentary.
Historical reviews retain their original policy and receipts.

SC receives a bounded table of receipt-bound validation metrics before authored
reports. Each value identifies its evaluation, metric, direction and aggregation:
pooled query scores and equally weighted task scores remain separate. Compatible
model/control and child/parent comparisons are calculated by the host. These
relations describe numerical differences, not statistical significance or scientific
usefulness. Reference tables supplied as research materials retain their own
provenance and are not converted into host-measured controls.

A negative verdict or missing evidence prevents technical Review and registration,
persists an actionable request for PI, and continues the research loop. Accepted
verdicts are bound to the selected experiment, runtime, evaluation and manifest
hashes. Completion rechecks the saved evidence and verdict; changing a model or
receipt cannot inherit the previous acceptance. Independent technical Review uses
its own fresh context and does not receive SC's endorsement. Successful deliveries
include the SC assessment and technical test report. Closed historical protocols
remain unchanged; resumed topics use the new acceptance policy.

## Computation evidence and working context

Completed MCP computations can provide execution evidence without fitting a model.
The worker binds the task, arguments, result and declared input/output file hashes;
the research host saves a scoped result under `analysis/`. A checkpoint must cite
that result, and acceptance rechecks all bound files. Failed or pending jobs,
modified outputs and model-written claims do not produce execution evidence.
Reading a sequence or calculating its properties does not establish a validated
training split, a fitted model, or scientific usefulness. Those remain separate
checks and SC decisions. The plan verification name `data_partition` specifically
requires protected training-data partitions; legacy `data` keeps that same meaning.

Across episodes, the host reloads its private computation receipts and rechecks
workspace ownership and input/output hashes. Valid earlier calculations can support
an updated report without being recorded as new executions. Changed files or
conflicting task identities cannot inherit that evidence. Verification is bounded;
current and checkpoint-cited results take priority over older inventory.

Analysis uses an interim SC review until a complete, unchanged Markdown or text
report is checkpointed. Interim feedback acknowledges verified computations and
returns concrete next actions to PI, but cannot complete plan steps or approve
registration. Final SC binds each computational claim to an actual task result,
the supplied input and an exact report passage. Source files, raw inputs and an
exit code alone do not establish a final report or the validity of a scientific claim.

Native execution keeps the objective, scientific criteria, plan requirements,
current feedback and prepared-data contract in context. Longer historical records
and duplicate plans may move into bounded, hash-checked reference files read on
demand. PI, SC and host validation retain the canonical context. References do not
increase tool budgets or replace experimental receipts.

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

## CPU and PBS GPU training

Agent-authored `train_research_model` fits use CPU by default; the opt-in PBS
backend below supports isolated GPU fitting and CPU prediction. Reuse through
`run_generated_capability` uses the CPU queue, including when GPU slots are busy;
prediction quotas and session ownership still apply. Platform PLM
training uses enabled tools: `generate_training_config` prepares the
session data and settings, `train_protein_model` produces a fitted checkpoint,
and `protein_model_predict` checks the resulting model. These calls use the
shared CPU/GPU job queue and preserve their parent research job and session
ownership. Online training retains the ESM2-8M policy.

On a PBS cluster, request GPU, CPU, memory and wall-time resources before
starting the research process and its tool workers on the allocated node. Honor
the scheduler's device binding; do not select a host GPU outside that allocation.
Use a separate queue directory for each independent experimental allocation.
The current SQLite queue is a single-host service, not a distributed queue shared
between PBS nodes. GPU allocation alone does not expose devices to session
Python or install missing dependencies; authored GPU fitting requires the explicit
PBS backend.

Record the PBS job, allocation, tool job IDs, fitted checkpoint, prediction result
and terminal states with the experiment. A completed training integration test
establishes execution, while a scientific model still requires measured utility,
independent SC acceptance and technical Review.

## Submit a complete Scientist goal to PBS

The command-line PBS runner places the complete research loop on one allocated
node, including its isolated training and prediction processes. It is separate
from the web service allocation. This initial resource profile requires one GPU;
the model can still choose CPU experiments within that allocation.

Copy [the configuration example](../configs/pbs-goals.example.json) to a private
configuration file and adapt its queue, Python environment and installation root.
Relative paths resolve from the configuration file's directory. `runtime_root`
is the project installation root containing optional harness runtimes;
`control_root` holds private job records and frozen source snapshots.
`goal_roots` bounds the research directories this controller may execute.
The selected Python environment must contain the training dependencies; optional
`extra_python_paths` are trusted host dependencies, not extra sandbox mounts.
The worker uses a private HOME. For a globally installed Kimi, set
`environment.VENUS_KIMI_BIN` to its absolute executable path; runtime paths inside
`environment` are not resolved relative to the configuration file.

Keep provider keys in the administrator-owned `env_file` or configured credential
file. Do not put key values in the PBS configuration. The worker reads only the
selected provider credential after validating its allocation. The model route
must be admitted as free; there is no paid fallback.

Create a goal with its profile and budget, then submit it from a clean, committed
checkout. For an already prepared goal, start at `submit-pbs`:

```bash
python scripts/rsi/run_autonomous_research.py create \
  --output runtime/local/scientist-goals/my-topic \
  --profile configs/my-scientist-profile.json \
  --objective "Develop and evaluate a predictor from the supplied measurements." \
  --input ./measurements.csv --input ./paper.pdf
python scripts/rsi/run_autonomous_research.py submit-pbs \
  --output runtime/local/scientist-goals/my-topic \
  --pbs-config configs/pbs-goals.local.json --resource-profile gpu
python scripts/rsi/run_autonomous_research.py status-pbs \
  --output runtime/local/scientist-goals/my-topic \
  --pbs-config configs/pbs-goals.local.json
python scripts/rsi/run_autonomous_research.py cancel-pbs \
  --output runtime/local/scientist-goals/my-topic \
  --pbs-config configs/pbs-goals.local.json
```

Repeat `--input PATH` during `create` for each supplied file, up to 32 files with
unique filenames. The CLI validates all inputs before creating the goal, copies
independent snapshots to `workspace/inputs/`, and records their original name,
size, SHA-256 and relative path in `progress.inputs` with `source_type="cli"`.
External file paths are not included in model context. Files generated later in
the workspace are not treated as uploaded materials.

Model-development tables use the same protected data store as Web uploads;
the workspace exposes metadata and, after preparation, training rows. Raw labels
remain outside the execution sandbox. Source files are not modified. Failed
preparation retains its files and `cli-input-preparation.json` diagnostics and
leaves the goal interrupted. Correct the inputs and create a new goal.
`--input` is not accepted by `run`, `resume` or PBS commands; later attachments
use research revision. These local paths are unavailable in online mode, where
inputs must use the existing session upload flow.

Submission freezes the source revision and records the goal, profile, resource
request and scheduler identity. Status and cancellation use that recorded job.
A later attempt requires `--new-attempt`; add `--resume-goal` when the goal itself
needs to resume. Neither flag grants a new model-call budget. Direct `run` and
`resume` refuse a goal owned by an active PBS execution. Existing observations,
checkpoints and model artifacts remain part of that goal across attempts.

Each PBS worker starts an authenticated MCP scheduler and a bounded worker pool
inside its allocation. Tool jobs carry a real parent lease and use the research
workspace. A stopped research task cancels its children and stops those services;
the web service queue and its processes are separate.

Submission freezes the installation's local administrator switches, model runtime
paths and free LLM catalog. Disabled capabilities remain disabled in that attempt.
Relative model paths are resolved against `runtime_root`, so the source snapshot
does not need a copy of installed models. Optional `capability_db`,
`model_runtimes` and `execution_catalog` select explicit administrator files.
`checkpoint_root` selects installed Factory and ProteinMPNN checkpoints (default:
`runtime_root/ckpt`). `huggingface_cache` optionally selects an existing Hub cache.
These paths are passed to trusted runtime workers; each job keeps its own home
and research workspace. The asset manifest binds directory
locations, not immutable weight contents.
These are execution snapshots: later Web policy changes require a new attempt.
The standalone PBS identity does not grant access to other Web users' private
models; live Web identity and policy integration remain separate.

To enable configured search services, list only the needed variable names in
`tool_credential_refs`, for example `TAVILY_API_KEY` or `SERPAPI_API_KEY`.
Their values are read from the protected `env_file` into trusted MCP processes;
model-generated code and native harness processes do not receive them. LLM
credentials are not exported to the tool service.

This CLI does not automatically submit web Scientist topics to PBS. GPU allocation
also does not establish scientific success: SC acceptance and technical Review
still govern model delivery.

## Repeated experiments and model changes

A renamed source or another parent link does not create a new model. If an
algorithm and its fitted state have already been measured on the same prepared
training data and evaluation split, fitting returns that experiment's identity
in a diagnostic instead of adding another generation. The attempted fit and its
source are retained for inspection.

Validation also compares the declared prediction values with earlier experiments
on the same frozen inputs. Exact matches are shown to PI and SC even when random
seed metadata or extra output fields differ. Equal aggregate scores alone are
not treated as equal predictions, and a match on validation inputs does not prove
equivalence outside them. A change may still have practical value, but benefits
such as lower cost or a different use case need evidence in the SC assessment.

## Inspecting inputs and saving progress

Supplied source is inspected before a new implementation is written. For focused
repairs, `edit_session_file` changes one exact fragment using the version hash
from `read_session_file`. Compiler errors include a bounded source excerpt and
column marker. A repair still requires a new training and validation attempt;
previous source snapshots and failed receipts remain unchanged.

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
canonical protein sequences.

For a sequence multilabel task, request `task="multilabel"` on the same tool. It
provides per-label prevalence, composition logistic regression and logistic
regression with N/C-terminal composition. These controls use all observed training
labels, skip null cells, retain ambiguous residues in a separate composition bin,
and fit scaling only on training rows. A label with one observed class uses its
training prevalence. Probabilities are independent; they do not sum to one. These
are sequence-only controls, not pretrained DeepLoc models or biological discoveries.

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


## DeepLoc localization study preparation

After `import_deeploc_source.py`, run
`scripts/rsi/data_preparation/prepare_deeploc_task.py --source <source.json>
--sha256 <frozen-source-hash> --output <new-private-directory>`.
The exporter verifies both curated assets and preserves the original DeepLoc 2.0
folds: 0/1/2 for training, 3 for validation, and 4 for internal test. It retains all
17,145 training rows. The native evaluator currently supports 512 rows per holdout;
a fixed sequence-hash ordering selects each holdout without looking at labels.
The protocol records every selected and omitted protein, source hashes and source
terms. One ten-output localization task counts as one model task.

`dataset.csv` is host-only staging input. Ingest it with `ResearchDataStore` and
use the frozen `native_prepare` parameters from `protocol.json`. Scientist then
sees only training rows and an opaque evaluation reference. Kingdom, membrane
annotations and partition IDs are not predictive input features. The 1,717-row HPA
cohort remains external to fitting and model selection. Source-coded zeros are
not independently confirmed absence assays; unobserved HPA locations remain null.

Run `scripts/rsi/evaluation/measure_deeploc_references.py --protocol <protocol.json>
--sha256 <frozen-protocol-hash> --output <new-run-directory>` in a bounded compute
job. It fits the three fixed controls through the same isolated training and
fresh-process validation API used by Scientist. It makes no LLM calls, does not
score final test/HPA, and does not register models. This validates the data and
execution path; autonomous development and utility acceptance are separate steps.
The bounded holdout study is not the published five-fold benchmark. Full-fold and
external evaluation require a frozen candidate and a separate declared protocol.

Sources: [DeepLoc 2.0 data](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/),
[publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC9252801/), and the authors'
[code license](https://github.com/teevee112/DeepLoc-2.0/blob/main/LICENSE).

## Research request and evidence boundaries

Native implementation has a physical API-request ceiling that leaves three calls
for SC and technical Review. Retries count toward that ceiling. If the native
phase reaches it without a valid checkpoint, admitted tools are stopped before
SC reviews the preserved partial results. This review cannot complete the plan
or publish a model; the next episode continues from the saved work. Native
failures after host-recorded scientific tool attempts use the same partial-review
path. Startup failures without such evidence and external cancellation do not.

For analysis deliverables, checkpoint the report and the recorded computation
artifacts. SC cites exact report quotations, result fields and input provenance.
The host checks file hashes and declared values, including numeric display
precision; SC judges whether those results answer the original research question.
This is separate from measured-model usefulness review and technical admission.

## Authored GPU training on PBS

Use `submit-pbs`, `status-pbs` and `cancel-pbs` from the section above for
standalone research jobs. They freeze the source and track the allocation with
an execution record.

The older `scientist_gpu.pbs` template below is a **low-level recipe** for running
a goal inside an operator-managed PBS allocation. Its direct `qsub` command does
not use the durable controller, freeze a source snapshot or reconcile submission
receipts. Keep it for manual allocation integrations, rather than as an equivalent
entrypoint to `submit-pbs`:

```bash
qsub -q YOUR_GPU_QUEUE \
  -v VENUS_SERVICE_PYTHON="$CONDA_PREFIX/bin/python",VENUS_RESEARCH_GOAL="runtime/local/scientist/my-topic" \
  scripts/examples/scientist_gpu.pbs
```

The template requests one GPU, two CPUs and 16 GiB of memory in a separate PBS
research job. It creates a job-specific queue and sandbox reservation directory;
it does not reuse the web service's SQLite queue across compute nodes. The goal
retains its existing profile, data, request budget and checkpoints. Configure its
provider credentials privately using the normal deployment setup. A paused goal
requires an explicit resume through the goal interface.

`get_research_strategy_guide` reports currently verified training devices.
`train_research_model(device="cuda")` fits authored Python in bubblewrap using
only the GPU assigned by PBS. This opt-in path requires
`VENUS_RESEARCH_GPU_BACKEND=pbs`, local mode, an unambiguous GPU UUID, and verified
cgroup v1 memory/device membership and permissions. Unsupported cgroup layouts,
missing allocations and online requests fail before authored execution. There is
no host execution or CPU fallback. Normal `device="cpu"` calls remain unchanged.

The sandbox keeps its network, filesystem and syscall isolation. Its private
process directory permits CUDA thread naming, while kernel controls remain
read-only. CUDA needs a large virtual address range, so GPU execution uses the
PBS allocation's kernel memory cap plus the configured per-process-tree resident
memory watchdog instead of the CPU address-space limit. The hard cap covers the
whole PBS job; the smaller watchdog limit is checked during execution. A sudden
allocation can therefore terminate the research job at its PBS limit. Queue,
wall-time, output/storage limits and cancellation cleanup still apply.

GPU fitting accepts `timeout_seconds` (default 120, maximum 1800). CPU fits keep
their 120-second maximum; ordinary CPU session execution and online limits are
unchanged. The requested allowance is bound to the measured experiment and its
training receipt; independent Review uses that same allowance. Validation and
prediction keep their existing CPU limits. Existing GPU records without a duration
retain their original 120-second allowance.

The native research phase and total goal budgets must leave time for the fit,
evaluation and later Review; a larger tool allowance does not extend those budgets
or the PBS walltime. When a phase exits or is cancelled, its admitted MCP calls are
cancelled and their sandbox cleanup is awaited before the next phase starts.

Return CPU-portable state from `fit`: for example, detach tensors to CPU or save
numeric parameters. Validation, final-test prediction and Agent reuse remain CPU
operations. The measured experiment binds the GPU training receipt; independent
Review must repeat that fitting step inside a verified GPU allocation and then
replay its predictions on CPU. A successful fit alone cannot publish a model.
The initial backend is for an allocated local research worker; it does not submit
a new PBS job for each tool call or enable GPU-authored training in the web UI.

SC uses an interim progress review before a measured model is submitted. It
separates completed validation from later failed fits and unfinished submission
or delivery. Submitted candidates still receive the full independent usefulness
assessment before technical Review; interim feedback cannot approve a model.
