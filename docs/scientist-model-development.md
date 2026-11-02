# Scientist: model development and recursive improvement

Scientist uses a goal-driven PI → planning → implementation → Critic → revision
loop. A separate admission Review controls registration. Model-development goals
now additionally require a measured development trajectory and a model delivery;
neither a training script nor a favorable research Critic can satisfy that gate.

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

## Experiment protocol

1. Inspect data, provenance, assay conditions and available compute. Define fixed
   train/validation/test groups before optimizing a model.
2. Fit a baseline; save a module implementing `run(inputs)` and its fitted state.
   Call `evaluate_research_model` with `variant="baseline"`.
3. Use its validation feedback to revise a hypothesis, representation, algorithm
   or fitted model. Evaluate the child with `parent_experiment_id`.
4. Repeat matched experiments and relevant ablations/seeds within the budget.
   Negative results are retained; superior accuracy is not mandatory for delivery.
5. Freeze the selected generation and provide its ID as
   `review_spec.development_experiment_id` in `propose_research_capability`.
6. Independent admission replays interfaces and final test predictions, recomputes
   metrics, reruns training without the test split, and checks scientific scope.
7. Approved models receive a private delivery under `workspace/deliveries/` and
   enter the configured registry. Online models retain the existing private
   registration policy; this change does not grant public access to user models.

The host snapshots runtime source and fitted artifacts outside the writable
session workspace, scores predictions itself and persists immutable receipts.
Lineages require identical evaluation records. Renaming a model, editing only
comments, forging a displayed receipt or changing validation labels cannot pass
as a measured improvement. The public workspace copy is informational only.

Validation feedback uses MAE/RMSE or accuracy plus a training-mean/majority
reference. The final Review additionally checks a submitted metric claim against
recomputed test predictions. Developer-supplied splits are not a concealed
benchmark: the report explicitly discloses this. Repeated use of a supplied test
set does not establish unbiased generalization, causal RSI benefit or statistical
significance. Publication-grade claims still need sealed external evaluation,
matched budgets, repeated independent tasks and suitable ablations.

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
