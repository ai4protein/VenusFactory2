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

## Delivered files

- `model.zip`: runtime source, fitted state, training source, dependency versions,
  interface/examples, documentation and checksums; raw evaluation data excluded.
- `MODEL_CARD.md`: purpose, applicable proteins/properties/scenarios, limitations,
  inference interface and training reproduction instructions.
- `test-report.md` and `test-report.json`: validation lineage, host-recomputed
  final test metrics, reference baseline, interface/boundary tests, independent
  review, reproduction evidence and limitations.

These files appear in the conversation's Files/Library and use the existing
authenticated preview and ZIP download routes. The goal API also exposes
`research.development`, `research.deliveries` and `research.delivery_pending`.
Completion checks revalidate delivery file hashes and registration identities.

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
