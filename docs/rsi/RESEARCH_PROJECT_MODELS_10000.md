# Research-project-driven acquisition of 10,000 specialized models

## Latest verified checkpoint - 25 October 2026

The source pool now has 3271 candidates: 3238 admitted tasks and 33 deferred,
across seven projects (stability 534, expression 2699, solubility 5). The added
2292 ProtPredict tasks retain all 19 measured features under a common genotype
holdout. Independent verification covers 2292000 exported observations and
43548000 feature values. No models were trained or published by preparation.
The new context interface passed 637 affected regression tests, 27 source/export
checks and replay of four real historical models. Paper-method controls and new
free-model acquisition experiments remain next. The target stays 10000 verified
callable models with broader domain coverage. See CONTEXT_V2_PROTPREDICT_20261025.md.


The target is 10,000 independently trained, validated and callable models for
protein research. The same modelling algorithm may be used for different proteins,
properties or measured scenarios. Algorithm diversity is reported separately.
Each model has an explicit scientific scope and an independently validated artifact.

## Model identity and coverage

The primary coverage axes are **protein or protein family × measured property ×
experimental or application context**. Different proteins sharing a property,
different properties of one protein, and distinct measured contexts may each yield
a counted model. Repeated seeds, renamed programs, repeated fits, arbitrary train/test
splits and prompt wrappers do not create additional scientific scopes. Revisions of
one scope are a lineage; the current released version contributes one primary count.
A shared backbone can support many specialized models when their learned components,
scope-specific validation and callable artifacts are explicit. A single multi-output
model is not multiplied by the number of questions it can answer.

Coverage includes solubility, expression, folding stability, thermal stability,
activity, kinetics, substrate specificity and other supported protein properties.
Scenarios use recorded variables such as expression system, assay method, substrate,
temperature, pH or formulation. Missing metadata remain unknown; do not fabricate
conditions or take a Cartesian product of unsupported combinations.

Researcher projects are the discovery unit. A project manifest records the original
study, protein scope, scientific question, measured endpoint and units, scenario,
source/permission evidence and observation groups. The source registry is
`configs/rsi/research_project_sources_v1.json`. Reviewed sources include eSOL protein production, curated designed-domain stability,
Arabidopsis thermal-context and food/fermentation yeast expression studies. SoDoPE, FireProtDB,
ProtaBank, SABIO-RK and researcher-provided projects are acquisition routes requiring
individual data review; listing a source does not imply it has been imported.

## Admission and implementation

`research_project_catalog.py` wraps existing model-task admission. It checks source
hashes, task/scope agreement, declared grouping and partition separation, minimum
data availability, duplicate scopes and duplicate measurements. It detects exact
sequence partition conflicts across admitted tasks. The initial feasibility floor
is 40 training, 15 validation and 15 future observations, with 5/3/3 declared groups.
These are software admission thresholds, not a power calculation or a guarantee of
biological independence. Group definitions require scientific review. A protein with
one observation remains a data opportunity, not a fitted protein-specific model.

Catalog entries are admitted, deferred, duplicate or invalid. Catalog sizes are not
published-model counts. Accepted tasks export to the existing persistent Scientist
campaign runner; the original manifest is bound into every exported task's references.
Campaign profiles may use authorized DMX free models or verified zero-priced
OpenRouter routes, with no paid fallback. Provider/model identities and budgets are
frozen per experiment. Catalog preparation and classical controls make no external
model requests.

The current bridge supports measured sequence-regression and contextual tasks. Classifier tasks are deferred
until their appropriate control suite is implemented. Existing generated-code model
interfaces cap some training tasks at 600 observations; increasing that capacity is
a separate sandbox/resource-validation change. No broad-scale autonomous literature
mining or 10,000-model acquisition has been completed by this catalog revision.

## Traditional and contemporary comparisons

Sequence tasks begin with training-mean, amino-acid-composition ridge regression,
random forest and support-vector regression controls, using 20 residue fractions
and log sequence length. Contextual tasks use their declared numeric covariates
with appropriate thermal-curve or transcript/culture controls. Hyperparameters are fixed before
validation; scaling is fitted on training data. The new runner scores validation
only, leaving the future partition for frozen final comparisons. These controls are
independently implemented host baselines, not Scientist-generated model discoveries.

Add domain controls when their inputs, endpoints, permissions and original training
overlap can be checked: SWI/Protein-Sol for compatible solubility tasks; additive
mutation and structure-energy models for suitable stability tasks; family averages,
homologue transfer and substrate/context regression for enzyme prediction. Imported
published predictors must be labelled as fixed external models. An inaccessible or
incompatible method is reported as such, never assigned a synthetic score. Modern
representation and deep-learning controls supplement this traditional comparison.

Compare methods on the same source data, partitions and endpoint. Predeclare tuning
budgets and count feature preparation, calls, CPU/GPU time and memory. Report
endpoint-specific errors, ranking, calibration where appropriate, and uncertainty
clustered by project/protein. Count **valid**, **better than a baseline**, **better
than the strongest eligible baseline**, and **reused successfully** separately.
Model count is not a statistical replicate count.

## Expansion and RSI experiments

1. Audit available data and prioritize concrete project demand, missing coverage,
   label quality and expected useful models per unit compute. Do not assign arbitrary
   quotas to properties or expand a source solely to reach 10,000.
2. Use the catalog to select reuse, adaptation or a new model. Train through the
   Scientist lifecycle, independently validate, register and demonstrate a fresh call.
3. Progress through 32, 128, 512, 1,000, 3,000 and 10,000 accepted scopes as planning
   checkpoints. Preserve a ledger for rejected, data-limited and failed tasks.
4. Freeze project/protein groups for evaluation before exposing their labels to
   acquisition or development-strategy selection. Separate interpolation within a
   known protein, transfer to a new protein and transfer to a new scenario.
5. Compare a fixed library, acquisition without reuse, acquisition with reuse, and
   evolving development strategy under matched total budgets. A persistent single
   agent separates long execution from expert-team coordination. An equal-budget
   extra-search control separates recursive improvement from additional attempts.
6. Promote development strategies on development projects, then measure their
   descendants on untouched projects and proteins. Trace strategy changes to model
   acquisition and subsequent research-task benefit. Library scale alone is not RSI.

The paper's central claim should be that a Scientist progressively turns research
needs into validated specialized models and learns to do this better. Report the
eventual observed model count, coverage and benefit; 10,000 remains a target until
verified. Keep the approved manuscript title unchanged.

## Primary references

- eSOL: https://doi.org/10.1073/pnas.0811922106 and https://doi.org/10.1073/pnas.1201380109
- SWI/SoDoPE: https://doi.org/10.1093/bioinformatics/btaa578
- Protein-Sol: https://protein-sol.manchester.ac.uk/software
- Designed-domain stability: https://doi.org/10.1038/s41586-023-06328-6
- FireProtDB: https://doi.org/10.1093/nar/gkaa981 and https://doi.org/10.1093/nar/gkaf1211
- ProtaBank: https://doi.org/10.1002/pro.3406
- SABIO-RK: https://doi.org/10.1093/nar/gkx1065

Reuse rights and original-study overlap are checked per data source. Counts of
measurements, proteins or database entries are not evidence of the number of
trainable and independently useful models obtainable from that source.


## Implemented scope-based completion — 24 October 2026

New campaigns now default to `scientific_scope.v1`: project-manifest-bound scope
identities and duplicate measurement groups determine completion, while raw
program entries remain a secondary count. Historical campaigns keep their prior
protocol. The change passed 180 tests and a read-only audit of three real newly
published scopes. At that checkpoint, the shared library's terminal audit had seven verified
entries across six task scopes. The original prepared catalog had 103 admitted tasks,
which are not 103 published models. See `SCIENTIFIC_SCOPE_COVERAGE_20261024.md` for
batch outcomes, comparisons and the remaining execution/scaling work.


## Federated acquisition and execution checkpoint — 25 October 2026

Three source manifests now contain 979 candidates: 946 admitted tasks and 33
deferred tasks. The admitted stages are stability (534), expression (407) and
solubility (5). These are acquisition opportunities, not trained models. A fixed
checkpoint replay verified 10 completed two-generation plant allocations across
7 proteins. The paired campaign is still running, so its final effect is unknown.

A 40-request free OpenRouter pilot produced no validated models. Its failures
motivate a versioned submission-to-host handoff contract, now covered by 221
passing tests and two historical-publication replays. Prospective benefit remains
to be measured. See `SCALING_AND_HANDOFF_20261025.md` for source, execution, budget
and evaluation evidence and the next controlled experiments.


## Source reconstruction checkpoint - 25 October 2026

ProtPredict adds 2292 source proteins with sufficient genotype-held-out
observations, but none are admitted yet: its 19 measured covariates require a
versioned interface beyond the existing 16-feature contract. The federated
catalog remains 946 admitted tasks. The source adapter passed 13 tests and a
full independent numerical reconstruction. The prospective handoff pilot
published and reused two first-generation models but completed neither
two-generation goal; both remained below the strongest fixed control.
See PROTPREDICT_AND_HANDOFF_20261025.md.
