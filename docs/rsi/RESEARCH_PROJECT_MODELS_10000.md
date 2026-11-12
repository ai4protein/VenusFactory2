# Property-first RSI research programme

## Objective

Develop a Scientist that turns protein-engineering questions and available evidence
into useful, independently evaluated models and reusable methods. Prioritize coverage
of **measured property, application scenario and independent source study**, followed
by protein coverage. A 10,000-model library remains a long-term scale target; reaching
that number through many copies of one endpoint does not complete the programme.

Maintain the three complementary modes:

- **Agent:** answer questions and use enabled tools through category search, compact
  capability descriptions and selected full interfaces. Test reuse of reviewed models.
- **Scientist:** PI, planning, implementation, experiments, critique, revision and
  independent delivery review. Preserve evidence, trained artifacts and a reproducible
  project with an English report. Test learning from earlier development experience.
- **Researcher:** gather protein-specific evidence, database records and computational
  results into a cited report and a downloadable material package, using specialist
  subtasks when appropriate.

Harnesses and model providers remain replaceable. Research runs use explicitly
budgeted free DMX/OpenRouter models, without paid fallback. Keep production services,
other users' files and online Overleaf separate from research execution. The manuscript
remains **Recursive Self-improvement AI Agents for Protein Engineering**. Claims follow
completed experiments and their actual scope.

## Property and scenario coverage

The next pilot allocates up to eight tasks in each of six families. Each family aims
for at least two independent studies, at most four tasks from a study, and no more
than two allocations per protein across the panel. These are sampling constraints,
not statistical power calculations. Insufficient data leave explicit gaps; another
property cannot consume that allocation.

| Family | Representative measured endpoints | Scenario distinctions to retain |
| --- | --- | --- |
| Enzyme catalysis | Activity, kcat, Km, Ki, substrate specificity | Substrate, assay, pH, temperature, new enzyme |
| Molecular recognition | Kd, binding and specificity | Target, ligand, construct, assay, selectivity panel |
| Stability and resilience | Folding free energy, thermal stability, measured stress tolerance | Sequence variants, temperature response, pH and solvent conditions |
| Production and developability | Solubility, expression yield, aggregation | Expression host, preparation, concentration, construct |
| Localization and transport | Localization, secretion, targeting | Organism, compartment, construct and assay |
| Immune properties | Immunogenicity, antigenicity, allergenicity | Host, assay and experimentally established endpoint |

The existing protein-abundance library is a retained specialist collection. Its new
allocation is **zero in this pilot**. It remains available for dedicated studies and
appropriate reuse; its scale is not evidence of broad protein-engineering coverage.
A thermal-response curve and sequence-dependent stability prediction are reported as
different input/scenario tasks even when both belong to the stability family.

The policy lives in `configs/rsi/property_coverage_v1.json`. Classification uses
explicit stage/endpoint rules. Unknown endpoints remain unclassified until reviewed;
renaming a task does not create a property. Canonical study identifiers prevent project
aliases from multiplying the same source quota. New endpoint rules require scientific
review of the measured quantity and the actual inputs.

## Data admission and counting

Use public measurements with original study references, reuse terms, measurement
units, uncertainty/censoring information and a traceable transformation record. Keep
source files and hashes. Deduplicate shared measurements across databases. Preserve
study-provided partitions or freeze grouped partitions before development; record
sequence, protein, study and assay overlap as applicable.

A protein with one observation is not enough for a dedicated predictor. The source
catalog checks minimum rows and independent groups. Classification tasks additionally
need appropriate class-aware metrics and controls; they are not relabelled as regression
to satisfy the pilot. Shortfalls guide source acquisition and adapter work.

Count these quantities separately:

1. Property families and measured endpoints covered.
2. Application scenarios, input types and independent source studies.
3. Protein identities and distinct protein/property/scenario scopes.
4. Generated programs, trained artifacts, independently accepted deliveries and
   successfully reused models.
5. Lineage revisions and original methods, with evidence for any novelty claim.

Repeated seeds, refits, renamed programs, prompt wrappers and arbitrary partitions do
not add scientific scopes. A multi-output model is not multiplied by the number of
questions it can answer. The same algorithm may serve distinct valid scopes; algorithm
diversity is a separate result.

## Curate the existing library

Keep a compact default catalog without removing distinct protein, property, assay,
scenario or interface coverage. Sharing an algorithm across different proteins is not
by itself redundant. Similar scores on different datasets are not a comparison.

`configs/rsi/model_retention_v1.json` defines the host-operated retention policy.
The planner compares entries only when their full scope, input interface, implementation,
training partition and validation partition agree. It verifies the archived numerical
predictions and their evidence bindings before using recorded validation metrics.

- Identical validation predictions and objectives retain the smaller serialized model;
  equal-size ties use a deterministic identity. This establishes agreement on the stored
  validation rows, not universal equivalence.
- A model may also be archived when a retained model is no worse on every configured
  validation objective, strictly better on at least one, and has no larger serialized
  payload. This is a catalog-selection rule, not a statistical significance claim.
- Preserve metric tradeoffs, smaller-payload alternatives, subgroup advantages and
  incomparable evidence. Serialized payload size does not measure latency or peak memory.
- Preserve entries that require runtime-migration review. Missing evidence does not
  justify declaring a model redundant. Unique scopes are retained without new fitting.

From the repository root, generate and inspect a plan before applying it:

```bash
python scripts/rsi/evaluation/curate_model_library.py \
  --library research_runs/rsi/scientist_model_library_v1 plan \
  --output research_runs/model_retention_plan.json

python scripts/rsi/evaluation/curate_model_library.py \
  --library research_runs/rsi/scientist_model_library_v1 apply \
  --plan research_runs/model_retention_plan.json

python scripts/rsi/evaluation/curate_model_library.py \
  --library research_runs/rsi/scientist_model_library_v1 restore \
  --entry-id <full-entry-id> --reason "Reconsider for a distinct use case"
```

Applying a plan rechecks the catalog and evidence, and records a reversible curation
event. Default Agent/Scientist capability discovery and Model Plaza use the same filtered
metadata. Each archived entry points to a retained replacement; archive/restore changes
invalidate discovery caches. This is a host operation, not an autonomous publication tool.

Original programs, fitted state, evaluation files, publication objects and frozen
experimental histories remain intact. Explicit entry-ID lookup keeps existing research
references usable, subject to the usual evidence and execution checks. If a replacement
is no longer canonical, the original becomes discoverable again. Archiving is distinct
from an administrator disabling an unsafe or unavailable capability.

Report historical registrations, active catalog entries, archived revisions and distinct
scientific scopes separately. Curation neither changes historical experiment results nor
adds accepted models. Preserve frozen RSI comparison libraries and donor histories;
new live discovery uses the curated catalog. Filling missing properties requires new
measurements and evaluated models, not discarding unique abundance scopes to improve a
percentage.

## Allocate a property-balanced pilot

From the repository root:

```bash
python scripts/rsi/data_preparation/plan_property_campaign.py \
  --manifest research_runs/my_sources/manifest.json \
  --exclude-panel research_runs/my_prior_panel/allocation.json \
  --registered-catalog research_runs/my_registry_snapshot/catalog.json \
  --output research_runs/my_property_pilot
```

Use real source-bound manifests and applicable prior snapshots. The output contains
`catalog.json`, `allocation.json` and `coverage.json`; it performs no model requests or
training. A partial allocation reports missing families and source diversity. Review
these gaps before choosing additional sources. Source, policy and selected task hashes
are rechecked before the allocation is returned. Omit an exclusion option only if there
is no corresponding history, and record the actual exclusion scope.

The existing `plan_acquisition` function accepts a bound `coverage_policy` reference.
Its legacy no-policy behavior remains for reproducing frozen historical experiments;
new breadth-oriented cohorts use the property policy. Do not overwrite closed studies
to make their composition appear balanced.

## Experiments and acceptance

First complete non-abundance pilot cases, then expand successful areas without
letting one property or study dominate subsequent cohorts. Select tasks from source
metadata before inspecting model performance. Log rejected, unsupported, data-limited
and failed tasks as well as successful deliveries.

Scientist receives the question, measured inputs and resource budget. It chooses and
implements appropriate methods, including classical learning, few-shot, semi-supervised
or meta-learning strategies where the data support them. Compare methods on identical
inputs and partitions with fixed domain controls, and report tuning, compute and request
costs. A useful reviewed model need not outperform every traditional baseline.

RSI experiments compare a frozen library, autonomous development without history,
development with reusable programs, and development with evolving strategy under
matched budgets. Use untouched projects for transfer claims and equal-budget search
controls. Report within-protein prediction, new-protein adaptation and new-scenario
transfer separately. Model-library growth alone does not establish improvement.

SC is the independent scientific acceptance role. It receives the objective and
host-bound experimental evidence without the developer's endorsements. It checks
usefulness, evaluation validity and supported scope; an unsupported or unhelpful model
returns to PI with specific revision requirements. Passing SC is followed by independent
technical Review and reproducible private delivery, not automatic public publication.

Saving a research checkpoint ends the native implementation turn after its MCP receipt
and trace have been verified. It does not mark the research complete. The coordinator
stops further implementation requests and invokes SC in a separate context. Queued
workspace mutations cannot change the saved result after this boundary.

Private model delivery requires reproducible training, real fitted state, input/output
checks, scope and resource limits, evaluation evidence and a successful fresh invocation.
Public platform admission is separate: an operator selects and reviews local products;
external contributors apply by email, with author email retained in approved metadata.
Scientist cannot automatically publish an anonymous user's model to the public catalog.

## Immediate execution order

1. Resolve the native model-interface failures observed in the closed CatPred Ki/Km
   studies, then run a separately budgeted Scientist trial and matched reference
   comparison. The closed pair produced no accepted model and stays in the acquisition
   denominator. Ki/Km are two endpoints from one data source, not two independent studies
   and not evidence of all six families.
2. Audit the existing eSOL, sequence-stability, enzyme-assay and antibody data for
   reusable partitions and eligible sources before allocating new studies.
3. Fill missing localization, secretion and immune-property coverage with measured,
   source-reviewed data and the required classification adapters and controls.
4. Run matched RSI/history and held-out transfer experiments on the admitted panel.
5. Test accepted products through Agent discovery and invocation, then update paper
   results and Researcher examples from completed evidence.

## Sources for the missing properties

| Source | Intended coverage | Required admission work |
| --- | --- | --- |
| [SKEMPI 2.0](https://life.bsc.es/pid/skempi2/) | Mutation-dependent protein interaction affinity and kinetics | Preserve exact/censored measurements and temperature; map wild-type/mutant residues to structures; group related complexes and original studies before splitting. Do not combine Kd, kon, koff and derived binding-energy changes into one target. |
| [Jain et al. antibody measurements](https://www.pnas.org/doi/10.1073/pnas.1616408114) | Measured antibody expression, thermal stability, self-interaction and nonspecific binding | Join paired variable-domain sequences from Dataset S2 to assay measurements in S3; retain assay-specific units and construct context. One cross-antibody endpoint model is a scope, not one fitted model for every antibody row. |
| [DeepLoc 2.0 source data](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/) | Experimentally supported subcellular localization and sorting signals | Preserve the published homology partitions and independent HPA test cohort. Add multilabel metrics and classification controls before task admission; unavailable labels are not negative observations. |
| [IEDB assay exports](https://www.iedb.org/database_export_v3.php) | Measured T-cell/B-cell response and peptide–MHC binding in defined contexts | Keep assay type, host, MHC, response definition and source publication. Functional response, binding and ligand elution are different endpoints; use explicit negatives and study/sequence-aware splits. |

These are source-admission priorities, not trained models or accepted tasks. Downloaded
snapshots stay under ignored `research_runs/sources/` with URLs, dates and checksums.
Source availability alone does not imply an adequate independent training/test split.
The CatPred Ki and Km experiments both occupy the enzyme-catalysis family in this policy;
they do not fill the molecular-recognition quota simply by renaming the endpoint.

### Curate the localization source

`import_deeploc_source.py` validates an existing, hash-bound source snapshot and
writes one localization dataset with Swiss-Prot and HPA cohorts. It retains the
ten-label output space, source annotations, original 0..4 folds, and sequence
identity hashes. Extracellular and plastid labels are null in HPA, which provides
only eight localization columns. Source zeros retain their annotation meaning;
they are not additional experimentally confirmed absence measurements. Membrane
annotation and kingdom stay metadata and must not become predictive inputs.

```bash
python scripts/rsi/data_preparation/import_deeploc_source.py \
  --manifest research_runs/sources/my_snapshot/manifest.json \
  --sha256 <frozen-source-manifest-sha256> \
  --output research_runs/sources/my_deeploc_curated
```

The command never downloads, calls an LLM, starts training or registers a model.
It checks source sizes, hashes, CSV schemas, binary labels, protein identities,
sequence lengths and exact-sequence split overlaps. A new output directory is
required. Counts, source coordinates and output checksums remain in `source.json`;
raw source files remain unchanged. The author's homology partition is preserved,
not independently recalculated. Full folds and bounded pilot subsets must be
reported separately; no silent truncation to Scientist's row limits is permitted.

Native Scientist now accepts `task=multilabel`: select a CSV/TSV target column
whose cells contain JSON objects, for example `{"Cytoplasm":1,"Nucleus":0,"Plastid":null}`.
Every row keeps the same 2..64 label names. Each label requires observed training
data; null values are omitted from its loss and evaluation. `fit` receives
`training["labels"]`; `predict` returns all label probabilities inside `value`
(or the chosen prediction key). Independent labels need not sum to one.

Development and independent Review share masked macro Brier as the selection
metric and a training-only per-label prevalence reference. Reports include micro
Brier, noninterpolated average precision, fixed-0.5 F1 and per-label observation
counts. AP is undefined without both classes and excluded from macro AP in that
case. No threshold is selected from test labels. Support/query tasks additionally
report equal-task scores and support prevalence; a label absent from support uses
the training prevalence, with the fallback count disclosed. The downloaded
package preserves these reports and replays independent probability invariants.

The localization pilot still needs a declared cohort/full-fold protocol, relevant
trained controls, data reuse review and biological performance evaluation. Software
protocol tests do not count as localization models. One source's labels, folds and
protein rows do not multiply its study or model counts.

Metric definitions: [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
and [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
