# Published GFP and TrpB prediction tasks

The Scientist task interface now admits 14 additional retrospective prediction
tasks: four GFP families and ten landscapes from one TrpB enzyme construct.
They use `generated_sequence.v1`, preserve complete published sequences and can
be passed to the existing Scientist model campaign as ordinary `model_tasks`
declarations. Preparing a task does not call an LLM or publish a model.

## Papers and data

| Source | Tasks | Prediction endpoint |
| --- | --- | --- |
| [Sarkisyan et al., Nature 2016](https://www.nature.com/articles/nature17995) | avGFP | Already logarithmic `medianBrightness` |
| [Somermeyer et al., eLife 2022](https://elifesciences.org/articles/75842) | amacGFP, cgreGFP, ppluGFP2 | `log10(replicates_mean_brightness)` |
| [Johnston et al., PNAS 2024](https://www.pnas.org/doi/10.1073/pnas.2400439121) | TrpB libraries A–I and the four-site library | Author-normalized growth fitness |

The 2022 GFP observations come from the final publisher's source data 3, version
3. The author repository's earlier table disagrees with it for 23,965 shared
genotypes and has a different inventory. That earlier table is not used as the
2022 target source. The pinned author notebook supplies reference sequences and
coordinate conventions; it is parsed as literal data and never executed. The
2016 observations are the original table preserved in the author archive.

The TrpB tables come from the [author dataset](https://data.caltech.edu/records/h5rah-5z170).
Only the ten measured summary/simplified table pairs are extracted. The separate
imputed table is excluded. Each reported fitness is checked against the
normalized measured replicate values. A–C have one reported growth replicate;
D–I have two growth replicates but one published input-count column; the four-site
table has two of each. These distinctions are preserved in the curation record.
Negative measured fitness remains valid. This endpoint is a growth proxy, not a
kinetic constant. The 397-residue author construct retains its recorded tag.

## Sequence reconstruction and partitions

GFP native indices start at the initiator methionine with index zero; the original
avGFP notation is converted using the author's offset. Every original residue,
position, canonical replacement and duplicate position is checked. The GFP tasks
exclude recorded stop variants and 35 indel genotypes outside the substitution
scope. TrpB positions are checked against the author's one-based reference.

The split is fixed before fitting:

- At most two substitutions: a sequence-hash bucket assigns approximately 75%
  to training and 25% to validation.
- Three or more substitutions: a held-out `future` partition, meaning published
  higher-order mutants, not a prospective experiment.
- A second sequence hash selects at most 600/250/250 training/validation/future
  examples without using their target values. Library I has 241 validation rows.

Each task binds its exact source tables, preparation protocol and selection
records. The catalog checks exact-sequence phase consistency across all tasks.
This establishes exact-sequence separation, not homology independence. The GFP
families and TrpB tasks support future transfer studies; their existence alone
does not demonstrate cross-protein generalization.

## Reproducible commands

From the research repository, using its existing `venus` Python environment:

```sh
PYTHONPATH=src python scripts/rsi/data_preparation/prepare_mutation_landscape_tasks.py \
  --sources research_runs/rsi/20261023_mutation_landscape_sources_v1 \
  --output research_runs/rsi/NEW_mutation_tasks

PYTHONPATH=src python scripts/rsi/run_mutation_landscape_controls.py \
  --catalog research_runs/rsi/NEW_mutation_tasks/catalog.json \
  --output research_runs/rsi/NEW_mutation_controls
```

Outputs must be new directories. Raw source digests are pinned in the preparation
script. Workbook export requires `openpyxl`, already present in the tested
environment. Author code is never imported. The controls use the existing
isolated, network-free numeric worker with NumPy and scikit-learn.

The prepared declarations are in
`research_runs/rsi/20261023_mutation_landscape_tasks_v1/catalog.json`. The merged
catalog at `research_runs/rsi/20261023_protein_task_catalog_v1/catalog.json`
contains 30 tasks from six papers: five solubility, five expression, six
stability and 14 fitness tasks, with 21,995 distinct selected sequences. These
are partial computational prediction reproductions. They do not reproduce the
papers' physical experiments or complete original model-training pipelines.

To schedule them, load an entry's `task.path` as the allocation's full `task`
declaration in the existing [campaign plan](SCIENTIST_MODEL_CAMPAIGNS.md).
Keep the harness/LLM profile and budget explicit. No new model request is issued
by preparing either catalog; the outstanding GLM access failure remains recorded.

## Verified computational controls, 2026-10-23

Two manually specified controls were frozen before fitting: additive categorical
kernel ridge and degree-two categorical kernel ridge. Both use alpha=1 and the
training-target mean; variable positions are identified from training sequences
only. All 28 fits and validation predictions passed fresh isolated re-fitting and
an independent scikit-learn one-hot/kernel reconstruction. Maximum disagreement
was 1.28e-13. Relevant regression tests passed: 76/76.

Additive controls beat the training-mean predictor on 11/14 validation tasks;
degree-two controls do so on 10/14. Poor results, including TrpB B, E and H, are
retained. The controls were not selected or retuned after validation. Higher-order
future labels remain unscored. These results verify numeric execution and expose
useful model-development tasks; they contribute zero autonomous models, zero
model publications and no evidence of recursive improvement.

The immutable control results and numerical runtime records are under
`research_runs/rsi/20261023_mutation_landscape_controls_v1/`. The source-version
comparison and failed adaptation attempts remain under
`research_runs/rsi/20261023_mutation_task_changes_v1/`.
