# Published sequence-assay tasks

The Scientist task adapter now covers variable-length sequence regression with
`generated_sequence.v1`. The fit/predict interface receives complete sequences,
without padding or truncation, and uses the existing isolated numerical worker,
independent acceptance assessment, model lifecycle and fresh-session reuse.
Harness and language-model configuration remain separate from the task.

## Prepared datasets

The first catalog is at
`research_runs/rsi/20261023_published_sequence_tasks_v1/catalog.json` on hl_gpu.
It contains 16 source-bound computational tasks from three papers and 7,539
distinct sequences across their selected partitions:

| Source and task family | Tasks | Training / validation / held-out rows per task |
| --- | ---: | --- |
| Niwa et al. 2009, enzyme solubility and protein yield | 2 | 562 / 202 / 175 |
| Niwa et al. 2012, solubility and yield in four recorded conditions | 8 | 155 / 63 / 57 |
| Tsuboyama et al. 2023, six designed-scaffold topology groups | 6 | 600 / 250 / 250 |

Sources are [eSOL](https://dbarchive.biosciencedbc.jp/data/esol/LATEST/README_e.html),
[W3110 AP009048.1](https://www.ncbi.nlm.nih.gov/nuccore/AP009048.1), and the
[Tsuboyama author data deposit](https://zenodo.org/records/7992926).
Paper DOIs are `10.1073/pnas.0811922106`, `10.1073/pnas.1201380109` and
`10.1038/s41586-023-06328-6`.

These are retrospective prediction tasks derived from published measurements.
Preparing them and fitting a control does not reproduce the original physical
experiments, establish the papers' main conclusions, or publish autonomous models.
The task field `future` means held-out retrospective observations, not newly
acquired experimental measurements.

## Data preparation and split design

The eSOL adapter selects the source's annotated-enzyme category `e`. It joins
JW identifiers to the exact K-12 W3110 reference and requires agreement of ECK
and b-number annotations. Noncanonical sequences, ambiguous mappings and all
members of duplicate identity/sequence classes are excluded. Missing or
nonfinite measurements are omitted independently for each endpoint, without
clipping or imputation. Derived data retain the eSOL CC BY-SA 2.1 Japan license
and required attribution in `licenses.json`.

All eSOL endpoints share one group assignment. The initial 940 measured enzymes
form 895 CD-HIT clusters at 40% identity with 80% coverage of both sequences.
Validation and held-out candidates are screened against all earlier-phase
sequences with CD-HIT-2D, allowing queries longer than references. One additional
sequence is excluded; 939 remain. This is a declared heuristic similarity
screen, not proof of complete homology independence.

The stability adapter reads the author's processed Dataset 2/3 file, selects
quality group G0 and the explicit designed-scaffold groups HHH, HEEH, EHEE,
EEHEE, EEHH and hall, and uses only finite `dG_ML` values. The original article
defines G0 as passing the quality filters; the `_ML` fields already omit the
author's low-quality observations. No natural domains are admitted by this
adapter. All variants of each parent scaffold remain in the same partition:
91 parent groups are assigned to training, 30 to validation and 25 to held-out
evaluation, before per-task row caps. Scaffold separation does not establish
absence of remote homology or structural similarity.

Group assignments use a fixed SHA256 rule with 60/20/20 buckets. Row caps use
sample-identity hashes and do not examine targets. Every task binds its source
references and all three partitions. A catalog-wide audit rejects exact-sequence
leakage between phases, including between different endpoints. Full source
tables and held-out labels remain host-side; only the existing bounded task
capabilities are exposed to the model-development episode.

Rebuild with the admitted source versions into a new directory:

```bash
PYTHONPATH=src python scripts/rsi/prepare_published_sequence_tasks.py \
  --sources /absolute/path/to/source-cache \
  --output /absolute/path/to/new-task-catalog
```

The command verifies fixed input digests and refuses to overwrite prior output.
It requires CD-HIT and CD-HIT-2D on the preparation host. It makes no model calls,
fits no model and publishes no registry entries. Individual `task.json` files
can be supplied to the existing Scientist goal/evolution commands with
`--task-definition`.

## Offline verification and controls

The variable-length interface passed 168 related regression tests, including
the original model interfaces, lifecycle, experience, model library, campaign
and Agent tool integration. Ten additional adapter tests verify source mapping,
quality exclusions, label-independent splits, duplicate handling, source
tampering and cross-task leakage detection.

The fixed manual control uses ridge regression on log sequence length and
20 amino-acid frequencies, with training-only standardization and a predeclared
regularization value of 10. All 16 tasks completed sandbox training, validation,
fresh sandbox refitting and an independent host reconstruction of predictions
and numerical metrics. The control was not tuned on validation data. Held-out
metrics were not evaluated. Results are in
`research_runs/rsi/20261023_published_sequence_controls_v1/report.json`.

Seven controls beat the training-mean predictor on validation; nine do not,
including all six stability tasks. This establishes executable tasks and
numerical consistency, not an autonomous improvement result. Control artifacts
are explicitly marked manually specified and were not registered or counted
toward the requested model collection. Acceptance against the training mean is
a registration gate; scientific comparisons must also include the fixed ridge
control and matched no-evolution runs.

## Remaining experiments

1. Restore actual model access before starting Scientist allocations. The last
   authorized GLM-5.2 probe on 23 October returned HTTP 403. No silent provider,
   credential or model substitution is performed.
2. Run bounded, matched Scientist allocations across the existing harness
   wrappers with identical tasks and budgets, retaining both successful and
   failed generations. Test the native Codex wrapper separately before pooling
   evidence across harnesses.
3. Compare fixed controls, model development without recurrence, and recurrent
   development. Measure improvement, cost and fresh-session reuse; keep held-out
   evaluation outside development feedback and do not select tasks by control
   success.
4. Test transfer to distinct held-out proteins/scaffolds with donor source and
   experience isolation. The six topology tasks and shared eSOL endpoints are
   correlated tasks, not 16 independent paper replications.
5. Expand other stages using independently sourced, justified tasks. Seeds,
   renamed copies, attempts and repeated publications do not count as new
   scientific models. The catalog currently adds three workflow stages and
   zero autonomous models.
