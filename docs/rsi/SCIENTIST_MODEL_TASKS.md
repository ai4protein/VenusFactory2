# Scientist model tasks

The Scientist model-development and model-evolution commands accept a host-admitted task definition with `--task-definition /absolute/path/task.json`. Task selection is independent of the existing harness and model profile. Each episode retains the four-role Scientist workflow, bounded development attempts, independent numerical assessment, registration and fresh-session reuse.

A task declares schema 1, a stable `id`, workflow `stage`, trusted `plugin_id`, scientific `scope`, paper title/DOI, source `references`, three `data` partitions and `minimum_skill`. Every reference and partition has an absolute `path` and SHA256. Training, validation and future partitions must contain distinct sample identifiers and exact sequences. The definition is bound into the goal admission and revalidated on reopening. This is exact-identity separation; homologous proteins require separately justified grouping.

Supported interfaces are `generated_python.v1`, `generated_sequence.v1`, `generated_features.v1`, `generated_classifier.v1` and `generated_classifier.large.v1`. Their existing resource and numerical contracts apply. Feature representations retain their encoder identity and dimensions. Prediction inputs exclude labels. Acceptance thresholds are fixed before execution and the first evolution generation honors that threshold; later generations must improve on the preceding accepted generation.

```bash
PYTHONPATH=src python scripts/rsi/run_science_evolution.py create \
  --output research_runs/rsi/unique_scientist_goal \
  --profile /absolute/path/scientist_profile.json \
  --task-definition /absolute/path/task.json \
  --objective 'Develop, validate and iteratively improve this protein model' \
  --generations 2 --max-cycles 8 --model-calls 192 --tool-calls 96 --wall-seconds 7200
```

Use the existing run, status, pause and resume actions for this goal. Model credentials remain in the existing provider configuration and are supplied only for execution. Source and runtime changes require a new allocation; the prepared examples use frozen repository copies.

## Current validation

On 23 October 2026, 101 related tests passed, including actual isolated fit/assessment/reuse on synthetic data, all four task interfaces, rejection of overlapping partitions and changed sources, feature-input label exclusion, and continuity of the initial selection threshold. These are software tests, not additional published protein models.

The real ParPgb task uses the admitted ALDE author source and the existing 216/90/90 historical cohorts. Its definition is in `research_runs/rsi/20261023_scientist_task_admission_v1/task.json`. Separate native Kimi and DeepSeek Scientist allocations each request two generations and preserve 706 frozen source files. They are prepared but have made zero model calls. The single GLM-5.2 access probe on this date returned HTTP 403, so neither allocation has started.

## Scaling to the requested model collection

The requested target is more than 1,000 validated small protein-engineering models spanning relevant modelling stages. This task interface removes the fixed-case restriction in the Scientist CLI. It does not by itself provide 1,000 datasets or publish 1,000 models. The persistent campaign runner and shared model library now select source-admitted tasks, reserve aggregate budgets, execute and recover goals, deduplicate publications and expose models to local Agent conversations. Actual scientific execution and stage coverage remain separate requirements. Catalogue entries, attempts, failed models, repeated seeds and renamed registrations must remain distinct from successful model publication counts.

The remaining study requires actual GLM execution across broader tasks and matched recurrence/transfer experiments. The study title is Recursive Self-improvement AI Agents for Protein Engineering. Methods and results should describe the scientific design and supported findings; detailed implementation histories remain in research records.


## Variable-length regression

`generated_sequence.v1` accepts intact proteins of different lengths. Its scope
declares `protein`, `assay`, `target`, `unit`, `minimum_sequence_length` and
`maximum_sequence_length` (integer bounds within 1..2048). It uses the existing
fit/predict source ABI, numeric sandbox, 10..600 training-record budget and
independent numerical acceptance. Each prediction batch admits at most 5000
records. There is no automatic sequence padding, truncation or alignment.

This plugin supports independent models for sequence-level measurements such
as solubility and expression. It does not reinterpret class indices as numeric
targets. Existing aligned-sequence and feature interfaces retain their source
hashes. The new scope is also supported by the bounded experience projection,
Scientist MCP task inspection, task admission, evolution and Agent inference.


## Published sequence-assay task catalog

The source-bound catalog now adds 16 tasks from three published papers covering
solubility, expression and designed-scaffold stability. See
[Published sequence-assay tasks](PUBLISHED_SEQUENCE_TASKS.md) for source mapping,
shared partitions, fixed control results and remaining Scientist experiments.
The related interface regression has 168 passing tests and the new data adapters
have 10 passing tests. Task preparation and manual controls publish no autonomous models.


## Published mutation landscapes

Four GFP families and ten measured TrpB landscapes are available through the same task contract. See [PUBLISHED_MUTATION_TASKS.md](PUBLISHED_MUTATION_TASKS.md) for source versions, mutation-distance partitions and fixed numerical controls. The unified 30-task catalog covers six papers; task admission does not count as autonomous model development.


## Transferring verified implementation experience

The same task admission supports optional external model-library experience through `--experience-selection`. This context is distinct from the within-goal evolution history and is available only during development. See [Cross-task model experience](LIBRARY_EXPERIENCE.md). The current integration passes 205 related regression tests, including the real capability builder for `generated_sequence.v1`.
