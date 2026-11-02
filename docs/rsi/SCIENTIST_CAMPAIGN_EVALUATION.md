# Independent Scientist campaign evaluation

`evaluate_scientist_campaign.py` connects the persistent campaign scheduler to
post-development evaluation. It reuses the existing native publication audit,
training replay, model registry and isolated prediction interfaces. Independent
host code calculates prediction metrics from the admitted observations.

## Prospective study and development barrier

Admit a study before its selected allocations have started. The admission binds
the campaign definition and plan versions, task sources and partitions, selected
allocations, comparison pairs, evaluator source and prediction implementations.
Keep a frozen evaluator repository for a study that must survive later edits.

Evaluation requires a quiescent campaign with no unresolved allocation and all
campaign allocations terminal. A queued, running, paused or model-access-waiting
allocation prevents scoring. A campaign can be `needs_tasks` after exhausting its
plan; the broader model-count objective need not be completed to evaluate that
finished plan. The evaluator freezes terminal state and goal-file identities
before producing future metrics and rechecks them afterwards.

The scorer writes to a new directory outside the campaign. It leaves goal
histories and the model library unchanged. It cannot train a replacement model,
make a model-service request, increase a model count or publish an entry.

## Models and outcomes

Every recorded successful publication is checked against its immutable attempt,
supporting origin, fitted packet and the receiving task. Its own native execution,
training, assessment and fresh-session reuse are independently revalidated.
Canonical registry entries are also resolved. When the library deduplicates a
program key, the scorer evaluates the fitted artifact from this allocation;
it does not substitute another allocation's canonical weights.

All selected allocations remain in the feasibility denominator. Missing models
retain null prediction scores. Invalid publication evidence fails the evaluation;
it is not converted into an ordinary missing model. Prediction failures and
publication failures remain distinct. A completed evaluation report means the
evaluation finished, not that its scientific allocations succeeded.

Published generations are evaluated in their recorded order. The final published
generation is the final-model outcome regardless of future scores. Where both
exist, the report also gives first-to-final future skill change. Within-goal
development-selection gains alone do not establish improvement on unseen data.

## Metrics and resources

Regression scores include MSE, RMSE, MAE, MSE skill against the training-mean
predictor and Spearman correlation. Classification scores include Brier skill
against the training class prior, log loss, accuracy and per-class recall.
Training baselines are recomputed from admitted training labels, not taken from
the fitted model's self-report. Prediction workers receive inputs without labels.
Constant-rank correlations and degenerate skill baselines are explicitly null.

Physical request records include retries. Observed calls, token counts and host
execution time remain separate from charged reservations. Missing usage and
interrupted executions retain incomplete-accounting flags; lower bounds are not
presented as full costs. Equal call limits do not imply equal token consumption.

Explicit paired comparisons currently support library-experience ablations.
Each pair must match task, donor IDs, model and harness profile, lineage length
and resource limits. Scores are differenced only when both values exist; missing
scores have no invented failure floor. The report contains descriptive paired
results and does not treat individual protein observations as independent agent
runs.

## Commands

Run with the Venus Python and `PYTHONPATH=src` from the admitted frozen evaluator.
The comparison file is a JSON list, for example:

```json
[
  {
    "id": "source_minus_none",
    "left": "task_source_kimi",
    "right": "task_none_kimi",
    "factor": "library_experience"
  }
]
```

```sh
python scripts/rsi/evaluate_scientist_campaign.py admit --campaign CAMPAIGN --pairs pairs.json --output STUDY
python scripts/rsi/evaluate_scientist_campaign.py evaluate --study STUDY --output SCORES
python scripts/rsi/evaluate_scientist_campaign.py audit --study STUDY --evaluation SCORES --output AUDIT
```

`--allocation-id` can select newly queued allocations when admitting a study.
All allocations in the campaign still need to terminate before scoring. Extending
the plan after admission changes the study design and requires a new prospective
admission. Do not execute the evaluator until the registered plan has finished.

The audit independently repeats publication validation, prediction and scoring
into another fresh directory. It compares the complete report, development
barrier and saved predictions; changing a score or rehashing an altered artifact
does not establish agreement. Keep the original study and evidence available.

## Current validation

On 23 October 2026, 171 focused integration tests passed across the evaluator,
campaign scheduling, model-library publication, cross-task experience and the
sequence/classification interfaces. The new tests include real isolated numerical
prediction, all-failure cohorts, held-out scoring barriers, deduplicated fitted
artifacts, missing token usage and altered-result rejection. Synthetic publication
fixtures contribute zero scientific models.

The GFP transfer pilot remains a 24-allocation prospective study. Its externally
generated models and efficacy results are still pending model access. The latest
explicitly authorized single GLM-5.2 access request returned HTTP 403. No
availability probe is an autonomous model or a scientific replicate.
