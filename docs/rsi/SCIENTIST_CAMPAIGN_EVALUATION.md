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
python scripts/rsi/evaluation/evaluate_scientist_campaign.py admit --campaign CAMPAIGN --pairs pairs.json --output STUDY
python scripts/rsi/evaluation/evaluate_scientist_campaign.py evaluate --study STUDY --output SCORES
python scripts/rsi/evaluation/evaluate_scientist_campaign.py audit --study STUDY --evaluation SCORES --output AUDIT
```

`--allocation-id` can select newly queued allocations when admitting a study.
All allocations in the campaign still need to terminate before scoring. Extending
the plan after admission changes the study design and requires a new prospective
admission. Do not execute the evaluator until the registered plan has finished.

The audit independently repeats publication validation, prediction and scoring
into another fresh directory. It compares the complete report, development
barrier and saved predictions; changing a score or rehashing an altered artifact
does not establish agreement. Keep the original study and evidence available.

## Terminal shared campaigns

Shared-program cohorts have a separate descriptive entry point. It includes every
task from every terminal batch, including executions that failed before creating a
goal directory. The current indexed library is discovered automatically; there is
no successful-task selector. Single-goal publication checks remain unchanged.

Admission hashes the training, validation and final dataset files without decoding
their labels. It freezes the full task allocation, coordinator and goal states,
publication reports, canonical origins, evaluator sources and numerical environment.
The supplied library may be a subsequently revalidated local library, but each
entry must still belong to the original goal and allocated scope. Revalidation is
not counted as a new autonomous acquisition.

```sh
python scripts/rsi/evaluation/evaluate_shared_campaign.py admit --campaign CAMPAIGN --library LIBRARY --output STUDY
python scripts/rsi/evaluation/evaluate_shared_campaign.py evaluate --study STUDY --output SCORES
python scripts/rsi/evaluation/evaluate_shared_campaign.py evaluate --study STUDY --output SCORES --resume
python scripts/rsi/evaluation/evaluate_shared_campaign.py audit --study STUDY --evaluation SCORES --output REPLAY
```

Scoring begins only after verifying the frozen study. Native publication checks
and training replay precede prediction; prediction inputs exclude labels. This
adapter currently supports measured context-response tasks. Every allocated task
also receives fixed training-only mean, ridge, random-forest and SVR controls, plus
applicable thermal-response controls. A separate implementation refits these
controls and compares their predictions before metrics are accepted.

Shared execution costs appear once per batch. Task rows reference that batch and
do not repeat its token or call totals. Original coordinator and goal statuses
are reported separately, even when they disagree. Missing models, failed
predictions and failed evidence checks remain distinct. A partially successful
batch never becomes a completed research goal through evaluation.

Checkpoints bind the study, task order and saved prediction files. Resuming
verifies all frozen evidence and completed checkpoints; it cannot skip a failed
batch or substitute a new model. A full audit recomputes into a fresh directory
and compares reports and artifacts. These retrospective scores describe the
frozen cohort; causal RSI claims require a separately admitted prospective
comparison with matched resources.

## Source ownership and validation

The single-goal evaluator owns paired-study rules. The shared study module owns
cohort freezing and publication bindings; the shared evaluator owns batch
checkpoints and aggregation. Both reuse the same acquisition-evidence reader,
numerical scorer, registry validator and baseline implementations.

Focused tests cover label-free admission, all-failure denominators, changed
sources or assets, missing publications, interrupted scoring, independent control
refits and numerical replay. Synthetic native seams test storage and lifecycle
contracts; their fitted fixtures are not autonomous scientific model evidence.
