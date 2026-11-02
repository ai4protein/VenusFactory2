<!-- Compatibility view; edit en/evaluation.md -->

# Evaluate a model

## Use independent data

Provide labeled sequences that were not used to fit or select the model. Match the training label definitions, measurement units and preprocessing. Preserve protein-family or sequence-group splits when generalization across proteins is the question.

## Run an evaluation

1. Open custom model Evaluation from Model Plaza.
2. Select the checkpoint and its matching configuration, then supply the evaluation data and required columns.
3. Choose metrics appropriate to the task and submit a queued job.
4. Review the metrics and per-sample predictions together. Inspect failed rows, sample counts and label balance.

For classification, consider imbalance and metrics such as MCC, precision/recall, F1 and AUROC when applicable. For regression, inspect error and rank/correlation metrics as well as the range of measured values. Metrics on different splits or label scales are not directly comparable.

## Compare methods

Use the same train/validation/test split, target definition and evaluation procedure for baselines and candidate models. A simple method is a valid result. Report uncertainty or variability when the sample size permits, and disclose missing cases rather than dropping them silently.

## Share a capability

Evaluation alone does not register a public model. Scientist publication also requires interface tests, reproducibility evidence, scope and independent review. Keep the model artifact, configuration, split description and evaluation outputs together. See [Scientist](./en/scientist.md).
