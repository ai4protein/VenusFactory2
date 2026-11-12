"""Masked multilabel probabilities, shared by development and independent Review.

Each label is a binary outcome, not a mutually exclusive class. Null targets are
unobserved and never converted to negatives. Metrics expose aggregates only.
"""

from __future__ import annotations

import json
import math
from itertools import groupby
from statistics import fmean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def label_names(labels):
    require(
        isinstance(labels, list)
        and 2 <= len(labels) <= 64
        and all(isinstance(label, str) and label.strip() == label and 0 < len(label) <= 100 for label in labels),
        "Multilabel tasks require 2..64 distinct, nonempty label names",
    )
    require(len(labels) == len(set(labels)), "Duplicate multilabel name")
    return labels


def validate_target(value, labels):
    require(
        isinstance(value, dict) and set(value) == set(labels),
        "Keep the same multilabel target keys; use null for unobserved labels",
    )
    require(
        all(item is None or (type(item) in (int, float) and item in (0, 1)) for item in value.values()),
        "Multilabel targets are binary 0/1 or null, never class strings or probabilities",
    )
    require(
        any(item is not None for item in value.values()),
        "A labeled row needs at least one observed label; put unlabeled inputs in the unlabeled pool",
    )


def parse_target(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate label in target JSON")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Multilabel target cells must be JSON objects with unique labels and 0/1/null values") from exc
    require(isinstance(value, dict), "Multilabel target cells must be JSON objects")
    labels = label_names(sorted(value))
    validate_target(value, labels)
    return value


def validate_prediction(value, labels):
    label_names(labels)
    require(
        isinstance(value, dict) and set(value) == set(labels),
        "Return one probability for every declared label, including unobserved evaluation labels",
    )
    require(
        all(type(item) in (int, float) and math.isfinite(item) and 0 <= item <= 1 for item in value.values()),
        "Multilabel predictions must be finite probabilities in [0, 1]",
    )


def training_prevalence(train):
    require(bool(train) and isinstance(train[0], dict), "Provide multilabel training targets")
    labels = label_names(sorted(train[0]))
    for row in train:
        validate_target(row, labels)
    result = {}
    for label in labels:
        observed = [row[label] for row in train if row[label] is not None]
        require(bool(observed), "Each declared label needs at least one observed training target")
        result[label] = fmean(observed)
    return result


def average_precision(pairs):
    """Noninterpolated AP with equal-score ties; undefined without both classes."""
    positive = sum(y for _, y in pairs)
    if not pairs or positive == 0 or positive == len(pairs):
        return None
    tp = count = area = 0.0
    for _, tied in groupby(sorted(pairs, reverse=True), key=lambda item: item[0]):
        group = list(tied)
        newly_positive = sum(y for _, y in group)
        tp += newly_positive
        count += len(group)
        area += newly_positive / positive * tp / count
    return area


def binary_scores(pairs):
    if not pairs:
        return {"observed": 0, "positive": 0, "brier": None, "average_precision": None, "f1": None}
    tp = sum(p >= 0.5 and y == 1 for p, y in pairs)
    fp = sum(p >= 0.5 and y == 0 for p, y in pairs)
    fn = sum(p < 0.5 and y == 1 for p, y in pairs)
    denominator = 2 * tp + fp + fn
    return {
        "observed": len(pairs),
        "positive": sum(y for _, y in pairs),
        "brier": fmean((p - y) ** 2 for p, y in pairs),
        "average_precision": average_precision(pairs),
        "f1": 2 * tp / denominator if denominator else 0.0,
    }


def metric_details(predictions, targets, labels):
    by_label = {
        label: [(p[label], y[label]) for p, y in zip(predictions, targets) if y[label] is not None] for label in labels
    }
    per_label = {label: binary_scores(pairs) for label, pairs in by_label.items()}
    observed = [item for item in per_label.values() if item["observed"]]
    aps = [item["average_precision"] for item in observed if item["average_precision"] is not None]
    pooled = binary_scores([pair for pairs in by_label.values() for pair in pairs])
    require(bool(observed), "Evaluation requires observed labels")
    return {
        "macro_brier": fmean(item["brier"] for item in observed),
        "micro_brier": pooled["brier"],
        "macro_average_precision": fmean(aps) if aps else None,
        "micro_average_precision": pooled["average_precision"],
        "macro_f1": fmean(item["f1"] for item in observed),
        "micro_f1": pooled["f1"],
        "observed_label_count": len(observed),
        "ap_label_count": len(aps),
        "observed_cells": pooled["observed"],
        "unobserved_cells": len(targets) * len(labels) - pooled["observed"],
        "per_label": per_label,
        "threshold": 0.5,
        "f1_zero_division": 0,
        "average_precision_scope": "Noninterpolated AP; labels lacking positives or negatives are undefined and excluded from macro AP.",
        "masking": "Only observed label cells contribute; each observed label has equal weight in macro metrics.",
    }


def masked_scores(predictions, targets, train, *, query_rows=None):
    require(len(predictions) == len(targets) and bool(targets), "Prediction count differs from query count")
    reference = training_prevalence(train)
    labels = list(reference)
    for prediction, target in zip(predictions, targets):
        validate_prediction(prediction, labels)
        validate_target(target, labels)
    details = metric_details(predictions, targets, labels)
    baseline = metric_details([reference] * len(targets), targets, labels)
    result = {
        "metric": "macro_brier",
        "direction": "minimize",
        "score": details["macro_brier"],
        "reference": baseline["macro_brier"],
        "reference_method": "training_label_prevalence",
        "multilabel": {"labels": labels, **details, "reference": baseline},
    }
    if query_rows is not None:
        result["episodes"] = episode_scores(predictions, targets, query_rows, reference)
    return result


def diagnostics(predictions, train, *, split):
    require(split in {"validation", "test"} and bool(predictions), "Provide predictions and an evaluated split")
    reference = training_prevalence(train)
    labels = list(reference)
    for row in predictions:
        validate_prediction(row, labels)

    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

    constants = [
        label
        for label in labels
        if close(min(row[label] for row in predictions), max(row[label] for row in predictions))
    ]
    matches = [label for label in labels if all(close(row[label], reference[label]) for row in predictions)]
    constant, reference_match = len(constants) == len(labels), len(matches) == len(labels)
    warnings = (["constant_predictions"] if constant else []) + (
        ["matches_training_reference"] if reference_match else []
    )
    return {
        "schema": "prediction_diagnostics.v1",
        "split": split,
        "prediction_count": len(predictions),
        "distinct_predictions": len({tuple(row[label] for label in labels) for row in predictions}),
        "constant_predictions": constant,
        "matches_training_reference": reference_match,
        "constant_labels": constants,
        "reference_matching_labels": matches,
        "reference_method": "training_label_prevalence",
        "warnings": warnings,
        "comparison_tolerance": {"relative": 1e-9, "absolute": 1e-12},
    }


def episode_scores(predictions, targets, rows, training_reference):
    """Task-balanced masked Brier with a supplied-support-only adaptation control."""
    require(isinstance(rows, list) and len(rows) == len(predictions), "Episode queries must align with predictions")
    labels = list(training_reference)
    groups = {}
    for row, prediction, target in zip(rows, predictions, targets):
        require(
            isinstance(row, dict)
            and isinstance(row.get("task_id"), str)
            and row["task_id"]
            and isinstance(row.get("inputs"), dict)
            and row.get("target") == target,
            "Episode task identity or query target does not match",
        )
        support = row["inputs"].get("support")
        require(
            isinstance(support, list) and bool(support) and all(isinstance(item, dict) for item in support),
            "Episode queries require frozen support observations",
        )
        for item in support:
            validate_target(item.get("target"), labels)
        identity = json.dumps(support, sort_keys=True, allow_nan=False)
        if row["task_id"] not in groups:
            prior = dict(training_reference)
            fallback = []
            for label in labels:
                values = [item["target"][label] for item in support if item["target"][label] is not None]
                if values:
                    prior[label] = fmean(values)
                else:
                    fallback.append(label)
            groups[row["task_id"]] = {
                "identity": identity,
                "prior": prior,
                "fallback": fallback,
                "predictions": [],
                "targets": [],
            }
        group = groups[row["task_id"]]
        require(group["identity"] == identity, "Support observations changed within a task")
        group["predictions"].append(prediction)
        group["targets"].append(target)

    def aggregate(use_reference=False):
        metrics, pooled = [], []
        for group in groups.values():
            values = [group["prior"]] * len(group["targets"]) if use_reference else group["predictions"]
            metrics.append(metric_details(values, group["targets"], labels)["macro_brier"])
            pooled.extend(values)
        result = {
            "macro_score": fmean(metrics),
            "pooled_score": metric_details(pooled, [y for group in groups.values() for y in group["targets"]], labels)[
                "macro_brier"
            ],
        }
        if use_reference:
            result["matching_prediction_tasks"] = sum(
                all(
                    all(
                        math.isclose(row[label], group["prior"][label], rel_tol=1e-9, abs_tol=1e-12) for label in labels
                    )
                    for row in group["predictions"]
                )
                for group in groups.values()
            )
        return result

    return {
        "schema": "venusfactory.episode-scores.v1",
        "task_count": len(groups),
        "query_count": len(rows),
        "metric": "macro_brier",
        "weighting": "one equal weight per task",
        "score": aggregate(),
        "references": {"support_label_prevalence": aggregate(True)},
        "training_fallback_label_tasks": sum(len(group["fallback"]) for group in groups.values()),
        "constant_within_task_count": sum(
            all(
                all(
                    math.isclose(row[label], group["predictions"][0][label], rel_tol=1e-9, abs_tol=1e-12)
                    for label in labels
                )
                for row in group["predictions"]
            )
            for group in groups.values()
        ),
        "reference_scope": "Same-task supplied support only; labels unobserved in support use the training-only prevalence. No query-target fitting.",
    }
