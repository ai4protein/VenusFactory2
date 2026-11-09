"""Shared partition validation and scoring for Scientist development and Review.

This module is pure: it owns no workers, workspaces or admission decisions.
Validation and final test use the same formulas on their respective queries.
"""
from __future__ import annotations

import json
import math

from .contracts import digest

ROW_LIMITS = {"train": (5, 20000), "validation": (2, 512), "test": (3, 512)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def evaluation_digest(data):
    # A final metric claim is checked by Review, never part of the data identity.
    return digest({k: v for k, v in data.items() if k != "claimed_metric"})


def validate_data(data, strategy):
    _require(isinstance(data, dict), "Evaluation data must be a JSON object")
    _require(data.get("task") in {"regression", "classification"},
             "Declare regression or classification")
    _require(all(isinstance(data.get(k), str) and data[k].strip()
                 for k in ["provenance", "grouping"]),
             "Declare data provenance and grouping")
    splits = data.get("splits")
    _require(isinstance(splits, dict), "Evaluation splits must be a JSON object")
    seen_ids = set()
    seen_groups = set()
    seen_inputs = set()
    task_sets = []
    for name, (minimum, maximum) in ROW_LIMITS.items():
        rows = splits.get(name)
        count = len(rows) if isinstance(rows, list) else "not a list"
        _require(isinstance(rows, list) and minimum <= len(rows) <= maximum,
                 f"Provide bounded train/validation/test splits: {name} requires {minimum}..{maximum} rows; got {count}")
        ids = set()
        groups = set()
        inputs = set()
        tasks = set()
        for row in rows:
            _require(isinstance(row, dict)
                     and all(isinstance(row.get(k), str) and row[k] for k in ["id", "group"])
                     and isinstance(row.get("inputs"), dict),
                     "Each row needs id, group and inputs")
            target = row.get("target")
            _require(type(target) in (int, float, str)
                     and (isinstance(target, str) or math.isfinite(target)),
                     "Provide finite targets")
            if data["task"] == "regression":
                _require(type(target) in (int, float), "Regression requires numeric targets")
            _require(row["id"] not in ids, "Duplicate sample ID")
            ids.add(row["id"])
            groups.add(row["group"])
            inputs.add(digest(row["inputs"]))
            if strategy == "meta_learning":
                _require(isinstance(row.get("task_id"), str) and row["task_id"],
                         "Meta-learning requires task_id for every row")
                tasks.add(row["task_id"])
        _require(not (ids & seen_ids or groups & seen_groups or inputs & seen_inputs),
                 "Leakage: IDs, groups or inputs overlap splits")
        seen_ids |= ids
        seen_groups |= groups
        seen_inputs |= inputs
        task_sets.append(tasks)
    if strategy == "meta_learning":
        _require(len(task_sets[0]) >= 2 and all(task_sets[1:])
                 and not any(task_sets[i] & task_sets[j] for i in range(3) for j in range(i)),
                 "Meta-learning requires multiple training tasks and disjoint validation/test tasks")
    rows = data.get("unlabeled", [])
    _require(isinstance(rows, list) and len(rows) <= 20000,
             "Provide at most 20,000 unlabeled observations as a list")
    if strategy == "semi_supervised":
        _require(bool(rows), "Semi-supervised learning requires genuine unlabeled observations")
    # Any strategy may use these rows during reproduction. Validate them before
    # either fitting or review, even when the baseline is labeled supervised.
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"id", "inputs"}
                 and isinstance(row["id"], str) and row["id"]
                 and isinstance(row["inputs"], dict),
                 "Unlabeled rows contain id and inputs only")
        key = digest(row["inputs"])
        _require(row["id"] not in seen_ids and key not in seen_inputs,
                 "Unlabeled data overlap labeled or held-out observations")
        seen_ids.add(row["id"])
        seen_inputs.add(key)
    json.dumps(data, allow_nan=False)
    return data


def scores(task, predictions, targets, train):
    _require(len(predictions) == len(targets) and bool(targets),
             "Prediction count differs from query count")
    if task == "regression":
        _require(all(type(p) in (int, float) and math.isfinite(p) for p in predictions),
                 "Predictions must be finite numbers")
        mean = sum(train) / len(train)
        return {
            "metric": "mae", "direction": "minimize",
            "score": sum(abs(p-y) for p, y in zip(predictions, targets)) / len(targets),
            "rmse": math.sqrt(sum((p-y)**2 for p, y in zip(predictions, targets)) / len(targets)),
            "reference": sum(abs(mean-y) for y in targets) / len(targets),
            "reference_method": "training_mean",
        }
    from collections import Counter
    _require(all(type(p) in (int, float, str)
                 and (isinstance(p, str) or math.isfinite(p)) for p in predictions),
             "Predictions must be finite scalar class labels")
    majority = Counter(train).most_common(1)[0][0]
    return {
        "metric": "accuracy", "direction": "maximize",
        "score": sum(p == y for p, y in zip(predictions, targets)) / len(targets),
        "reference": sum(majority == y for y in targets) / len(targets),
        "reference_method": "training_majority",
    }
