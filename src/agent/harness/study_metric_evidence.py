"""Bounded aggregate metrics from caller-verified experiment receipts.

The caller must verify each full experiment receipt before projection. Content
IDs bind this projection; they do not authenticate arbitrary caller-supplied
measurements. No file, authored reference table, query row or target is read.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from itertools import islice

POLICY = "study-metric-evidence.v1"
MAX_EXPERIMENTS = 12
MAX_METRICS = MAX_EXPERIMENTS * 8
DIRECTIONS = {"mae": "minimize", "rmse": "minimize", "accuracy": "maximize", "macro_brier": "minimize"}
REFERENCES = {
    "mae": ("training_mean", ("support_mean", "support_median")),
    "accuracy": ("training_majority", ("support_majority",)),
    "macro_brier": ("training_label_prevalence", ("support_label_prevalence",)),
}
METRIC_FIELDS = frozenset({
    "id", "value", "experiment_id", "receipt_sha256", "evaluation_sha256", "split",
    "metric", "direction", "aggregation", "method", "source_pointer", "cohort", "cohort_id",
})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{" + str(length) + "}", value) is not None


def _count(value):
    return type(value) is int and 0 < value < 2**63


def _number(value, metric):
    if type(value) not in (int, float):
        raise ValueError("invalid_metric_value")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("invalid_metric_value") from exc
    if not math.isfinite(result) or result < 0 or (metric in {"accuracy", "macro_brier"} and result > 1):
        raise ValueError("invalid_metric_value")
    return result


def _cohort(evaluation_sha256, aggregation, query_count, task_count=None):
    counts = {"query_count": query_count}
    if aggregation == "equal_task":
        counts["task_count"] = task_count
    binding = {"evaluation_sha256": evaluation_sha256, "split": "validation", **counts}
    return counts, "cohort_" + _digest(binding)


def _metric(row, value, metric, aggregation, method, pointer, task_count=None):
    cohort, cohort_id = _cohort(row["evaluation_sha256"], aggregation,
                               row["split_counts"]["validation"], task_count)
    value = {
        "value": _number(value, metric),
        "experiment_id": row["id"], "receipt_sha256": row["receipt_sha256"],
        "evaluation_sha256": row["evaluation_sha256"], "split": "validation",
        "metric": metric, "direction": DIRECTIONS[metric], "aggregation": aggregation,
        "method": method, "source_pointer": pointer, "cohort": cohort, "cohort_id": cohort_id,
    }
    return {"id": "metric_" + _digest(value), **value}


def _checked_row(row):
    if not isinstance(row, dict) or not _hex(row.get("id"), 32):
        raise ValueError("invalid_experiment_identity")
    if not all(_hex(row.get(key), 64) for key in ("receipt_sha256", "evaluation_sha256")):
        raise ValueError("missing_receipt_binding")
    if (row.get("score_origin") != "host_validation" or row.get("fresh_process_replay") is not True
            or row.get("test_used_for_feedback") is not False):
        raise ValueError("unverified_validation_origin")
    counts = row.get("split_counts")
    if not isinstance(counts, dict) or not _count(counts.get("validation")):
        raise ValueError("invalid_validation_cohort")
    validation = row.get("validation")
    if (not isinstance(validation, dict) or not isinstance(validation.get("metric"), str)
            or validation["metric"] not in REFERENCES):
        raise ValueError("unsupported_validation_metric")
    if validation.get("direction") != DIRECTIONS[validation["metric"]]:
        raise ValueError("inconsistent_metric_direction")
    return validation


def _episode_group(row, validation):
    episode = validation["episodes"]
    metric = validation["metric"]
    if not isinstance(episode, dict) or episode.get("schema") != "venusfactory.episode-scores.v1":
        raise ValueError("unsupported_episode_schema")
    if episode.get("metric") != metric or episode.get("weighting") != "one equal weight per task":
        raise ValueError("inconsistent_episode_metric")
    query_count, task_count = episode.get("query_count"), episode.get("task_count")
    if (not _count(query_count) or query_count != row["split_counts"]["validation"]
            or not _count(task_count) or task_count > query_count):
        raise ValueError("inconsistent_episode_cohort")
    scores = episode.get("score")
    if not isinstance(scores, dict):
        raise ValueError("invalid_episode_scores")
    pooled = _number(scores.get("pooled_score"), metric)
    if not math.isclose(pooled, _number(validation["score"], metric), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("inconsistent_episode_pooled_score")
    # The top-level score is the canonical pooled item; do not duplicate it.
    macro = _metric(row, scores.get("macro_score"), metric, "equal_task", "model",
                    "/validation/episodes/score/macro_score", task_count)
    return episode, macro


def build_metric_evidence(experiments):
    """Project at most twelve receipt-checked full rows into aggregate-only evidence.

    Missing or inconsistent metadata is reported as a bounded issue, never filled
    from an authored claim. Valid independent metrics remain available.
    """
    rows = list(islice(iter(experiments), MAX_EXPERIMENTS + 1))
    if len(rows) > MAX_EXPERIMENTS:
        raise ValueError("Metric evidence exceeds the bounded experiment history")
    result = {"schema": POLICY, "metrics": [], "issues": []}
    seen = set()
    for row in rows:
        identity = row.get("id") if isinstance(row, dict) and _hex(row.get("id"), 32) else None
        if identity in seen:
            raise ValueError("Duplicate experiment in metric evidence")
        if identity is not None:
            seen.add(identity)

        def issue(code, experiment_id=identity):
            result["issues"].append({"experiment_id": experiment_id, "code": code})

        try:
            validation = _checked_row(row)
            metric = validation["metric"]
            result["metrics"].append(_metric(row, validation.get("score"), metric,
                                             "pooled_query", "model", "/validation/score"))
        except ValueError as exc:
            issue(str(exc))
            continue
        reference_method, support_methods = REFERENCES[metric]
        if "reference" in validation:
            if validation.get("reference_method") != reference_method:
                issue("unsupported_host_reference")
            else:
                try:
                    result["metrics"].append(_metric(row, validation["reference"], metric,
                                                     "pooled_query", reference_method, "/validation/reference"))
                except ValueError as exc:
                    issue(str(exc))
        if metric == "mae" and "rmse" in validation:
            try:
                result["metrics"].append(_metric(row, validation["rmse"], "rmse",
                                                 "pooled_query", "model", "/validation/rmse"))
            except ValueError as exc:
                issue(str(exc))
        if "episodes" not in validation:
            continue
        try:
            episode, macro = _episode_group(row, validation)
            result["metrics"].append(macro)
        except ValueError as exc:
            issue(str(exc))
            continue
        references = episode.get("references")
        if not isinstance(references, dict):
            issue("invalid_episode_references")
            continue
        if any(name not in support_methods for name in references):
            issue("unsupported_episode_reference")
        for method in support_methods:
            if method not in references:
                continue
            values = references[method]
            if not isinstance(values, dict):
                issue("invalid_episode_reference")
                continue
            try:
                reference_metrics = [
                    _metric(row, values.get(key), metric, aggregation, method,
                            "/validation/episodes/references/" + method + "/" + key, episode["task_count"])
                    for key, aggregation in (("macro_score", "equal_task"), ("pooled_score", "pooled_query"))
                ]
                result["metrics"].extend(reference_metrics)
            except ValueError as exc:
                issue(str(exc))
    return result


def _validate_metric(item):
    if not isinstance(item, dict) or set(item) != METRIC_FIELDS:
        raise ValueError("Malformed metric evidence")
    body = {key: value for key, value in item.items() if key != "id"}
    if item["id"] != "metric_" + _digest(body):
        raise ValueError("Metric evidence content binding changed")
    if (not _hex(item["experiment_id"], 32) or not _hex(item["receipt_sha256"], 64)
            or not _hex(item["evaluation_sha256"], 64) or item["split"] != "validation"):
        raise ValueError("Invalid metric provenance")
    metric, aggregation = item["metric"], item["aggregation"]
    if (not isinstance(metric, str) or metric not in DIRECTIONS or item["direction"] != DIRECTIONS[metric]
            or not isinstance(aggregation, str) or aggregation not in {"equal_task", "pooled_query"}):
        raise ValueError("Unknown metric semantics")
    _number(item["value"], metric)
    method = item["method"]
    reference, support = REFERENCES.get(metric, (None, ()))
    if method == "model" and aggregation == "pooled_query":
        pointer = "/validation/rmse" if metric == "rmse" else "/validation/score"
    elif method == "model" and metric != "rmse":
        pointer = "/validation/episodes/score/macro_score"
    elif reference is not None and method == reference and aggregation == "pooled_query":
        pointer = "/validation/reference"
    elif method in support:
        key = "macro_score" if aggregation == "equal_task" else "pooled_score"
        pointer = "/validation/episodes/references/" + method + "/" + key
    else:
        raise ValueError("Unsupported metric method")
    if item["source_pointer"] != pointer:
        raise ValueError("Metric source pointer does not match its semantics")
    counts = item["cohort"]
    keys = {"query_count", "task_count"} if aggregation == "equal_task" else {"query_count"}
    if (not isinstance(counts, dict) or set(counts) != keys or not all(_count(x) for x in counts.values())
            or counts.get("task_count", 1) > counts["query_count"]):
        raise ValueError("Invalid metric cohort")
    _, expected = _cohort(item["evaluation_sha256"], aggregation, counts["query_count"], counts.get("task_count"))
    if item["cohort_id"] != expected:
        raise ValueError("Metric cohort binding changed")


def compare_metrics(evidence, left_id, right_id):
    """Return only a numeric relation between compatible IDs in a host-owned table.

    This does not authenticate an untrusted table, decide scientific usefulness,
    infer significance, or turn a numerical improvement into model approval.
    """
    if not isinstance(evidence, dict) or evidence.get("schema") != POLICY:
        raise ValueError("Unsupported metric evidence schema")
    rows = evidence.get("metrics")
    if not isinstance(rows, list) or len(rows) > MAX_METRICS:
        raise ValueError("Unbounded metric evidence")
    metrics = {}
    for item in rows:
        _validate_metric(item)
        if item["id"] in metrics:
            raise ValueError("Duplicate metric identity")
        metrics[item["id"]] = item
    if not isinstance(left_id, str) or not isinstance(right_id, str) or left_id not in metrics or right_id not in metrics:
        raise ValueError("Comparison must cite metric IDs from this evidence table")
    left, right = metrics[left_id], metrics[right_id]
    fields = ("evaluation_sha256", "split", "metric", "direction", "aggregation", "cohort_id")
    if any(left[key] != right[key] for key in fields):
        raise ValueError("Metric comparison requires the same evaluation, metric, direction, aggregation and cohort")
    return {
        "schema": "study-metric-comparison.v1", "left_id": left_id, "right_id": right_id,
        "left_value": left["value"], "right_value": right["value"],
        "relation": "lt" if left["value"] < right["value"] else "gt" if left["value"] > right["value"] else "eq",
        **{key: left[key] for key in fields},
    }
