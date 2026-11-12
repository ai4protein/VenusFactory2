"""Shared partition validation and scoring for Scientist development and Review.

This module is pure: it owns no workers, workspaces or admission decisions.
Validation and final test use the same formulas on their respective queries.
"""
from __future__ import annotations

import json
import math

from .contracts import digest
from .research_multilabel import label_names, validate_target, training_prevalence, masked_scores

ROW_LIMITS = {"train": (5, 20000), "validation": (2, 512), "test": (3, 512)}
EPISODE_SCHEMA = "venusfactory.support-query.v1"
MAX_EVALUATION_BYTES = 20 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bounded_encoding(data):
    total = 0
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False)
    for text in encoder.iterencode(data):
        chunk = text.encode("utf-8")
        total += len(chunk)
        _require(total <= MAX_EVALUATION_BYTES,
                 "Prepared evaluation exceeds 20 MiB; reduce the cohort or support_size before exposing data")
        yield chunk


def evaluation_bytes(data):
    """Serialize a bounded table without first allocating its full expansion."""
    output = bytearray()
    for chunk in _bounded_encoding(data):
        output.extend(chunk)
    return bytes(output)


def evaluation_digest(data):
    # A final metric claim is checked by Review, never part of the data identity.
    return digest({k: v for k, v in data.items() if k != "claimed_metric"})


def _episode_records(data):
    """Validate explicit support/query bindings, then reuse ordinary split checks."""
    preparation = data.get("preparation", {})
    episode = preparation.get("episode", {}) if isinstance(preparation, dict) else {}
    _require(isinstance(episode, dict) and episode.get("schema") == EPISODE_SCHEMA
             and episode.get("support_key") == "support", "Unknown support/query contract")
    size = episode.get("support_size")
    _require(type(size) is int and 1 <= size <= 64, "support_size must be 1..64")
    support = data["support"]
    _require(isinstance(support, dict) and set(support) == set(ROW_LIMITS),
             "Support observations require train, validation and test partitions")
    records = {}
    total = 0
    for split in ROW_LIMITS:
        rows = support[split]
        _require(isinstance(rows, list) and len(rows) <= 21024, "Support observations must be a bounded list")
        total += len(rows)+len(data["splits"][split])
        _require(total <= 21024, "At most 21,024 source observations are supported")
        tasks = {}
        for row in rows:
            _require(isinstance(row, dict) and isinstance(row.get("task_id"), str)
                     and row["task_id"] and isinstance(row.get("inputs"), dict)
                     and "support" not in row["inputs"] and "id" in row and "target" in row,
                     "Support rows require task_id, id, inputs and target")
            tasks.setdefault(row["task_id"], []).append(row)
        _require(all(len(values) == size for values in tasks.values()),
                 "Every task must use the frozen support_size")
        task_queries = set()
        queries = []
        for row in data["splits"][split]:
            _require(isinstance(row, dict) and isinstance(row.get("inputs"), dict)
                     and isinstance(row.get("task_id"), str) and row["task_id"] in tasks,
                     "Every query requires support from its own task")
            task_queries.add(row["task_id"])
            values = tasks[row["task_id"]]
            expected = [{"id": item["id"], "inputs": item["inputs"], "target": item["target"]} for item in values]
            _require(row["inputs"].get("support") == expected,
                     "Query support differs from the frozen task observations")
            inputs = {key: value for key, value in row["inputs"].items() if key != "support"}
            fingerprints = {digest(item["inputs"]) for item in values}
            _require(len(fingerprints) == size and digest(inputs) not in fingerprints,
                     "Support and query inputs overlap within a task")
            queries.append({**row, "inputs": inputs})
        _require(task_queries == set(tasks), "Support without query observations is not an evaluation task")
        records[split] = queries+rows
    return records


def validate_data(data, strategy):
    _require(isinstance(data, dict), "Evaluation data must be a JSON object")
    _require(data.get("task") in {"regression", "classification", "multilabel"},
             "Declare regression, classification or multilabel")
    _require(all(isinstance(data.get(k), str) and data[k].strip()
                 for k in ["provenance", "grouping"]),
             "Declare data provenance and grouping")
    splits = data.get("splits")
    _require(isinstance(splits, dict), "Evaluation splits must be a JSON object")
    for name, (minimum, maximum) in ROW_LIMITS.items():
        rows = splits.get(name)
        count = len(rows) if isinstance(rows, list) else "not a list"
        _require(isinstance(rows, list) and minimum <= len(rows) <= maximum,
                 f"Provide bounded train/validation/test splits: {name} requires {minimum}..{maximum} rows; got {count}")
    preparation = data.get("preparation", {})
    _require(not isinstance(preparation, dict) or "episode" not in preparation or "support" in data,
             "Prepared task episodes require their frozen support observations")
    if "support" in data:
        # Bound repeated support lists before scanning their expanded contents.
        for _ in _bounded_encoding(data):
            pass
    records = _episode_records(data) if "support" in data else splits
    labels = label_names(data.get("labels")) if data["task"] == "multilabel" else None
    needs_tasks = strategy == "meta_learning" or "support" in data
    seen_ids = set()
    seen_groups = set()
    seen_inputs = set()
    task_sets = []
    for name in ROW_LIMITS:
        rows = records[name]
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
            if labels is not None:
                validate_target(target, labels)
            else:
                _require(type(target) in (int, float, str)
                         and (isinstance(target, str) or math.isfinite(target)),
                         "Provide finite targets")
            if data["task"] == "regression":
                _require(type(target) in (int, float), "Regression requires numeric targets")
            _require(row["id"] not in ids, "Duplicate sample ID")
            ids.add(row["id"])
            groups.add(row["group"])
            inputs.add(digest(row["inputs"]))
            if needs_tasks:
                _require(isinstance(row.get("task_id"), str) and row["task_id"],
                         "Meta-learning requires task_id for every row")
                tasks.add(row["task_id"])
        _require(not (ids & seen_ids or groups & seen_groups or inputs & seen_inputs),
                 "Leakage: IDs, groups or inputs overlap splits")
        seen_ids |= ids
        seen_groups |= groups
        seen_inputs |= inputs
        task_sets.append(tasks)
    if needs_tasks:
        _require(len(task_sets[0]) >= 2 and all(task_sets[1:])
                 and not any(task_sets[i] & task_sets[j] for i in range(3) for j in range(i)),
                 "Meta-learning requires multiple training tasks and disjoint validation/test tasks")
    if labels is not None:
        training_prevalence([row["target"] for row in splits["train"]])
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
    if "support" not in data:
        for _ in _bounded_encoding(data):
            pass
    return data


def training_view(data, strategy):
    """The only observation view supplied to host-managed fitting."""
    return {"task": data["task"], "strategy": strategy, "train": data["splits"]["train"],
            "unlabeled": data.get("unlabeled", []),
            **({"labels": data["labels"]} if data["task"] == "multilabel" else {})}


def training_evidence_summary(value, data, strategy):
    """Bind a submitted training input to frozen data without returning rows."""
    expected = training_view(data, strategy)
    _require(digest(value) == digest(expected),
             "Training evidence differs from the selected generation's training-only data")
    fields = list(expected["train"][0]["inputs"]) if expected["train"] else []
    return {"kind": "training_data", "scope": "training_only", "task": expected["task"],
            "training_rows": len(expected["train"]), "unlabeled_rows": len(expected["unlabeled"]),
            "input_fields": fields[:16], "omitted_input_fields": max(0, len(fields)-16),
            "training_view_sha256": digest(expected)}


def prediction_values_digest(predictions):
    """Exact scored-value identity, ignoring numeric spelling and output metadata."""
    def canonical(value):
        if type(value) in (int, float):
            numerator, denominator = (value, 1) if type(value) is int else value.as_integer_ratio()
            return ["number", numerator, denominator]
        if isinstance(value, str):
            return ["label", value]
        if isinstance(value, dict):
            return ["labels", [[name, canonical(value[name])] for name in sorted(value)]]
        raise ValueError("Use validated scalar predictions or multilabel probabilities")
    return digest([canonical(value) for value in predictions])


def scores(task, predictions, targets, train, *, query_rows=None):
    _require(len(predictions) == len(targets) and bool(targets),
             "Prediction count differs from query count")
    if task == "multilabel":
        return masked_scores(predictions, targets, train, query_rows=query_rows)
    _require(task in {"regression", "classification"}, "Unsupported evaluation task")
    if task == "regression":
        _require(all(type(p) in (int, float) and math.isfinite(p) for p in predictions),
                 "Predictions must be finite numbers")
        mean = sum(train) / len(train)
        result = {
            "metric": "mae", "direction": "minimize",
            "score": sum(abs(p-y) for p, y in zip(predictions, targets)) / len(targets),
            "rmse": math.sqrt(sum((p-y)**2 for p, y in zip(predictions, targets)) / len(targets)),
            "reference": sum(abs(mean-y) for y in targets) / len(targets),
            "reference_method": "training_mean",
        }
    else:
        from collections import Counter
        _require(all(type(p) in (int, float, str)
                     and (isinstance(p, str) or math.isfinite(p)) for p in predictions),
                 "Predictions must be finite scalar class labels")
        majority = Counter(train).most_common(1)[0][0]
        result = {
            "metric": "accuracy", "direction": "maximize",
            "score": sum(p == y for p, y in zip(predictions, targets)) / len(targets),
            "reference": sum(majority == y for y in targets) / len(targets),
            "reference_method": "training_majority",
        }
    if query_rows is not None:
        result["episodes"] = _episode_scores(task, predictions, targets, query_rows)
    return result


def _episode_scores(task, predictions, targets, rows):
    """Equal-task metrics and support-only controls; expose no per-query labels."""
    from collections import Counter
    from statistics import fmean, median

    _require(isinstance(rows, list) and len(rows) == len(predictions),
             "Episode queries must align with predictions")
    groups = {}
    allowed = (int, float) if task == "regression" else (int, float, str)
    for row, prediction, target in zip(rows, predictions, targets):
        _require(isinstance(row, dict) and isinstance(row.get("task_id"), str)
                 and bool(row["task_id"]) and isinstance(row.get("inputs"), dict)
                 and row.get("target") == target,
                 "Episode task identity or query target does not match")
        support = row.get("inputs", {}).get("support")
        _require(isinstance(support, list) and bool(support)
                 and all(isinstance(item, dict) for item in support),
                 "Episode queries require frozen support observations")
        values = [item.get("target") for item in support]
        _require(all(type(value) in allowed and
                     (isinstance(value, str) or math.isfinite(value)) for value in values+[target]),
                 "Episode support and query targets must be finite scalar values")
        identity = digest(support)
        if row["task_id"] not in groups:
            references = ({"support_mean": fmean(values), "support_median": median(values)}
                          if task == "regression" else
                          {"support_majority": Counter(values).most_common(1)[0][0]})
            groups[row["task_id"]] = {"support": identity, "references": references,
                                      "predictions": [], "targets": []}
        group = groups[row["task_id"]]
        _require(group["support"] == identity, "Support observations changed within a task")
        group["predictions"].append(prediction)
        group["targets"].append(target)

    def equal(a, b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) if task == "regression" else a == b

    def aggregate(method=None):
        per_task, pooled, matches = [], [], 0
        for group in groups.values():
            values = (group["predictions"] if method is None else
                      [group["references"][method]] * len(group["targets"]))
            measurements = [abs(p-y) if task == "regression" else float(p == y)
                            for p, y in zip(values, group["targets"])]
            per_task.append(fmean(measurements)); pooled.extend(measurements)
            matches += int(all(equal(a, b) for a, b in zip(values, group["predictions"])))
        result = {"macro_score": fmean(per_task), "pooled_score": fmean(pooled)}
        if method is not None:
            result["matching_prediction_tasks"] = matches
        return result

    return {"schema": "venusfactory.episode-scores.v1", "task_count": len(groups),
            "query_count": len(rows), "metric": "mae" if task == "regression" else "accuracy",
            "weighting": "one equal weight per task",
            "score": aggregate(),
            "references": {name: aggregate(name) for name in next(iter(groups.values()))["references"]},
            "constant_within_task_count": sum(all(equal(value, group["predictions"][0])
                                                  for value in group["predictions"])
                                              for group in groups.values()),
            "reference_scope": "Each task's supplied support measurements only; no query-target fitting."}
