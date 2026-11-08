"""Resolve a failed fit only through a later measured attempt on its frozen task."""
from __future__ import annotations

import hashlib
import json
import math
import re

from .contracts import digest
from utils.scoped_files import open_file, read_bytes

VERSION = "training_attempt.v1"


class TrainingAttemptError(ValueError):
    """A failed host-owned attempt with a stable, non-path reference."""

    def __init__(self, identity, message):
        self.attempt_id = identity
        super().__init__(f"Training attempt {identity} failed: {message}")


def attempt_id(message):
    match = re.search(r"\bTraining attempt ([a-f0-9]{32}) failed:", str(message))
    return match.group(1) if match else None


def _receipt(root, identity):
    if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Invalid training attempt")
    value = json.loads(read_bytes(root, identity+"/receipt.json", max_bytes=1024**2))
    if not isinstance(value, dict):
        raise ValueError("Invalid training receipt")
    if (value.get("id") != identity or value.get("attempt_protocol") != VERSION
            or value.get("receipt_sha256") != digest({k:v for k,v in value.items() if k!="receipt_sha256"})):
        raise ValueError("Training receipt changed or has no attempt binding")
    if (value.get("variant") not in {"baseline", "candidate", "ablation"}
            or not isinstance(value.get("prediction_key"), str)
            or not value["prediction_key"]
            or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("evaluation_sha256", "")))
            or any(type(value.get(key)) not in (int, float) or not math.isfinite(value[key])
                   for key in ("created", "finished"))):
        raise ValueError("Incomplete training attempt binding")
    source = read_bytes(root, identity+"/algorithm.py", max_bytes=200000)
    if hashlib.sha256(source).hexdigest() != value.get("algorithm_sha256"):
        raise ValueError("Training source snapshot changed")
    return value


def _scope(row):
    return (row.get("evaluation_sha256"), row.get("parent_experiment_id"),
            row.get("variant"), row.get("prediction_key"))


def _verified_replacement(development, row):
    experiment, _, _ = development.evaluation_snapshot(row["experiment_id"])
    if (experiment.get("evaluation_sha256") != row.get("evaluation_sha256")
            or experiment.get("parent") != row.get("parent_experiment_id")
            or experiment.get("variant") != row.get("variant")
            or experiment.get("prediction_key") != row.get("prediction_key")
            or experiment.get("source") != row.get("source")
            or set(experiment.get("files", {})) != set(row.get("runtime_files", []))
            or experiment.get("score_origin") != "host_validation"
            or experiment.get("fresh_process_replay") is not True
            or experiment.get("test_used_for_feedback") is not False):
        raise ValueError("Replacement lacks matching validation evidence")
    root = development.root/"experiments"/experiment["id"]/"runtime"
    for name, expected in experiment["files"].items():
        sha = hashlib.sha256()
        with open_file(root, name, max_bytes=200*1024**2) as stream:
            for block in iter(lambda:stream.read(1024**2), b""):
                sha.update(block)
        if sha.hexdigest() != expected:
            raise ValueError("Replacement runtime changed")
    return experiment


def resolve_attempts(feedback, development):
    """Host receipts, never a model's success claim, link a measured replacement."""
    from .research_development import immutable_copy

    pending = list(feedback.get("pending", []))
    resolved = list(feedback.get("superseded", []))
    training = development.root/"training"
    completed = []
    for path in training.glob("*/receipt.json"):
        try:
            row = _receipt(training, path.parent.name)
            if row.get("status") == "completed":
                completed.append(row)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    completed.sort(key=lambda row:row["created"], reverse=True)
    remaining = []
    for failure in pending:
        identity = failure.get("training_attempt_id")
        if failure.get("tool", "").rsplit("__", 1)[-1] != "train_research_model" or not identity:
            remaining.append(failure)
            continue
        try:
            failed = _receipt(training, identity)
            if failed.get("status") != "failed" or failed.get("feedback_key") != failure.get("key"):
                raise ValueError("Failure does not match the host attempt")
        except (OSError, ValueError, KeyError, TypeError):
            remaining.append(failure)
            continue
        replacement = None
        for candidate in completed:
            if (_scope(candidate) != _scope(failed)
                    or candidate["created"] < failed["finished"]):
                continue
            try:
                experiment = _verified_replacement(development, candidate)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            replacement = {
                "failure_key": failure["key"], "failed_attempt_id": identity,
                "replacement_attempt_id": candidate["id"],
                "replacement_experiment_id": experiment["id"],
                "failed_receipt_sha256": failed["receipt_sha256"],
                "replacement_receipt_sha256": candidate["receipt_sha256"],
                "experiment_receipt_sha256": experiment["receipt_sha256"],
                "reason": "Later fit and validation on the same frozen task and lineage",
            }
            name = identity+"-"+candidate["id"]+".json"
            immutable_copy(development.root, "attempt_resolutions/"+name,
                           json.dumps(replacement, sort_keys=True).encode())
            break
        if replacement:
            if replacement not in resolved:
                resolved.append(replacement)
        else:
            remaining.append(failure)
    result = {"pending": remaining,
              "errors": [{"tool":row["tool"], "error":row["error"]} for row in remaining]}
    if resolved:
        result["superseded"] = resolved[-64:]
    return result
