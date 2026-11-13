"""Resolve a failed fit only through a later measured attempt on its frozen task."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tokenize

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



def equivalent_fit(development, *, algorithm_sha256, state_sha256, evaluation_sha256, prediction_key):
    """Find an already measured fit, independent of filenames and parent choice."""
    training = development.root / "training"
    for path in sorted(training.glob("*/receipt.json")):
        try:
            row = _receipt(training, path.parent.name)
            if (row.get("status") != "completed"
                    or row.get("algorithm_ast_sha256") != algorithm_sha256
                    or row.get("state_sha256") != state_sha256
                    or row.get("evaluation_sha256") != evaluation_sha256
                    or row.get("prediction_key") != prediction_key):
                continue
            experiment = _verified_replacement(development, row)
            return experiment["id"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


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
        from .research_execution_feedback import detached_request_failure

        request = detached_request_failure(failure, development.root)
        if request is not None:
            remaining.append(request)
            failure = {key: value for key, value in failure.items() if key != "request_failure"}
        identity = failure.get("training_attempt_id")
        if (failure.get("failure_kind") == "request"
                or failure.get("tool", "").rsplit("__", 1)[-1] != "train_research_model" or not identity):
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


def training_snapshot(development):
    """Capture host record identities before an episode, without exposing data."""
    return {
        "attempts": {p.parent.name for p in (development.root / "training").glob("*/receipt.json")},
        "experiments": {row["id"] for row in development.records()},
    }


def _algorithm_identity(development, row):
    if row.get("algorithm_ast_sha256"):
        return "ast:" + row["algorithm_ast_sha256"]
    raw = read_bytes(development.root / "training", row["id"] + "/algorithm.py", max_bytes=200000)
    if hashlib.sha256(raw).hexdigest() != row["algorithm_sha256"]:
        raise ValueError("Training source snapshot changed")
    # Syntax failures have no AST. Token comparison still distinguishes edits
    # while ignoring comments/blank lines, without executing the submitted code.
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(raw.decode("utf-8")).readline):
            if token.type not in {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}:
                tokens.append((token.type, token.string))
    except tokenize.TokenError as error:
        if error.args[0] != "EOF in multi-line statement":
            # An unterminated string can hide its content from tokenization.
            return "source:" + row["algorithm_sha256"]
    except (IndentationError, SyntaxError, UnicodeError):
        return "source:" + row["algorithm_sha256"]
    return "tokens:" + digest(tokens)


def training_progress(development, before, previous=None):
    """Detect unchanged failed fits across three actual implementation cycles.

    A renamed file or comment change is not a new algorithm. Changed source,
    data, seed, lineage, failure or a new measured generation breaks the streak.
    Scientific rejection of successfully measured experiments is not stagnation.
    """
    from .tool_diagnostics import bounded_message

    after = training_snapshot(development)
    attempts = []
    for identity in after["attempts"] - before["attempts"]:
        try:
            attempts.append(_receipt(development.root / "training", identity))
        except (OSError, ValueError, KeyError, TypeError):
            # Missing, unfinished or modified receipts cannot prove repeated work.
            return {"same_failure_cycles": 0, "intervention_required": False}
    attempts.sort(key=lambda row: row["created"])
    report = {
        "fit_attempts": len(attempts),
        "measured_generations": len(after["experiments"] - before["experiments"]),
        "same_failure_cycles": 0,
        "intervention_required": False,
    }
    if not attempts or report["measured_generations"] or any(row["status"] != "failed" for row in attempts):
        return report

    def fingerprint(row):
        error = row.get("error", "").replace(row["source_file"], "<algorithm>")
        check = row.get("source_check", {})
        if row.get("stage") == "preflight" and check.get("valid") is False:
            # Presentation excerpts/carets vary with whitespace and comments.
            # The verified source identity below still binds the actual code.
            error = json.dumps([{key: issue.get(key) for key in ("code", "severity", "message")}
                                for issue in check["issues"]], sort_keys=True)
        # The source reporter adds a line number after the path. Comments and
        # blank lines move it without changing the measured algorithm/failure.
        error = re.sub(r"(?m)^At <algorithm>:[0-9]+:", "At <algorithm>:<n>:", error)
        error = re.sub(r"line [0-9]+(?:, column [0-9]+)?", "line <n>", error)
        return digest({
            "algorithm": _algorithm_identity(development, row),
            "scope": _scope(row), "training": row["training_sha256"], "seed": row["seed"],
            "stage": row.get("stage"), "error_type": row.get("error_type"), "error": error,
        })

    try:
        identities = {fingerprint(row) for row in attempts}
    except (OSError, ValueError, KeyError, TypeError):
        return report
    if len(identities) != 1:
        return report
    signature = identities.pop()
    previous = previous or {}
    repeats = previous.get("same_failure_cycles", 0) + 1 if previous.get("signature") == signature else 1
    latest = attempts[-1]
    report.update(
        signature=signature, same_failure_cycles=repeats, intervention_required=repeats >= 3,
        source_file=latest["source_file"],
        error=bounded_message(latest.get("error", "Fitting failed"), development.root, 600),
        evidence=[{"attempt_id": row["id"], "receipt_sha256": row["receipt_sha256"]} for row in attempts[-3:]],
        instruction="Repeated fitting failures use the same algorithm and inputs. Repair the saved implementation "
                    "from its observed error or change the justified approach; renaming a file does not repair it.",
    )
    return report
