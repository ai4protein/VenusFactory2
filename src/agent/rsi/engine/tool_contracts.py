"""Trusted operational contracts for generated-model tools.

An empty lookup and negative training feedback are valid operations. These
checks never admit a model: held-out acceptance stays in ModelLifecycle.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from .development_handoff import FIELD as HANDOFF_FIELD
from .development_handoff import description as handoff_description
from .development_handoff import receipt_fields as handoff_receipt_fields
from .experience import LIMITATION, validate_projection
from agent.rsi.datasets.library_source import REFIT_TOOL, validate_source_origin
from .protein_model import digest
from agent.rsi.engine.protein_references import audit_source_references, reference_catalog


def _read(root, name):
    path = Path(root) / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("Declared scientific artifact is missing or is a symlink")
    return json.loads(path.read_text())


def _equal(actual, expected):
    if digest(actual) != digest(expected):
        raise ValueError("Scientific result disagrees with its bound artifact")


def _source_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()


def _durable_ledger_required(capability):
    path = Path(capability) / "generated_tools.py"
    if not path.is_file() or path.is_symlink():
        return False
    tree = ast.parse(path.read_text())
    return any(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 1
        and any(isinstance(target, ast.Name) and target.id == "DEVELOPMENT_LEDGER_SCHEMA" for target in node.targets)
        for node in tree.body
    )


def _ledger(workspace, capability, task, snapshot=None):
    """Audit durable tool evidence without importing or executing the MCP entry."""
    if snapshot is not None and (
        not isinstance(snapshot, str) or len(snapshot) != 64 or any(c not in "0123456789abcdef" for c in snapshot)
    ):
        raise ValueError("Invalid development ledger snapshot")
    state = _read(workspace, "code_ledger_" + snapshot + ".json" if snapshot else ".model_development.json")
    if snapshot and digest(state) != snapshot:
        raise ValueError("Development ledger snapshot changed")
    sources = {}
    for path in sorted(Path(capability).glob("*.py")):
        if path.is_symlink():
            raise ValueError("Symlinked development implementation")
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        set(state) != {"schema", "binding", "attempts", "submission"}
        or type(state["schema"]) is not int
        or state["schema"] != 1
    ):
        raise ValueError("Invalid development ledger schema")
    _equal(state["binding"], {"task_sha256": digest(task), "implementation_sha256": digest(sources)})
    _equal(_read(workspace, "code_ledger_" + digest(state) + ".json"), state)
    attempts = state["attempts"]
    if not isinstance(attempts, list) or len(attempts) > 2:
        raise ValueError("Durable model-development budget differs")
    tested = {}
    for number, record in enumerate(attempts, 1):
        if (
            type(record.get("number")) is not int
            or record["number"] != number
            or record.get("status") not in {"reserved", "completed"}
        ):
            raise ValueError("Invalid durable attempt status or order")
        sha = _source_hash(record["source_code"])
        _equal(sha, record["source_sha256"])
        if "source_origin" in record:
            validate_source_origin(capability, task, record["source_code"], record["source_origin"])
        if record["status"] == "reserved":
            if number != len(attempts):
                raise ValueError("Development continued after unresolved execution")
            continue
        name = f"code_attempt_{number}.json"
        attempt = _read(workspace, name)
        _equal(hashlib.sha256((Path(workspace) / name).read_bytes()).hexdigest(), record.get("artifact_sha256"))
        _equal(attempt.get("source_code"), record["source_code"])
        _equal(attempt.get("rationale"), record["rationale"])
        _equal(attempt.get("source_origin"), record.get("source_origin"))
        if attempt.get("status", "tested_not_published") == "tested_not_published":
            tested[sha] = attempt
        elif attempt.get("status") != "development_failed":
            raise ValueError("Invalid durable feedback status")
    if snapshot is None:
        names = {f"code_attempt_{row['number']}.json" for row in attempts if row["status"] == "completed"}
        possible_pending = {f"code_attempt_{row['number']}.json" for row in attempts if row["status"] == "reserved"}
        actual = {path.name for path in Path(workspace).glob("code_attempt_*.json")}
        if names - actual or actual - names - possible_pending:
            raise ValueError("Durable attempt artifact coverage differs")
    submission = state["submission"]
    if submission is not None:
        if submission.get("status") not in {"reserved", "completed"} or submission.get("source_sha256") not in tested:
            raise ValueError("Durable submission lacks a successful source")
        if submission["status"] == "completed":
            hashes = submission.get("artifact_sha256", {})
            if set(hashes) != {"candidate.json", "source_provenance.json"}:
                raise ValueError("Durable submission artifact coverage differs")
            for name, expected in hashes.items():
                _read(workspace, name)
                _equal(hashlib.sha256((Path(workspace) / name).read_bytes()).hexdigest(), expected)
            model, provenance = _read(workspace, "candidate.json"), _read(workspace, "source_provenance.json")
            sha = submission["source_sha256"]
            _equal(_source_hash(model["recipe"]["source_code"]), sha)
            _equal(provenance.get("training_feedback_attempts"), len(attempts))
            _equal(provenance.get("training_feedback"), tested[sha]["training_feedback"])
            _equal(provenance.get("rationale"), submission["rationale"])
            _equal(provenance.get("source_origin"), tested[sha].get("source_origin"))
            if "source_origin" in tested[sha]:
                _equal(provenance.get("origin"), "admitted_library_source_refit")
    return state, tested


def validate_generated_result(name, raw, *, workspace, capability, _require_durable=False):
    value = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not isinstance(value, dict):
        raise ValueError("Scientific result must be an object")
    value = dict(value)
    if "success" in value and value.pop("success") is not True:
        raise ValueError("Scientific operation reported failure")
    # Canonical serialization also rejects NaN/Infinity in feedback or artifacts.
    digest(value)
    task = _read(capability, "task.json")
    if task.get("schema") == "shared_model_program.v1":
        from .shared_model_contracts import validate_shared_result

        return validate_shared_result(name, value, workspace=workspace, capability=capability)
    handoff = handoff_description(task)
    phase = task["phase"]
    durable = (
        (_require_durable or _durable_ledger_required(capability))
        and phase == "develop"
        and name in {"inspect_model_code_task", "evaluate_model_code", REFIT_TOOL, "submit_model_code"}
    )
    ledger, tested_sources = _ledger(workspace, capability, task) if durable else (None, None)

    def input_rows(filename):
        rows = _read(workspace, filename)
        _equal(digest(rows), task["inputs"][filename])
        return rows

    def packet():
        p = _read(capability, "registered/capability.json")
        _equal(digest(p), task["capability_sha256"])
        _equal(digest(p["model"]), p["model_sha256"])
        _equal(p["model"]["scope"], task["scope"])
        return p

    if name in {"read_model_experience", "read_model_library_experience"}:
        if phase != "develop":
            raise ValueError("Experience unavailable in this phase")
        field = "library_experience" if name == "read_model_library_experience" else "experience"
        if field == "library_experience" and field + "_sha256" not in task:
            raise ValueError("No library experience admitted for this episode")
        expected = {"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION}
        if field + "_sha256" in task:
            expected = _read(capability, field + ".json")
            _equal(digest(expected), task[field + "_sha256"])
        validate_projection(value)
        _equal(value, expected)
    elif name == "discover_generated_models":
        models = []
        if phase == "reuse":
            p = packet()
            models = [
                {
                    "model_id": p["model_id"],
                    "scope": p["model"]["scope"],
                    "source_sha256": p["model"]["source_sha256"],
                    "status": "registered",
                    "assessment": p["assessment"]["metrics"],
                    **(
                        {"source_reference_audit": p["assessment"]["source_reference_audit"]}
                        if "source_reference_audit" in p["assessment"]
                        else {}
                    ),
                }
            ]
        _equal(value, {"models": models})
    elif name == "inspect_model_code_task":
        _equal(value.get(HANDOFF_FIELD), handoff)
        if value.get("phase") != phase or value.get("scope") != task["scope"]:
            raise ValueError("Task metadata mismatch")
        if task.get("plugin_id") == "generated_sequence.v1":
            _equal(
                value.get("sequence_layout"),
                {
                    "kind": "variable_length",
                    "minimum": task["scope"]["minimum_sequence_length"],
                    "maximum": task["scope"]["maximum_sequence_length"],
                },
            )
            _equal(value.get("acceptance_metric"), "mse_skill_vs_training_mean")
        if set(value.get("abi", {})) != {"fit", "predict"} or not value.get("constraints"):
            raise ValueError("Missing model contract")
        if durable:
            observed = value.get("development_state", {})
            snapshot, tested = _ledger(workspace, capability, task, observed.get("ledger_sha256"))
            if len(snapshot["attempts"]) > len(ledger["attempts"]):
                raise ValueError("Development status predates no known ledger state")
            for before, after in zip(snapshot["attempts"], ledger["attempts"]):
                for key in ("number", "source_code", "source_sha256", "rationale"):
                    _equal(before[key], after[key])
                if before["status"] == "completed":
                    _equal(before, after)
            previous_submission = snapshot["submission"]
            if previous_submission:
                if not ledger["submission"]:
                    raise ValueError("Submission disappeared after status observation")
                for key in ("source_sha256", "rationale"):
                    _equal(previous_submission[key], ledger["submission"][key])
                if previous_submission["status"] == "completed":
                    _equal(previous_submission, ledger["submission"])
            _equal(
                observed,
                {
                    "schema": 1,
                    "ledger_sha256": digest(snapshot),
                    "attempts_used": len(snapshot["attempts"]),
                    "attempts_remaining": 2 - len(snapshot["attempts"]),
                    "submittable_sources": sorted(tested),
                    "pending_attempts": [row["number"] for row in snapshot["attempts"] if row["status"] == "reserved"],
                    "submission_status": (snapshot["submission"] or {}).get("status", "not_submitted"),
                },
            )
        if phase == "develop" and task.get("plugin_id") not in {
            "generated_features.v1",
            "generated_context.v1",
            "generated_context.v2",
            "generated_context.v3",
            "generated_assay.v1",
            "generated_antibody.v1",
        }:
            if (Path(capability) / "protein_references.py").is_file() or "residue_reference_catalog" in value:
                _equal(value.get("residue_reference_catalog"), reference_catalog())
        if phase == "develop":
            rows = input_rows("training.json")
            examples = rows[:3]
            if task.get("plugin_id") in {
                "generated_context.v1",
                "generated_context.v2",
                "generated_context.v3",
                "generated_assay.v1",
                "generated_antibody.v1",
            }:
                from .context_interfaces import validate_rows

                validate_rows(rows, task["scope"], labels=True)
                from .context_feedback_dispatch import FIELD, description, policy_for

                feedback_policy = policy_for(task)
                _equal(
                    value.get(FIELD),
                    description(task["scope"], policy=feedback_policy) if feedback_policy != "none" else None,
                )
                _equal(value.get("covariates"), task["scope"]["covariates"])
                if task.get("plugin_id") in {"generated_assay.v1", "generated_antibody.v1"}:
                    _equal(value.get("representation"), task["scope"]["representation"])
                _equal(value.get("acceptance_metric"), "mse_skill_vs_training_mean")
                _equal(
                    value.get("feedback_split"),
                    "whole independent groups within training; no group crosses the development split",
                )
            if task.get("plugin_id") == "generated_features.v1":
                from .generated_features import training_examples, validate_rows

                validate_rows(rows, task["scope"], labels=True)
                examples = training_examples(rows)
                _equal(value.get("representation"), task["scope"]["representation"])
                _equal(value.get("acceptance_metric"), "mse_skill_vs_training_mean")
            _equal(
                value.get("training"),
                {
                    "rows": len(rows),
                    "examples": examples,
                    "target_min": min(x["target"] for x in rows),
                    "target_max": max(x["target"] for x in rows),
                },
            )
        if phase == "develop" and task.get("plugin_id") in {"generated_classifier.v1", "generated_classifier.large.v1"}:
            _equal(
                value.get("training_class_counts"),
                [sum(r["target"] == k for r in rows) for k in range(len(task["scope"]["class_labels"]))],
            )
    elif name in {"evaluate_model_code", REFIT_TOOL}:
        if (
            phase != "develop"
            or type(value.get("attempts_remaining")) is not int
            or value["attempts_remaining"] not in {0, 1}
        ):
            raise ValueError("Invalid development budget or phase")
        attempt = _read(workspace, "code_attempt_" + str(2 - value["attempts_remaining"]) + ".json")
        if durable:
            number = 2 - value["attempts_remaining"]
            if number > len(ledger["attempts"]) or ledger["attempts"][number - 1]["status"] != "completed":
                raise ValueError("Evaluation has no completed durable reservation")
        status = attempt.get("status", "tested_not_published")
        if status not in {"tested_not_published", "development_failed"}:
            raise ValueError("Invalid model-development attempt status")
        expected = {
            "source_sha256": _source_hash(attempt["source_code"]),
            "training_only_feedback": attempt["training_feedback"],
            "attempts_remaining": value["attempts_remaining"],
            "status": status,
        }
        if name == REFIT_TOOL:
            expected["source_origin"] = validate_source_origin(
                capability, task, attempt["source_code"], attempt.get("source_origin")
            )
        elif "source_origin" in attempt:
            raise ValueError("Library refit cannot be reported as source authoring")
        if status == "development_failed":
            feedback = attempt["training_feedback"]
            if (
                set(feedback) != {"status", "stage", "error_type", "message"}
                or feedback["status"] != "failed"
                or feedback["stage"] not in {"source_validation", "execution", "prediction_validation"}
                or not isinstance(feedback["error_type"], str)
                or not feedback["error_type"]
                or not isinstance(feedback["message"], str)
                or len(feedback["message"]) > 3000
            ):
                raise ValueError("Invalid candidate-failure feedback")
            available = set()
            for number in range(1, 2 - value["attempts_remaining"]):
                prior = _read(workspace, "code_attempt_" + str(number) + ".json")
                if prior.get("status", "tested_not_published") == "tested_not_published":
                    available.add(_source_hash(prior["source_code"]))
            _equal(attempt.get("submittable_sources"), sorted(available))
            expected["submittable_sources"] = sorted(available)
        if status == "tested_not_published" and task.get("plugin_id") in {
            "generated_context.v1",
            "generated_context.v2",
            "generated_context.v3",
            "generated_assay.v1",
            "generated_antibody.v1",
        }:
            from .context_feedback_dispatch import verify_feedback

            verify_feedback(task, input_rows("training.json"), attempt["training_feedback"])
        _equal(value, expected)
    elif name == "submit_model_code":
        if phase != "develop":
            raise ValueError("Submission unavailable in this phase")
        model = _read(workspace, "candidate.json")
        provenance = _read(workspace, "source_provenance.json")
        rows = input_rows("training.json")
        sha = _source_hash(model["recipe"]["source_code"])
        _equal(model["source_sha256"], sha)
        _equal(provenance["source_sha256"], sha)
        if (Path(capability) / "protein_references.py").is_file() or "source_reference_audit" in provenance:
            _equal(provenance.get("source_reference_audit"), audit_source_references(model["recipe"]["source_code"]))
        _equal(model["training_sha256"], digest(rows))
        _equal(model["scope"], task["scope"])
        _equal(model["training_rows"], len(rows))
        if durable:
            submission = ledger["submission"]
            if not submission or submission["status"] != "completed":
                raise ValueError("Submission lacks completed durable evidence")
            _equal(submission["source_sha256"], sha)
            _equal(submission["receipt"], value)
        _equal(
            value,
            {
                "artifact": "candidate.json",
                "source_sha256": sha,
                "training_rows": len(rows),
                "status": "candidate",
                "registered": False,
                **handoff_receipt_fields(task, sha),
            },
        )
    elif name == "predict_with_generated_model":
        if phase != "reuse":
            raise ValueError("Prediction unavailable in this phase")
        p = packet()
        rows = input_rows("inputs.json")
        predictions = _read(workspace, "predictions.json")
        if task.get("plugin_id") in {
            "generated_context.v1",
            "generated_context.v2",
            "generated_context.v3",
            "generated_assay.v1",
            "generated_antibody.v1",
        }:
            from .context_interfaces import module_for, validate_rows

            validate_rows(rows, task["scope"], labels=False)
            if not module_for(task["plugin_id"]).prediction_rows_match(predictions, rows):
                raise ValueError("Context prediction identity or covariates changed")
        elif task.get("plugin_id") == "generated_features.v1":
            from .generated_features import validate_rows

            validate_rows(rows, task["scope"], labels=False)
            _equal(
                [{k: x[k] for k in ("id", "sequence")} for x in predictions],
                [{k: x[k] for k in ("id", "sequence")} for x in rows],
            )
            _equal([x.get("features_sha256") for x in predictions], [digest(x["features"]) for x in rows])
        else:
            _equal([{k: x[k] for k in ("id", "sequence")} for x in predictions], rows)
        if task.get("plugin_id") in {"generated_classifier.v1", "generated_classifier.large.v1"}:
            from .generated_classifier import validate_probabilities

            values = validate_probabilities(
                [x.get("probabilities") for x in predictions], len(rows), len(task["scope"]["class_labels"])
            )
            if any(
                type(x.get("prediction")) is not int or x["prediction"] != int(values[i].argmax())
                for i, x in enumerate(predictions)
            ):
                raise ValueError("Prediction class disagrees with probability vector")
        if any(type(x.get("prediction")) not in (int, float) for x in predictions):
            raise ValueError("Non-numeric prediction")
        _equal(
            value,
            {
                "artifact": "predictions.json",
                "model_id": p["model_id"],
                "rows": len(predictions),
                "prediction_sha256": digest(predictions),
            },
        )
    else:
        raise ValueError("No trusted result contract for this scientific tool")
