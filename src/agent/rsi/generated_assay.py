"""Generated numeric models for source-bound protein/substrate assay observations.

Protein sequences may vary in length and repeat under distinct measured
conditions. Source-defined groups govern development, acceptance and reuse.
Only the host-bound numeric representation and permitted labels enter workers.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np

try:
    from .code_sandbox import ModelOutputError, guard_source, run_code
    from .protein_model import digest
except ImportError:  # Standalone capability bundle.
    from code_sandbox import ModelOutputError, guard_source, run_code
    from protein_model import digest

PLUGIN_ID = "generated_assay.v1"
ACCEPTANCE_METRIC = "mse_skill_vs_training_mean"
GROUP_LEVELS = {"publication", "protein_identity", "protein_publication_component"}
INPUT_FIELDS = ("id", "protein", "sequence", "substrate", "study", "group", "values")
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def implementation_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in [
                "generated_assay.py",
                "code_sandbox.py",
                "code_worker.py",
                "protein_references.py",
                "numeric_runtime.py",
                "protein_model.py",
            ]
        )
    ).hexdigest()


def text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 256


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_scope(scope):
    if not isinstance(scope, dict) or set(scope) != {
        "protein",
        "assay",
        "target",
        "unit",
        "context",
        "covariates",
        "grouping",
        "interface_version",
        "representation",
    }:
        raise ValueError("Context model requires protein, endpoint, context, covariates and grouping")
    if scope["interface_version"] != PLUGIN_ID:
        raise ValueError("Explicit generated_assay.v1 scope version required")
    if any(not text(scope[key]) for key in ("protein", "assay", "target", "unit")):
        raise ValueError("Nonempty scientific identity and units required")
    context = scope["context"]
    if (
        not isinstance(context, dict)
        or not 1 <= len(context) <= 16
        or any(not text(k) or not text(v) for k, v in context.items())
    ):
        raise ValueError("Recorded assay context required")
    grouping = scope["grouping"]
    if (
        not isinstance(grouping, dict)
        or set(grouping) != {"level", "evidence_sha256"}
        or not isinstance(grouping["level"], str)
        or grouping["level"] not in GROUP_LEVELS
        or not isinstance(grouping["evidence_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", grouping["evidence_sha256"]) is None
    ):
        raise ValueError("Independent-group level and source evidence digest required")
    columns = scope["covariates"]
    if not isinstance(columns, list) or not 1 <= len(columns) <= 512:
        raise ValueError("Declare 1..512 source-bound numeric covariates")
    representation = scope["representation"]
    if (
        not isinstance(representation, dict)
        or set(representation) != {"name", "dimensions", "encoder_sha256"}
        or not text(representation["name"])
        or type(representation["dimensions"]) is not int
        or representation["dimensions"] != len(columns)
        or not isinstance(representation["encoder_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", representation["encoder_sha256"]) is None
    ):
        raise ValueError("Frozen representation identity, dimensions and encoder digest required")
    seen = set()
    for column in columns:
        if (
            not isinstance(column, dict)
            or set(column) != {"name", "unit", "minimum", "maximum"}
            or not text(column["name"])
            or column["name"] in seen
            or not text(column["unit"])
            or not finite(column["minimum"])
            or not finite(column["maximum"])
            or column["minimum"] >= column["maximum"]
        ):
            raise ValueError("Unique covariates with units and finite increasing domains required")
        seen.add(column["name"])


def validate_rows(rows, scope, *, labels):
    validate_scope(scope)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("Declare 1..10000 assay observations")
    seen, entity_sequences, membership = set(), {}, {}
    required = set(INPUT_FIELDS) | ({"target"} if labels else set())
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row) <= required | {"target"}:
            raise ValueError("Invalid assay observation fields")
        if any(not text(row[key]) for key in ("id", "protein", "substrate", "study", "group")) or row["id"] in seen:
            raise ValueError("Unique observation IDs and explicit scientific identities required")
        seen.add(row["id"])
        sequence = row["sequence"]
        if not isinstance(sequence, str) or not 1 <= len(sequence) <= 4096 or set(sequence) - AA:
            raise ValueError("A complete variable-length protein sequence is required")
        if row["protein"] in entity_sequences and entity_sequences[row["protein"]] != sequence:
            raise ValueError("One protein identity maps to multiple sequences")
        entity_sequences[row["protein"]] = sequence
        for kind, identity in partition_identities(row, scope):
            key = (kind, identity)
            if key in membership and membership[key] != row["group"]:
                raise ValueError("Dependent assay identity crosses independent groups")
            membership[key] = row["group"]
        values = row["values"]
        if not isinstance(values, list) or len(values) != len(scope["covariates"]):
            raise ValueError("Numeric inputs must follow the frozen representation")
        for value, column in zip(values, scope["covariates"], strict=True):
            if not finite(value) or not column["minimum"] <= value <= column["maximum"]:
                raise ValueError("Finite numeric inputs inside declared domains required")
            if column["unit"] == "binary_indicator" and value not in (0, 1):
                raise ValueError("Binary missingness or condition indicators require zero or one")
        if "target" in row and not finite(row["target"]):
            raise ValueError("Finite observed targets required")


def partition_identities(row, scope):
    level = scope["grouping"]["level"]
    identities = [("group", row["group"]), ("id", row["id"])]
    if level in {"publication", "protein_publication_component"}:
        identities.append(("study", row["study"]))
    if level in {"protein_identity", "protein_publication_component"}:
        identities.extend([("protein", row["protein"]), ("sequence", row["sequence"])])
    return identities


def inference_inputs(rows):
    return [{key: row[key] for key in INPUT_FIELDS} for row in rows]


def input_digest(row):
    return digest({key: row[key] for key in INPUT_FIELDS})


def require_independent(model, rows):
    scope = model["scope"]
    identities = {tuple(item) for item in model["training_partition_identities"]}
    if any(identity in identities for row in rows for identity in partition_identities(row, scope)):
        raise ValueError("Training/assessment assay identity or independent-group overlap")


def require_disjoint_parts(data, scope):
    seen = set()
    entity_sequences = {}
    for rows in data.values():
        for row in rows:
            if row["protein"] in entity_sequences and entity_sequences[row["protein"]] != row["sequence"]:
                raise ValueError("One protein identity maps to multiple sequences across partitions")
            entity_sequences[row["protein"]] = row["sequence"]
        identities = {item for row in rows for item in partition_identities(row, scope)}
        if identities & seen:
            raise ValueError("Assay protein, publication, observation or group crosses partitions")
        seen.update(identities)


def validate_task_evidence(definition, data, bound_file):
    scope = definition["scope"]
    references = definition["references"]
    if sum(ref["sha256"] == scope["representation"]["encoder_sha256"] for ref in references) != 1:
        raise ValueError("Assay encoder source must be uniquely bound in task references")
    matches = [ref for ref in references if ref["sha256"] == scope["grouping"]["evidence_sha256"]]
    if len(matches) != 1:
        raise ValueError("Assay grouping evidence must be uniquely bound")
    import json

    evidence = json.loads(bound_file(matches[0]).read_text())
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"schema", "level", "study_doi", "sources", "observations"}
        or evidence["schema"] != "assay_groups.v1"
        or evidence["level"] != scope["grouping"]["level"]
        or evidence["study_doi"] != definition["paper"]["doi"]
        or not isinstance(evidence["sources"], list)
        or not evidence["sources"]
    ):
        raise ValueError("Invalid source-bound assay grouping evidence")
    for source in evidence["sources"]:
        if source == matches[0] or source not in references:
            raise ValueError("Grouping source must be a separately bound task reference")
        bound_file(source)
    expected = {
        row["id"]: {"group": row["group"], "phase": phase, "input_sha256": input_digest(row)}
        for phase, rows in data.items()
        for row in rows
    }
    if evidence["observations"] != expected:
        raise ValueError("Assay inputs or groups differ from source-bound evidence")
    require_disjoint_parts(data, scope)
    development_split(data["training"])
    return evidence


def development_split(rows):
    """Hold out whole source-defined groups within the permitted training cohort."""
    groups = sorted({r["group"] for r in rows}, key=lambda g: digest({"seed": 314159, "group": g}))
    if len(groups) < 2:
        raise ValueError("Training feedback requires at least two independent groups")
    selected = set(groups[: max(1, min(len(groups) - 1, int(len(groups) * 0.8)))])
    training = sorted([r for r in rows if r["group"] in selected], key=lambda r: r["id"])
    feedback = sorted([r for r in rows if r["group"] not in selected], key=lambda r: r["id"])
    if len(training) < 8 or len(feedback) < 2:
        raise ValueError("Insufficient observations for group-held-out training feedback")
    return training, feedback


def prediction_rows_match(predictions, rows):
    return len(predictions) == len(rows) and all(
        p.get("id") == row["id"]
        and p.get("protein") == row["protein"]
        and p.get("group") == row["group"]
        and p.get("inputs_sha256") == input_digest(row)
        for p, row in zip(predictions, rows, strict=True)
    )


def fit_candidate(rows, scope, recipe):
    validate_rows(rows, scope, labels=True)
    if not 8 <= len(rows) <= 600 or set(recipe) != {"source_code", "seed"} or type(recipe["seed"]) is not int:
        raise ValueError("Context fitting requires source, integer seed and 8..600 observations")
    reviewed = guard_source(recipe["source_code"])
    result = run_code(
        recipe["source_code"],
        {
            "operation": "fit",
            "input_kind": "features",
            "features": [r["values"] for r in rows],
            "targets": [r["target"] for r in rows],
            "seed": recipe["seed"],
        },
    )
    return {
        "schema": 1,
        "plugin_id": PLUGIN_ID,
        "scope": scope,
        "development_type": "agent_generated_model_code",
        "input_kind": "features",
        "recipe": recipe,
        "source_sha256": reviewed["source_sha256"],
        "training_sha256": digest(rows),
        "training_rows": len(rows),
        "training_ids": [r["id"] for r in rows],
        "training_groups": sorted({r["group"] for r in rows}),
        "training_sequences": sorted({r["sequence"] for r in rows}),
        "training_partition_identities": [
            list(item) for item in sorted({item for row in rows for item in partition_identities(row, scope)})
        ],
        "training_mean": float(np.mean([r["target"] for r in rows])),
        "model_state": result["model"],
    }


def predict(model, rows, scope):
    if model.get("schema") != 1 or model.get("plugin_id") != PLUGIN_ID or model.get("scope") != scope:
        raise ValueError("Context model applicability, covariates or units mismatch")
    validate_rows(rows, scope, labels=False)
    code = model["recipe"]["source_code"]
    if hashlib.sha256(code.encode()).hexdigest() != model["source_sha256"]:
        raise ValueError("Context model source changed")
    result = run_code(
        code,
        {
            "operation": "predict",
            "input_kind": "features",
            "model": model["model_state"],
            "features": [r["values"] for r in rows],
            "seed": model["recipe"]["seed"],
        },
    )
    values = result.get("predictions")
    if not isinstance(values, list) or len(values) != len(rows) or any(not finite(v) for v in values):
        raise ModelOutputError("One finite numeric prediction required per contextual observation")
    return [
        {
            "id": row["id"],
            "protein": row["protein"],
            "group": row["group"],
            "inputs_sha256": input_digest(row),
            "prediction": float(value),
        }
        for row, value in zip(rows, values, strict=True)
    ]


def evaluate_with_predictions(model, rows, scope):
    from scipy.stats import spearmanr

    validate_rows(rows, scope, labels=True)
    require_independent(model, rows)
    predictions = predict(model, rows, scope)
    truth = np.asarray([r["target"] for r in rows], dtype=float)
    values = np.asarray([r["prediction"] for r in predictions], dtype=float)
    mse = float(np.square(truth - values).mean())
    baseline = float(np.square(truth - model["training_mean"]).mean())
    if baseline <= 0:
        raise ValueError("Degenerate assessment baseline")
    correlation = float(spearmanr(truth, values).statistic) if np.ptp(truth) > 0 and np.ptp(values) > 0 else 0.0
    metrics = {
        "rows": len(rows),
        "independent_groups": len({r["group"] for r in rows}),
        "rmse": math.sqrt(mse),
        "mae": float(np.abs(truth - values).mean()),
        "training_mean_rmse": math.sqrt(baseline),
        ACCEPTANCE_METRIC: 1 - mse / baseline,
        "spearman": correlation,
        "predictions_sha256": digest(predictions),
    }
    return metrics, predictions


def evaluate(model, rows, scope):
    return evaluate_with_predictions(model, rows, scope)[0]
