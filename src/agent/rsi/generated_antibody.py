"""Generated models with native paired-antibody identity and held-out groups.

The frozen descriptor ABI feeds isolated fit/predict code. Chain identities,
patents and target constructs remain bound to every numerical observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

try:
    from . import generated_assay as assay
    from . import poisk_pair_encoder as encoder
    from .code_sandbox import ModelOutputError, guard_source, run_code
    from .protein_model import digest
except ImportError:
    import generated_assay as assay
    import poisk_pair_encoder as encoder
    from code_sandbox import ModelOutputError, guard_source, run_code
    from protein_model import digest

PLUGIN_ID = "generated_antibody.v1"
ACCEPTANCE_METRIC = "mse_skill_vs_training_mean"
INPUT_FIELDS = ("id", "protein", "antibody", "heavy", "light", "target_sequence_sha256", "patents", "group", "values")
ENDPOINTS = {
    "KD": "negative_log10_KD_molar",
    "IC50": "negative_log10_IC50_molar",
    "EC50": "negative_log10_EC50_molar",
    "reported_molar_concentration": "negative_log10_reported_molar_concentration",
}
finite, text = assay.finite, assay.text


def implementation_hash():
    root = Path(__file__).resolve().parent
    names = [
        "generated_antibody.py",
        "poisk_pair_encoder.py",
        "generated_assay.py",
        "code_sandbox.py",
        "code_worker.py",
        "protein_references.py",
        "numeric_runtime.py",
        "protein_model.py",
    ]
    return hashlib.sha256(b"".join((root / n).read_bytes() for n in names)).hexdigest()


def validate_scope(scope):
    if not isinstance(scope, dict) or scope.get("interface_version") != PLUGIN_ID:
        raise ValueError("Explicit generated_antibody.v1 applicability required")
    if not isinstance(scope.get("grouping"), dict) or scope["grouping"].get("level") != "antibody_patent_component":
        raise ValueError("Antibody/patent component grouping required")
    # Reuse only generic scope-shape validation; no row or identity conversion.
    generic = {
        **scope,
        "interface_version": assay.PLUGIN_ID,
        "grouping": {**scope["grouping"], "level": "protein_publication_component"},
    }
    assay.validate_scope(generic)
    context = scope["context"]
    if (
        scope["protein"] not in encoder.TARGETS
        or scope["unit"] != "negative_log10_molar"
        or context.get("antibody_format") not in {"Conventional", "VHH"}
        or context.get("measurement_method") not in {"SPR", "BLI"}
        or context.get("reported_endpoint") not in ENDPOINTS
        or scope["target"] != ENDPOINTS[context["reported_endpoint"]]
        or context.get("condition") != "unreported_in_source"
    ):
        raise ValueError("Explicit target, antibody format and unpooled measurement endpoint required")
    if scope["covariates"] != encoder.covariates() or scope["representation"] != {
        "name": encoder.NAME,
        "dimensions": len(encoder.covariates()),
        "encoder_sha256": hashlib.sha256(Path(encoder.__file__).read_bytes()).hexdigest(),
    }:
        raise ValueError("Antibody descriptor map differs from the frozen encoder")


def partition_identities(row, scope):
    return [
        ("group", row["group"]),
        ("id", row["id"]),
        ("antibody", row["antibody"]),
        *(("chain", hashlib.sha256(seq.encode()).hexdigest()) for seq in (row["heavy"], row["light"]) if seq),
        *(("patent", patent) for patent in row["patents"]),
    ]


def validate_rows(rows, scope, *, labels):
    validate_scope(scope)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("Declare 1..10000 antibody measurements")
    required = set(INPUT_FIELDS) | ({"target"} if labels else set())
    seen, membership = set(), {}
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row) <= required | {"target"}:
            raise ValueError("Invalid paired-antibody fields")
        if any(not text(row[k]) for k in ("id", "protein", "antibody", "group")) or row["id"] in seen:
            raise ValueError("Unique antibody observation IDs required")
        seen.add(row["id"])
        heavy, light = row["heavy"], row["light"]
        if not isinstance(heavy, str) or not 80 <= len(heavy) <= 180 or set(heavy) - set(encoder.AA):
            raise ValueError("Complete variable heavy chain required")
        if (
            not isinstance(light, str)
            or (scope["context"]["antibody_format"] == "VHH" and light)
            or (scope["context"]["antibody_format"] == "Conventional" and not 70 <= len(light) <= 160)
            or set(light) - set(encoder.AA)
        ):
            raise ValueError("Light chain differs from declared antibody format")
        if row["antibody"] != digest([heavy, light]) or row["protein"] != scope["protein"]:
            raise ValueError("Paired antibody or target identity differs")
        if not isinstance(row["target_sequence_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", row["target_sequence_sha256"]
        ):
            raise ValueError("Source target construct digest required")
        patents = row["patents"]
        if (
            not isinstance(patents, list)
            or not patents
            or any(not isinstance(x, str) or not re.fullmatch(r"[0-9]+", x) for x in patents)
            or sorted(set(patents)) != patents
        ):
            raise ValueError("Sorted unique source US patent IDs required")
        for identity in partition_identities(row, scope):
            if identity in membership and membership[identity] != row["group"]:
                raise ValueError("Dependent antibody or patent crosses groups")
            membership[identity] = row["group"]
        values = row["values"]
        if not isinstance(values, list) or any(not finite(v) for v in values) or values != encoder.encode(row):
            raise ValueError("Numeric descriptors differ from the bound antibody chains")
        if "target" in row and (not finite(row["target"]) or not 0 <= row["target"] <= 18):
            raise ValueError("Finite negative-log molar point measurement required")


def inference_inputs(rows):
    return [{k: r[k] for k in INPUT_FIELDS} for r in rows]


def input_digest(row):
    return digest({k: row[k] for k in INPUT_FIELDS})


def pairs_overlap(left, right):
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 0, -1
    aligner.open_gap_score = aligner.extend_gap_score = -1

    def close(a, b):
        if not a or not b:
            return a == b
        maximum = int(0.05 * max(len(a), len(b)))
        return abs(len(a) - len(b)) <= maximum and aligner.score(a, b) >= -maximum

    return any(close(a[0], b[0]) and close(a[1], b[1]) for a in left for b in right)


def require_disjoint_parts(data, scope):
    seen, pairs = set(), set()
    for rows in data.values():
        identities = {i for row in rows for i in partition_identities(row, scope)}
        current = {(row["heavy"], row["light"]) for row in rows}
        if identities & seen or pairs_overlap(pairs, current):
            raise ValueError("Antibody, patent, exact chain, near clone or group crosses partitions")
        seen.update(identities)
        pairs.update(current)


def require_independent(model, rows):
    identities = {tuple(i) for i in model["training_partition_identities"]}
    if any(i in identities for r in rows for i in partition_identities(r, model["scope"])) or pairs_overlap(
        model["training_pairs"], {(r["heavy"], r["light"]) for r in rows}
    ):
        raise ValueError("Training/assessment antibody or patent overlap")


def development_split(rows):
    groups = sorted({r["group"] for r in rows}, key=lambda g: digest({"seed": 314159, "group": g}))
    options = []
    for offset in range(len(groups)):
        order = groups[offset:] + groups[:offset]
        for n in range(1, len(order)):
            selected = set(order[:n])
            train = sorted([r for r in rows if r["group"] in selected], key=lambda r: r["id"])
            feedback = sorted([r for r in rows if r["group"] not in selected], key=lambda r: r["id"])
            if len(train) >= 8 and len(feedback) >= 2:
                options.append(((abs(len(train) / len(rows) - 0.8), offset, n), train, feedback))
    if not options:
        raise ValueError("Insufficient group-held-out antibody training observations")
    _, train, feedback = min(options, key=lambda item: item[0])
    return train, feedback


def prediction_rows_match(predictions, rows):
    return len(predictions) == len(rows) and all(
        p.get("id") == r["id"]
        and p.get("protein") == r["protein"]
        and p.get("group") == r["group"]
        and p.get("inputs_sha256") == input_digest(r)
        for p, r in zip(predictions, rows, strict=True)
    )


def validate_task_evidence(definition, data, bound_file):
    scope, refs = definition["scope"], definition["references"]
    matches = [ref for ref in refs if ref["sha256"] == scope["grouping"]["evidence_sha256"]]
    if len(matches) != 1 or sum(ref["sha256"] == scope["representation"]["encoder_sha256"] for ref in refs) != 1:
        raise ValueError("Unique antibody grouping and encoder references required")
    evidence = json.loads(bound_file(matches[0]).read_text())
    if (
        set(evidence) != {"schema", "level", "study_doi", "sources", "observations", "source_observations"}
        or evidence["schema"] != "antibody_groups.v1"
        or evidence["level"] != scope["grouping"]["level"]
        or evidence["study_doi"] != definition["paper"]["doi"]
    ):
        raise ValueError("Invalid antibody grouping evidence")
    if (
        not isinstance(evidence["sources"], list)
        or not evidence["sources"]
        or evidence["source_observations"] not in evidence["sources"]
    ):
        raise ValueError("Bound source observations required")
    for ref in evidence["sources"]:
        if ref not in refs or ref == matches[0]:
            raise ValueError("Grouping sources must be separately bound")
        bound_file(ref)
    expected = {
        r["id"]: {"group": r["group"], "phase": phase, "input_sha256": input_digest(r)}
        for phase, rows in data.items()
        for r in rows
    }
    if evidence["observations"] != expected:
        raise ValueError("Antibody inputs or groups changed")
    source = {r["id"]: r for r in json.loads(bound_file(evidence["source_observations"]).read_text())}
    for rows in data.values():
        for r in rows:
            original = source.get(r["id"], {})
            fields = ("heavy", "light", "antibody", "target_sequence_sha256", "patents", "target")
            if (
                any(original.get(k) != r[k] for k in fields)
                or original.get("target_accession") != r["protein"]
                or original.get("endpoint") != scope["context"]["reported_endpoint"]
                or original.get("method") != scope["context"]["measurement_method"]
                or original.get("antibody_type") != scope["context"]["antibody_format"]
                or original.get("condition") != ""
            ):
                raise ValueError("Antibody measurement differs from source or endpoint")
    require_disjoint_parts(data, scope)
    development_split(data["training"])
    return evidence


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
        "training_pairs": [list(pair) for pair in sorted({(r["heavy"], r["light"]) for r in rows})],
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
