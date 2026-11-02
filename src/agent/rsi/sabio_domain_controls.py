"""Fixed, train-only enzyme-family and substrate/condition reference predictors."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import generated_assay
from . import sabio_assay_encoder as encoder
from .model_tasks import bound_file
from .storage import sha256_file

METHODS = ("enzyme_family_median", "substrate_condition_reference")


def source_metadata(task, data):
    import json

    if task["scope"]["representation"]["name"] != encoder.NAME:
        raise ValueError("The domain controls require the declared plant assay representation")
    source = json.loads(bound_file(task["references"][0]).read_text())
    if not isinstance(source, list) or len({r["id"] for r in source}) != len(source):
        raise ValueError("Unique source observations required for domain controls")
    indexed = {row["id"]: row for row in source}
    metadata = {}
    for rows in data.values():
        for row in rows:
            raw = indexed[row["id"]]
            if (
                raw["protein"] != row["protein"]
                or raw["publication_group"] != row["study"]
                or raw["log10_value"] != row["target"]
                or encoder.substrate_identity(raw) != row["substrate"]
                or encoder.encode(raw, row["sequence"]) != row["values"]
            ):
                raise ValueError("Domain metadata differs from the admitted observation")
            if raw["property"] == "kcat":
                substrates = sorted(
                    {
                        part["compound"]["id"]
                        for part in raw["reaction"]["species"]
                        if part.get("role") == "Substrate" and part.get("compound", {}).get("id", 0) > 0
                    }
                )
                substrate = "reaction_substrates:" + ",".join(map(str, substrates))
            else:
                substrate = row["substrate"]
            metadata[row["id"]] = {
                "ec": raw["ec_number"],
                "substrate": substrate,
                "ph": encoder.interval(raw["conditions"].get("envvar_ph")),
                "temperature": encoder.interval(raw["conditions"].get("envvar_temperature"), temperature=True),
            }
    return metadata


def grouped_median(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row["target"])
    return float(np.median([np.median(values) for values in grouped.values()]))


def family_reference(training, item, metadata):
    ec = metadata[item["id"]]["ec"].split(".")
    for level in (4, 3, 2, 1):
        selected = [r for r in training if metadata[r["id"]]["ec"].split(".")[:level] == ec[:level]]
        groups = len({r["group"] for r in selected})
        if len(selected) >= 3 and groups >= 2:
            return grouped_median(selected), f"ec_prefix_{level}", groups
    return grouped_median(training), "global_group_median", len({r["group"] for r in training})


def interval_distance(a, b, scale):
    if a[2] or b[2]:
        return float(a[2] != b[2])
    return max(a[0] - b[1], b[0] - a[1], 0.0) / scale


def predictions(training, scoring, metadata):
    """Metadata contains inputs only; only training rows provide fitted labels."""
    result = {name: [] for name in METHODS}
    for row in scoring:
        target = metadata[row["id"]]
        fallback, basis, support = family_reference(training, row, metadata)
        result[METHODS[0]].append(
            {
                "id": row["id"],
                "inputs_sha256": generated_assay.input_digest(row),
                "prediction": fallback,
                "basis": basis,
                "fit_support_groups": support,
            }
        )
        selected = [
            r
            for r in training
            if metadata[r["id"]]["ec"].split(".")[:2] == target["ec"].split(".")[:2]
            and metadata[r["id"]]["substrate"] == target["substrate"]
        ]
        prediction = fallback
        condition_basis = "family_fallback:" + basis
        if len(selected) >= 3 and len({r["group"] for r in selected}) >= 2:
            ranked = sorted(
                selected,
                key=lambda r: (
                    interval_distance(metadata[r["id"]]["ph"], target["ph"], 14.0)
                    + interval_distance(metadata[r["id"]]["temperature"], target["temperature"], 100.0),
                    r["id"],
                ),
            )
            group_nearest = {}
            for candidate in ranked:
                group_nearest.setdefault(candidate["group"], candidate)
            neighbours = list(group_nearest.values())[:3]
            prediction = float(np.median([r["target"] for r in neighbours]))
            support = len(neighbours)
            condition_basis = "same_ec_prefix_2_substrate_nearest_conditions"
        result[METHODS[1]].append(
            {
                "id": row["id"],
                "inputs_sha256": generated_assay.input_digest(row),
                "prediction": prediction,
                "basis": condition_basis,
                "fit_support_groups": support,
            }
        )
    return result


def score_rows(training, scoring, scope, metadata):
    generated_assay.validate_rows(training, scope, labels=True)
    generated_assay.validate_rows(scoring, scope, labels=True)
    generated_assay.require_disjoint_parts({"training": training, "scoring": scoring}, scope)
    if any(set(metadata[r["id"]]) != {"ec", "substrate", "ph", "temperature"} for r in training + scoring):
        raise ValueError("Domain predictor metadata must contain inputs only")
    outcomes = predictions(training, scoring, metadata)
    truth = np.asarray([r["target"] for r in scoring])
    null = float(np.mean([r["target"] for r in training]))
    null_mse = float(np.square(truth - null).mean())
    result = []
    for name, rows in outcomes.items():
        values = np.asarray([r["prediction"] for r in rows])
        mse = float(np.square(truth - values).mean())
        result.append(
            {
                "method": name,
                "status": "completed",
                "predictions": rows,
                "metrics": {
                    "rmse": mse**0.5,
                    "mae": float(np.abs(truth - values).mean()),
                    "mse_skill_vs_training_mean": 1 - mse / null_mse if null_mse else None,
                    "spearman": float(spearmanr(truth, values).statistic)
                    if np.ptp(truth) > 0 and np.ptp(values) > 0
                    else None,
                },
                "fit_partition": "training",
                "hyperparameters": "fixed; equal component weight for family medians; nearest three eligible components for condition reference",
                "implementation_sha256": sha256_file(Path(__file__)),
            }
        )
    return result
