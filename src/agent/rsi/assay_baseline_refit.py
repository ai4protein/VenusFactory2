"""Independent numerical refits of fixed assay controls from frozen inputs.

This implementation does not call the production baseline or scoring modules.
Numerical estimators use the same declared sklearn algorithms and versions.
Descriptor construction is not reimplemented here: frozen descriptor and source
bindings are checked separately from estimator refitting and score replay.
"""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .assay_score_replay import INPUTS, canonical, require

NUMERICAL = ("training_mean", "assay_ridge", "assay_random_forest", "assay_svr")
DOMAIN = ("enzyme_family_median", "substrate_condition_reference")


def read_bound(reference):
    path = Path(reference["path"])
    require(not any(p.is_symlink() for p in [path, *path.parents]), "Symlinked input")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == reference["sha256"], "Bound input changed")
    return json.loads(payload)


def query_rows(rows):
    return [{field: row[field] for field in INPUTS} for row in rows]


def condition_interval(raw):
    if raw is None or raw == {}:
        return (0.0, 0.0, True)
    low = raw["start_value"]
    high = raw.get("end_value")
    high = low if high is None else high
    require(
        all(type(v) in (float, int) and math.isfinite(v) for v in [low, high]) and low <= high,
        "Invalid source condition",
    )
    return (float(low), float(high), False)


def source_metadata(task, training, inputs):
    """Read source input fields; assessment source targets never enter refitting."""
    raw = read_bound(task["references"][0])
    require(isinstance(raw, list), "Source observations required")
    index = {row["id"]: row for row in raw}
    require(len(index) == len(raw), "Duplicate source observation")
    for row in training:
        require(row["target"] == index[row["id"]]["log10_value"], "Training label differs from source")
    metadata = {}
    for row in query_rows(training) + inputs:
        original = index[row["id"]]
        require(
            row["protein"] == original["protein"] and row["study"] == original["publication_group"],
            "Source identity differs",
        )
        participants = sorted(
            {
                part["compound"]["id"]
                for part in original["reaction"]["species"]
                if part.get("role") == "Substrate"
                and type(part.get("compound", {}).get("id")) is int
                and part["compound"]["id"] > 0
            }
        )
        if original["property"] == "kcat":
            require(original["associated_substrate"] is None, "Reaction-level target has associated substrate")
            identity = f"sabio_reaction:{original['reaction']['id']}"
            substrate = "reaction_substrates:" + ",".join(map(str, participants))
        else:
            require(original["property"] in {"Km", "kcat/Km"}, "Unsupported endpoint")
            compound = original["associated_substrate"]["id"]
            require(compound in participants, "Associated substrate absent from reaction")
            identity = substrate = f"sabio_compound:{compound}"
        require(participants and row["substrate"] == identity, "Substrate identity differs")
        ec = original["ec_number"].split(".")
        require(len(ec) == 4 and all(part.isdigit() for part in ec), "Complete EC required")
        conditions = original["conditions"]
        metadata[row["id"]] = {
            "ec": tuple(ec),
            "substrate": substrate,
            "ph": condition_interval(conditions.get("envvar_ph")),
            "temperature": condition_interval(conditions.get("envvar_temperature")),
        }
    return metadata


def _group_median(rows):
    values = defaultdict(list)
    for row in rows:
        values[row["group"]].append(row["target"])
    return float(statistics.median(statistics.median(items) for items in values.values()))


def _distance(left, right, scale):
    if left[2] or right[2]:
        return float(left[2] != right[2])
    return max(0.0, left[0] - right[1], right[0] - left[1]) / scale


def domain_predictions(training, inputs, metadata):
    require(
        all(set(metadata[row["id"]]) == {"ec", "substrate", "ph", "temperature"} for row in training + inputs),
        "Domain metadata must contain inputs only",
    )
    results = {method: [] for method in DOMAIN}
    for query in inputs:
        target = metadata[query["id"]]
        fallback = _group_median(training)
        basis, support = "global_group_median", len({row["group"] for row in training})
        for level in range(4, 0, -1):
            cohort = [row for row in training if metadata[row["id"]]["ec"][:level] == target["ec"][:level]]
            groups = {row["group"] for row in cohort}
            if len(cohort) >= 3 and len(groups) >= 2:
                fallback, basis, support = _group_median(cohort), f"ec_prefix_{level}", len(groups)
                break
        results[DOMAIN[0]].append((fallback, basis, support))
        selected = [
            row
            for row in training
            if metadata[row["id"]]["ec"][:2] == target["ec"][:2]
            and metadata[row["id"]]["substrate"] == target["substrate"]
        ]
        prediction, condition_basis = fallback, "family_fallback:" + basis
        if len(selected) >= 3 and len({row["group"] for row in selected}) >= 2:
            by_group = {}
            ordered = sorted(
                selected,
                key=lambda row: (
                    _distance(metadata[row["id"]]["ph"], target["ph"], 14.0)
                    + _distance(metadata[row["id"]]["temperature"], target["temperature"], 100.0),
                    row["id"],
                ),
            )
            for row in ordered:
                by_group.setdefault(row["group"], row)
            neighbours = list(by_group.values())[:3]
            prediction = float(statistics.median(row["target"] for row in neighbours))
            support, condition_basis = len(neighbours), "same_ec_prefix_2_substrate_nearest_conditions"
        results[DOMAIN[1]].append((prediction, condition_basis, support))
    return results


def refit_predictions(training, inputs, *, metadata=None):
    require(bool(training) and bool(inputs), "Nonempty training and query rows required")
    require(all(set(row) == set(INPUTS) for row in inputs), "Queries must contain only unlabelled inputs")
    for rows in [training, inputs]:
        require(len({row["id"] for row in rows}) == len(rows), "Duplicate row identity")
    for field in ["id", "protein", "sequence", "study", "group"]:
        require(not {row[field] for row in training} & {row[field] for row in inputs}, "Partition overlap: " + field)
    require(
        all(type(row["target"]) in (float, int) and math.isfinite(row["target"]) for row in training),
        "Finite training targets required",
    )
    x = np.asarray([row["values"] for row in training], dtype=float)
    q = np.asarray([row["values"] for row in inputs], dtype=float)
    require(
        x.ndim == q.ndim == 2
        and x.shape[1] == q.shape[1]
        and x.shape[1] > 0
        and np.isfinite(x).all()
        and np.isfinite(q).all(),
        "Invalid descriptor matrices",
    )
    y = np.asarray([row["target"] for row in training], dtype=float)
    centre, scale = float(np.mean(y)), float(np.std(y)) or 1.0
    standardized = (y - centre) / scale
    scaler = StandardScaler().fit(x)
    fitted = {
        "assay_ridge": Ridge(alpha=1.0).fit(scaler.transform(x), standardized),
        "assay_random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=0, n_jobs=1
        ).fit(x, standardized),
        "assay_svr": SVR(C=1.0, epsilon=0.1, gamma="scale").fit(scaler.transform(x), standardized),
    }
    predictions = {"training_mean": [centre] * len(inputs)}
    for name, estimator in fitted.items():
        matrix = q if name == "assay_random_forest" else scaler.transform(q)
        predictions[name] = (estimator.predict(matrix) * scale + centre).tolist()
    domain = domain_predictions(training, inputs, metadata) if metadata is not None else {}
    predictions.update({name: [value[0] for value in values] for name, values in domain.items()})
    result = {}
    for name, values in predictions.items():
        require(all(math.isfinite(value) for value in values), "Nonfinite independent prediction")
        result[name] = []
        for index, (row, value) in enumerate(zip(inputs, values, strict=True)):
            prediction = {
                "id": row["id"],
                "protein": row["protein"],
                "group": row["group"],
                "inputs_sha256": canonical(row),
                "prediction": float(value),
            }
            if name in domain:
                prediction.update(basis=domain[name][index][1], fit_support_groups=domain[name][index][2])
            result[name].append(prediction)
    return result


def verify_controls(task, controls, *, partition="future"):
    require(task["plugin_id"] == "generated_assay.v1" and partition in {"validation", "future"}, "Assay phase required")
    training = read_bound(task["data"]["training"])
    inputs = query_rows(read_bound(task["data"][partition]))
    domain = task["scope"]["representation"]["name"] == "plant_assay_descriptors.v1"
    expected_methods = set(NUMERICAL + (DOMAIN if domain else ()))
    require(
        len(controls) == len(expected_methods) and {row["method"] for row in controls} == expected_methods,
        "Control methods missing, duplicated or unexpected",
    )
    require(all(row["status"] == "completed" for row in controls), "Incomplete controls cannot pass refit audit")
    metadata = source_metadata(task, training, inputs) if domain else None
    independent = refit_predictions(training, inputs, metadata=metadata)
    errors = {}
    for control in controls:
        name = control["method"]
        saved = control["predictions"]
        require(len(saved) == len(inputs), "Prediction count differs")
        maximum = 0.0
        for actual, expected in zip(saved, independent[name], strict=True):
            for key in expected:
                if key != "prediction":
                    require(actual.get(key) == expected[key], "Prediction input or provenance differs: " + key)
            value = actual.get("prediction")
            require(type(value) in (float, int) and math.isfinite(value), "Invalid saved prediction")
            error = abs(value - expected["prediction"])
            require(error <= 1e-9 * max(1.0, abs(expected["prediction"])), "Independent baseline prediction differs")
            maximum = max(maximum, error)
        errors[name] = maximum
    return {
        "task_id": task["id"],
        "task_sha256": canonical(task),
        "status": "independent_baselines_refitted",
        "fit_partition": "training",
        "score_partition": partition,
        "methods": list(independent),
        "maximum_absolute_prediction_error": max(errors.values()),
        "per_method_maximum_error": errors,
        "prediction_sha256": {name: canonical(rows) for name, rows in independent.items()},
        "source_metadata_reconstructed": domain,
        "descriptor_encoding_reimplemented": False,
        "baseline_refit_performed": True,
        "new_autonomous_models": 0,
        "external_model_calls": 0,
    }
