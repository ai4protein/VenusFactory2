"""Host-side scores from predictions, independent of model-plugin evaluators."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

from agent.harness.contracts import digest


def require(value, message):
    if not value:
        raise ValueError(message)


def regression_scores(train, y, p):
    """Numerical scores shared by sequence and grouped-condition interfaces."""
    require(p.shape == y.shape and np.isfinite(p).all(), "Invalid regression predictions")
    mse = math.fsum(float(a - b) ** 2 for a, b in zip(y, p, strict=True)) / len(y)
    mean = math.fsum(train) / len(train)
    baseline = math.fsum(float(a - mean) ** 2 for a in y) / len(y)
    require(math.isfinite(mse) and math.isfinite(baseline), "Nonfinite regression loss")
    ry, rp = rankdata(y), rankdata(p)
    constant = bool(np.ptp(ry) == 0 or np.ptp(rp) == 0)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": math.fsum(abs(float(a - b)) for a, b in zip(y, p, strict=True)) / len(y),
        "training_mean": mean,
        "training_mean_mse": baseline,
        "mse_skill_vs_training_mean": None if baseline == 0 else 1 - mse / baseline,
        "spearman": None if constant else float(np.corrcoef(ry, rp)[0, 1]),
        "undefined_spearman": constant,
        "degenerate_baseline": baseline == 0,
    }


def measure_context(training, future, predictions, *, scope):
    """Independently score condition predictions, preserving sample-group isolation.

    Repeated physical conditions across independent groups are valid. Observation
    IDs, protein, group, numeric inputs and order must match the prediction witness.
    Labels and the training-mean reference come from the bound datasets only.
    """
    require(len(training) > 0 and len(future) >= 2, "Insufficient scoring observations")
    columns = scope["covariates"]
    require(bool(columns), "Missing contextual covariates")
    for rows in (training, future):
        require(len({r["id"] for r in rows}) == len(rows), "Repeated scoring observations")
        for row in rows:
            require(
                all(isinstance(row[k], str) and row[k].strip() for k in ("id", "protein", "group"))
                and row["protein"] == scope["protein"],
                "Invalid contextual observation identity",
            )
            require(type(row["target"]) in (int, float) and math.isfinite(row["target"]), "Nonfinite target")
            require(isinstance(row["values"], list) and len(row["values"]) == len(columns), "Invalid covariates")
            for value, column in zip(row["values"], columns, strict=True):
                require(
                    type(value) in (int, float)
                    and math.isfinite(value)
                    and column["minimum"] <= value <= column["maximum"]
                    and (column["unit"] != "binary_indicator" or value in (0, 1)),
                    "Invalid covariate value or domain",
                )
    require(not {r["id"] for r in training} & {r["id"] for r in future}, "Training/scoring observation overlap")
    require(
        not {r["group"] for r in training} & {r["group"] for r in future}, "Training/scoring independent-group overlap"
    )
    expected = [(r["id"], r["protein"], r["group"], digest(r["values"])) for r in future]
    require(
        [(r.get("id"), r.get("protein"), r.get("group"), r.get("values_sha256")) for r in predictions] == expected,
        "Prediction identities, covariates or order differ from the admitted cohort",
    )
    require(all(type(r["prediction"]) in (int, float) for r in predictions), "Nonnumeric prediction")
    train = np.asarray([r["target"] for r in training], dtype=float)
    y = np.asarray([r["target"] for r in future], dtype=float)
    p = np.asarray([r["prediction"] for r in predictions], dtype=float)
    result = {
        "rows": len(future),
        "independent_groups": len({r["group"] for r in future}),
        "training_groups": len({r["group"] for r in training}),
        "grouping_level": scope["grouping"]["level"],
        "predictions_sha256": digest(predictions),
        **regression_scores(train, y, p),
    }
    if scope["unit"] == "ratio_to_within_replicate_reference":
        temperature = [i for i, c in enumerate(columns) if c["name"] == "temperature" and c["unit"] == "degC"]
        reference = [i for i, c in enumerate(columns) if c["name"] == "reference_temperature" and c["unit"] == "degC"]
        require(len(temperature) == len(reference) == 1, "Reference-normalized score requires temperature columns")
        ti, ri = temperature[0], reference[0]
        mask = np.asarray(
            [not math.isclose(r["values"][ti], r["values"][ri], rel_tol=0, abs_tol=1e-12) for r in future]
        )
        result["non_reference_metrics"] = (
            {"rows": int(mask.sum()), **regression_scores(train, y[mask], p[mask])} if mask.any() else None
        )
    return result


def measure(training, future, predictions, *, classes=None):
    """Recompute baselines from admitted training labels, never from model state."""
    require(len(training) > 0 and len(future) >= 2, "Insufficient scoring observations")
    identities = [(r["id"], r["sequence"]) for r in future]
    require(
        len(identities) == len({r["id"] for r in future}) == len({r["sequence"] for r in future}),
        "Repeated scoring observations",
    )
    require(
        [(r["id"], r["sequence"]) for r in predictions] == identities,
        "Prediction identities or order differ from the admitted cohort",
    )
    require(
        not ({r["id"] for r in training} & {r["id"] for r in future})
        and not ({r["sequence"] for r in training} & {r["sequence"] for r in future}),
        "Training and scoring observations overlap",
    )
    y = np.asarray([r["target"] for r in future], dtype=float)
    train = np.asarray([r["target"] for r in training], dtype=float)
    require(np.isfinite(y).all() and np.isfinite(train).all(), "Nonfinite target")
    result = {"rows": len(y), "predictions_sha256": digest(predictions)}
    if classes is None:
        require(all(type(r["prediction"]) in (int, float) for r in predictions), "Nonnumeric prediction")
        p = np.asarray([r["prediction"] for r in predictions], dtype=float)
        result.update(regression_scores(train, y, p))
    else:
        require(type(classes) is int and classes >= 2, "Invalid class count")
        require(
            np.all(y == np.floor(y))
            and np.all(train == np.floor(train))
            and np.all((y >= 0) & (y < classes))
            and np.all((train >= 0) & (train < classes)),
            "Invalid class labels",
        )
        y, train = y.astype(int), train.astype(int)
        p = np.asarray([r["probabilities"] for r in predictions], dtype=float)
        require(
            p.shape == (len(y), classes)
            and np.isfinite(p).all()
            and np.all((p >= 0) & (p <= 1))
            and np.allclose(p.sum(1), 1, rtol=0, atol=1e-7),
            "Invalid prediction probabilities",
        )
        predicted = p.argmax(axis=1)
        require(
            [r["prediction"] for r in predictions] == predicted.tolist(), "Predicted class differs from probabilities"
        )
        prior = np.bincount(train, minlength=classes) / len(train)
        truth = np.eye(classes)[y]
        brier = float(np.sum(np.mean((p - truth) ** 2, axis=0)))
        baseline = float(np.sum(np.mean((prior - truth) ** 2, axis=0)))
        recalls = [float(np.mean(predicted[y == i] == i)) if np.any(y == i) else None for i in range(classes)]
        result.update(
            class_counts=np.bincount(y, minlength=classes).tolist(),
            training_class_prior=prior.tolist(),
            brier=brier,
            training_prior_brier=baseline,
            brier_skill_vs_training_prior=None if baseline == 0 else 1 - brier / baseline,
            log_loss=float(np.mean(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)))),
            accuracy=float(np.mean(predicted == y)),
            class_recall=recalls,
            balanced_accuracy=float(np.mean([v for v in recalls if v is not None])),
            degenerate_baseline=baseline == 0,
        )
    return result
