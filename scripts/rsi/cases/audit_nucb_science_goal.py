"""NucB classification audit domain for the shared full-goal evidence replay.

Only host-side public-data reconstruction, independent refitting and mathematical
scoring occur. No external model is invoked and source experiments are unchanged.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import json
from types import SimpleNamespace

import numpy as np
from scripts.rsi.cases.audit_reap_science_goal import audit as audit_goal
from sklearn.metrics import average_precision_score, roc_auc_score

from agent.harness.contracts import digest
from agent.rsi import generated_classifier as classifier
from agent.rsi.cases.nucb_case import prepare_nucb_case

METRIC = "brier_skill_vs_training_prior"


def independent_metrics(model, rows, scope):
    predictions = classifier.predict(model, rows, scope)
    expected_inputs = [{"id": row["id"], "sequence": row["sequence"]} for row in rows]
    if [{"id": row["id"], "sequence": row["sequence"]} for row in predictions] != expected_inputs:
        raise ValueError("Prediction input order changed")
    p = np.asarray([row["probabilities"] for row in predictions])
    y = np.asarray([row["target"] for row in rows], dtype=int)
    k = len(scope["class_labels"])
    if (
        p.shape != (len(rows), k)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-7)
    ):
        raise ValueError("Invalid probabilities")
    truth = np.eye(k)[y]
    brier = float(np.sum(np.mean((p - truth) ** 2, axis=0)))
    prior_brier = float(np.sum(np.mean((truth - np.asarray(model["training_class_prior"])) ** 2, axis=0)))
    predicted = p.argmax(axis=1)
    recall = [float(np.mean(predicted[y == i] == i)) if (y == i).any() else None for i in range(k)]
    result = {
        "rows": len(rows),
        "class_counts": np.bincount(y, minlength=k).tolist(),
        "brier": brier,
        "training_prior_brier": prior_brier,
        METRIC: 1 - brier / prior_brier,
        "log_loss": float(np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)))),
        "accuracy": float(np.mean(predicted == y)),
        "class_recall": recall,
        "balanced_accuracy": float(np.mean([v for v in recall if v is not None])),
        "predictions_sha256": digest(predictions),
    }
    if k == 2:
        result.update(
            positive_prevalence=float(np.mean(y)),
            average_precision=float(average_precision_score(y, p[:, 1])),
            roc_auc=float(roc_auc_score(y, p[:, 1])),
        )
    return predictions, result


def compare_metrics(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Independent metric fields differ")
    for key, value in expected.items():
        observed = actual[key]
        if key in {"rows", "class_counts", "predictions_sha256"}:
            same = observed == value
        elif isinstance(value, list):
            same = len(observed) == len(value) and all(
                a is None if b is None else a is not None and np.isclose(a, b, rtol=1e-9, atol=1e-12)
                for a, b in zip(observed, value, strict=True)
            )
        else:
            same = bool(np.isclose(observed, value, rtol=1e-9, atol=1e-12))
        if not same:
            raise ValueError("Independent score differs: " + key)


def feedback_context(feedback):
    return (
        "\nPrevious independently replayed source and aggregate DEVELOPMENT acceptance feedback. "
        "Discuss the observed failure and revise the model source accordingly. These are adaptive "
        "development-selection observations, not confirmatory outcomes. No G4 labels, metrics, "
        "donor weights or prior-campaign model sources are provided.\n" + json.dumps(feedback)
    )


def audit(root, implementation_root, *, assets, rebuilt, defer_future_scoring=False):
    import sys

    sys.path.insert(0, str(implementation_root / "scripts/rsi"))
    from scripts.rsi.cases.run_nucb_science_goal import NucbModelGoal

    domain = SimpleNamespace(
        goal_factory=NucbModelGoal,
        rebuild=lambda protocol: prepare_nucb_case(assets, rebuilt),
        fit_candidate=classifier.fit_candidate,
        independent_metrics=independent_metrics,
        predict=classifier.predict,
        compare_metrics=compare_metrics,
        acceptance_metric=METRIC,
        feedback_context=feedback_context,
    )
    return audit_goal(root, implementation_root, domain=domain, defer_future_scoring=defer_future_scoring)
