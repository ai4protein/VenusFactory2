"""Fixed sequence controls for multilabel tasks, portable into isolated training.

These are platform reference algorithms, not Scientist-generated models.
Unknown residues retain their own composition bin; no sequence is truncated.
"""

import math
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "sequence-multilabel-reference.v1"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = frozenset(AMINO_ACIDS + "BXZJUO")
REFERENCE_METHOD = "aac_logistic"
SEQUENCE_COLUMN = "sequence"
OUTPUT_FIELD = "value"


def sequence_features(sequence, terminal=False):
    if not isinstance(sequence, str) or not sequence or not set(sequence) <= ALPHABET:
        raise ValueError("Provide a nonempty uppercase amino-acid sequence (including BXZJUO)")

    def composition(part):
        counts = [part.count(residue) / len(part) for residue in AMINO_ACIDS]
        return counts + [sum(residue not in AMINO_ACIDS for residue in part) / len(part)]

    values = composition(sequence) + [math.log1p(len(sequence))]
    if terminal:
        values += composition(sequence[:50]) + composition(sequence[-50:])
    return values


def fit(training, seed):
    if training.get("task") != "multilabel":
        raise ValueError("This reference requires independent multilabel targets")
    if REFERENCE_METHOD not in {"training_prevalence", "aac_logistic", "terminal_logistic"}:
        raise ValueError("Unknown multilabel reference")
    labels = training["labels"]
    rows = training["train"]
    if not rows or not labels or len(labels) != len(set(labels)):
        raise ValueError("Provide training observations and distinct output labels")
    terminal = REFERENCE_METHOD == "terminal_logistic"
    x = np.asarray([sequence_features(row["inputs"][SEQUENCE_COLUMN], terminal) for row in rows])
    fitted = {}
    for label in labels:
        observed = [i for i, row in enumerate(rows) if row["target"][label] is not None]
        values = [rows[i]["target"][label] for i in observed]
        if not values or any(type(value) not in (int, float) or value not in (0, 1) for value in values):
            raise ValueError("Each label needs observed binary training targets; null is unobserved")
        if REFERENCE_METHOD == "training_prevalence" or len(set(values)) == 1:
            fitted[label] = float(sum(values) / len(values))
        else:
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x[observed], np.asarray(values))
            fitted[label] = model
    return {"labels": list(labels), "models": fitted, "terminal": terminal}


def predict(model, inputs):
    x = np.asarray([sequence_features(inputs[SEQUENCE_COLUMN], model["terminal"])])
    probabilities = {}
    for label in model["labels"]:
        fitted = model["models"][label]
        probability = (
            fitted if isinstance(fitted, float) else float(fitted.predict_proba(x)[0, list(fitted.classes_).index(1)])
        )
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Nonfinite or out-of-range reference probability")
        probabilities[label] = probability
    return {OUTPUT_FIELD: probabilities}
