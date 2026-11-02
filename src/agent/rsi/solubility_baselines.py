"""Published SWI scores with an explicitly training-only regression calibration.

The published binary-solubility probability is not a measured soluble fraction.
This module uses the original score and names its optional affine calibration.
"""

from __future__ import annotations

import math

import numpy as np

from .protein_model import digest

WEIGHTS = {
    "A": 0.8356471476582918,
    "C": 0.5208088354857734,
    "E": 0.9876987431418378,
    "D": 0.9079044671339564,
    "G": 0.7997168496420723,
    "F": 0.5849790194237692,
    "I": 0.6784124413866582,
    "H": 0.8947913996466419,
    "K": 0.9267104557513497,
    "M": 0.6296623675420369,
    "L": 0.6554221515081433,
    "N": 0.8597433107431216,
    "Q": 0.789434648348208,
    "P": 0.8235328714705341,
    "S": 0.7440908318492778,
    "R": 0.7712466317693457,
    "T": 0.8096922697856334,
    "W": 0.6374678690957594,
    "V": 0.7357837119163659,
    "Y": 0.6112801822947587,
}
REFERENCE = {
    "doi": "10.1093/bioinformatics/btaa578",
    "source_url": "https://github.com/Gardner-BinfLab/SoDoPE_paper_2020/blob/master/SWI/swi.py",
    "source_git_blob_sha1": "d83ef5906ffd03befb303f4985507bc92b3fee09",
    "source_sha256": "6e9c0fdb18f65e20c885f5d59b7f5b2cebf557f8c471adb74713be1848e658f3",
}


def swi(sequence):
    """Return the published mean-weight score, without editing the sequence."""
    if not isinstance(sequence, str) or not sequence or any(aa not in WEIGHTS for aa in sequence):
        raise ValueError("SWI requires a nonempty canonical uppercase amino-acid sequence")
    return math.fsum(WEIGHTS[aa] for aa in sequence) / len(sequence)


def _scores(rows, *, labelled):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nonempty sequence rows required")
    fields = {"id", "sequence", "target"} if labelled else {"id", "sequence"}
    if any(not isinstance(row, dict) or set(row) != fields for row in rows):
        raise ValueError("Training must be labelled and query must contain only id and sequence")
    ids = [row["id"] for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Distinct nonempty observation identities required")
    return np.asarray([swi(row["sequence"]) for row in rows], dtype=float)


def predict_swi_control(training, query):
    """Use only training labels to map the frozen SWI score into endpoint units.

    No feature or parameter is selected from query labels. No probability-to-
    percentage conversion or clipping is applied. Constant scores use the
    training mean, the identifiable intercept-only least-squares solution.
    """
    x = _scores(training, labelled=True)
    z = _scores(query, labelled=False)
    if any(type(row["target"]) not in (int, float) for row in training):
        raise ValueError("Numeric training labels required")
    y = np.asarray([row["target"] for row in training], dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("Finite training labels required")
    location, target_mean = float(x.mean()), float(y.mean())
    centered = x - location
    constant = float(np.ptp(x)) <= 1e-12
    slope = 0.0 if constant else float(np.dot(centered, y - target_mean) / np.dot(centered, centered))
    intercept = target_mean - slope * location
    predicted = target_mean + slope * (z - location)
    if not np.all(np.isfinite(predicted)) or not all(math.isfinite(v) for v in [slope, intercept]):
        raise ValueError("Nonfinite SWI calibration")
    return {
        "method": "published_swi_training_ols",
        "reference": dict(REFERENCE),
        "weights_sha256": digest(WEIGHTS),
        "fit_partition": "training",
        "training_sha256": digest(training),
        "query_sha256": digest(query),
        "training_rows": len(training),
        "calibration": {
            "slope": slope,
            "intercept": intercept,
            "score_center": location,
            "target_center": target_mean,
            "constant_score_fallback": constant,
            "clipping": False,
            "hyperparameter_search": False,
        },
        "raw_scores": [{"id": row["id"], "score": float(value)} for row, value in zip(query, z, strict=True)],
        "predictions": [
            {"id": row["id"], "prediction": float(value)} for row, value in zip(query, predicted, strict=True)
        ],
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
