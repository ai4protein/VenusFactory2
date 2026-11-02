"""Provenance-bound reuse of generated classifier source on a new protein.

A source recipe is portable; a fitted model remains restricted to its original
scientific scope. This helper never promotes target weights without assessment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protein_model import digest
from .storage import sha256_file


def registered_source(donor_run, donor_audit):
    donor_run, donor_audit = Path(donor_run), Path(donor_audit)
    report = json.loads((donor_run / "report.json").read_text())
    audit = json.loads(donor_audit.read_text())
    if (
        report.get("status") != "campaign_completed"
        or audit.get("status") != "passed"
        or audit.get("source_report_sha256") != sha256_file(donor_run / "report.json")
    ):
        raise ValueError("A completed, independently audited donor study is required")
    eligible = [
        arm
        for arm in report["arms"]
        if arm.get("status") == "completed" and arm.get("assessment", {}).get("passed") is True
    ]
    if len(eligible) != 1:
        raise ValueError("This protocol requires exactly one eligible donor; no outcome-based donor search")
    arm = eligible[0]
    verified = [a for a in audit["arms"] if a["id"] == arm["id"]]
    if (
        len(verified) != 1
        or verified[0].get("status") != "generation_acceptance_registration_and_fresh_session_reuse_verified"
    ):
        raise ValueError("Donor lifecycle lacks independent verification")
    folder = donor_run / ("arm_{:03d}".format(arm["id"]))
    state = json.loads((folder / "controller/state.json").read_text())
    mid = arm["model_id"]
    if mid not in state["published"] or state["candidates"][mid]["status"] != "registered":
        raise ValueError("Donor model is not currently registered")
    path = folder / "develop/workspace/candidate.json"
    if path.is_symlink():
        raise ValueError("Donor artifact must not be a symlink")
    model = json.loads(path.read_text())
    source = model["recipe"]["source_code"]
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != model["source_sha256"] or source_hash != verified[0]["source_sha256"]:
        raise ValueError("Donor source identity changed")
    artifact = json.loads((folder / "controller/artifacts/versions" / mid / "model.json").read_text())
    if digest(artifact) != digest(model):
        raise ValueError("Donor workspace differs from registered artifact")
    record = {
        "donor_run": str(donor_run),
        "donor_model_id": mid,
        "source_sha256": source_hash,
        "donor_scope": model["scope"],
        "source_report_sha256": sha256_file(donor_run / "report.json"),
        "source_audit_sha256": sha256_file(donor_audit),
        "selection": "only independently verified registered donor",
        "transfer_unit": "exact Python source only; target weights are fitted afresh",
        "target_selection_uses_donor_future_scores": False,
    }
    return source, record, model


def refit_recipe(source, seed=314159):
    if not isinstance(source, str) or type(seed) is not int:
        raise ValueError("Expected source text and an integer fitting seed")
    return {"source_code": source, "seed": seed}


# An operator-fixed reference, not an agent-authored model or paper reproduction.
LOGISTIC_REFERENCE = """import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

def fit(sequences, targets, seed):
    x = np.array([list(s) for s in sequences])
    y = np.asarray(targets, dtype=int)
    if set(y.tolist()) != {0, 1}:
        raise ValueError('Binary reference requires classes zero and one')
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
    encoded = encoder.fit_transform(x)
    estimator = LogisticRegression(C=1.0, solver='liblinear', max_iter=500, tol=1e-4, random_state=seed)
    estimator.fit(encoded, y)
    if int(estimator.n_iter_[0]) >= 500:
        raise ValueError('Fixed reference did not converge within its declared limit')
    return {'categories': [a.tolist() for a in encoder.categories_],
            'coefficients': estimator.coef_[0].tolist(), 'intercept': float(estimator.intercept_[0]),
            'iterations': int(estimator.n_iter_[0])}

def predict(model, sequences):
    score = np.full(len(sequences), model['intercept'], dtype=float)
    weights = np.asarray(model['coefficients'])
    offset = 0
    for position, categories in enumerate(model['categories']):
        lookup = {c: i for i, c in enumerate(categories)}
        for row, sequence in enumerate(sequences):
            code = lookup.get(sequence[position], -1)
            if code >= 0:
                score[row] += weights[offset + code]
        offset += len(categories)
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -700, 700)))
    return np.column_stack([1 - probability, probability]).tolist()
"""
