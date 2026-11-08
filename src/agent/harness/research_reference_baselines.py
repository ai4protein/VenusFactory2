"""On-demand, portable adapters for the repository's fixed sequence controls."""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

METHODS = {
    "aac_ridge": "Ridge regression on amino-acid composition and log sequence length.",
    "aac_random_forest": "Random forest on the same composition and length features.",
    "aac_svr": "Support-vector regression on the same composition and length features.",
    "training_mean": "Training-target mean reference without predictive sequence effects.",
}
REFERENCE = Path(__file__).resolve().parents[1] / "rsi/engine/traditional_baselines.py"
IMPORTS = """import math
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
"""

BRIDGE = """
REFERENCE_METHOD = __METHOD__
SEQUENCE_COLUMN = __SEQUENCE__
OUTPUT_FIELD = __OUTPUT__

def fit(training, seed):
    if training.get("task") != "regression":
        raise ValueError("This reference supports measured regression targets only")
    rows = training["train"]
    matrix = features([{"sequence": row["inputs"][SEQUENCE_COLUMN]} for row in rows])
    targets = np.asarray([row["target"] for row in rows], dtype=float)
    if not np.all(np.isfinite(targets)):
        raise ValueError("Training targets must be finite")
    location, scale = float(targets.mean()), float(targets.std())
    scale = scale if scale > 0 else 1.0
    estimator = estimators()[REFERENCE_METHOD]
    seeds = {key: seed for key in estimator.get_params(deep=True) if key.endswith("random_state")}
    if seeds:
        estimator.set_params(**seeds)
    estimator.fit(matrix, (targets - location) / scale)
    return {"estimator": estimator, "location": location, "scale": scale}

def predict(model, inputs):
    matrix = features([{"sequence": inputs[SEQUENCE_COLUMN]}])
    value = float(model["estimator"].predict(matrix)[0] * model["scale"] + model["location"])
    if not math.isfinite(value):
        raise ValueError("Reference prediction is nonfinite")
    return {OUTPUT_FIELD: value}
"""


def reference_baseline(reference="", sequence_column="sequence", output_field="value"):
    if not isinstance(reference, str):
        raise ValueError("Choose a listed reference baseline")
    if not reference:
        return {"kind": "platform_reference", "task": "regression",
                "scope": "Canonical amino-acid sequences; composition and log length only. Assay covariates and residue positions are not represented.",
                "references": [{"id": key, "description": text} for key, text in METHODS.items()],
                "next_step": "Select a reference ID; provide output_file to save runnable baseline code, or omit it to read the complete source."}
    if not isinstance(reference, str) or reference not in METHODS:
        raise ValueError("Choose a listed reference baseline")
    for field in (sequence_column, output_field):
        if not isinstance(field, str) or not field.strip() or len(field) > 80 or any(ord(c) < 32 for c in field):
            raise ValueError("Use a nonempty column/output name of at most 80 characters")
    raw = REFERENCE.read_bytes()
    text = raw.decode("utf-8")
    nodes = ast.parse(text).body
    functions = {n.name: ast.get_source_segment(text, n) for n in nodes if isinstance(n, ast.FunctionDef)}
    assignments = {n.targets[0].id: n for n in nodes
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    if not {"features", "estimators"} <= functions.keys() or not {"AMINO_ACIDS", "VERSION"} <= assignments.keys():
        raise ValueError("Reference implementation changed; inspect its adapter")
    constants = ast.get_source_segment(text, assignments["AMINO_ACIDS"])
    version = ast.literal_eval(assignments["VERSION"].value)
    replacements = {"__METHOD__": repr(reference), "__SEQUENCE__": repr(sequence_column), "__OUTPUT__": repr(output_field)}
    bridge = re.sub("__METHOD__|__SEQUENCE__|__OUTPUT__", lambda match: replacements[match.group()], BRIDGE)
    upstream_hash = hashlib.sha256(raw).hexdigest()
    header = "# Platform reference: " + reference + " (" + version + ")\n"
    header += "# Upstream: src/agent/rsi/engine/traditional_baselines.py; sha256=" + upstream_hash + "\n"
    source = header + IMPORTS + "\n" + constants + "\n\n" + functions["features"] + "\n\n" + functions["estimators"] + "\n" + bridge
    ast.parse(source)
    return {"kind": "platform_reference", "reference_id": reference, "version": version,
            "description": METHODS[reference], "task": "regression",
            "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "upstream": "src/agent/rsi/engine/traditional_baselines.py",
            "upstream_sha256": upstream_hash,
            "input_field": sequence_column, "output_field": output_field,
            "scope": "Canonical amino-acid sequences only; sequence composition cannot resolve all positional or mechanistic effects.",
            "provenance": "A fixed platform reference implementation, not a Scientist-created method or a fitted model.",
            "next_step": "Save this source and fit with train_research_model(variant='baseline'). Develop and measure a justified candidate with that baseline's parent_experiment_id; retain the reference provenance in the report."}
