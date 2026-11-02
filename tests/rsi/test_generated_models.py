import json

import pytest

from agent.rsi.code_sandbox import CodeExecutionError, guard_source, run_code
from agent.rsi.generated_model import evaluate, fit_candidate, predict
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json

CODE = """import numpy as np
AA='ACDEFGHIKLMNPQRSTVWY'
def features(sequences):
    return np.array([[1.0]+[float(AA.index(a)) for a in seq] for seq in sequences])
def fit(sequences, targets, seed):
    x=features(sequences)
    weights=np.linalg.lstsq(x,np.asarray(targets),rcond=None)[0]
    return {'weights':weights.tolist()}
def predict(model, sequences):
    return (features(sequences) @ np.array(model['weights'])).tolist()
"""
SCOPE = {"protein": "test", "assay": "measured", "target": "activity", "unit": "AU", "sequence_length": 2}
TRAIN = [
    {"id": a + b, "sequence": a + b, "target": float(i * 2 + j)}
    for i, a in enumerate("ACDE")
    for j, b in enumerate("ACDE")
]
VALID = [{"id": "FG", "sequence": "FG", "target": 13.0}, {"id": "GF", "sequence": "GF", "target": 14.0}]


def test_generated_source_has_real_fit_predict_and_portable_state(tmp_path):
    recipe = {"source_code": CODE, "seed": 1}
    model = fit_candidate(TRAIN, SCOPE, recipe)
    assert model["development_type"] == "agent_generated_model_code"
    assert [r["prediction"] for r in predict(json.loads(json.dumps(model)), VALID, SCOPE)] == pytest.approx([13, 14])
    assert evaluate(model, VALID, SCOPE)["rmse"] < 1e-10
    plugin = get_model_plugin("generated_python.v1")
    policy = {
        "schema": 1,
        "scope": SCOPE,
        "plugin_id": plugin.id,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(TRAIN),
        "validation_sha256": digest(VALID),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }
    lc = ModelLifecycle(tmp_path / "registry", policy)
    atomic_json(tmp_path / "model.json", model)
    mid = lc.submit(tmp_path / "model.json", session_id="creation", mode="science_agent")
    assert lc.assess(mid, training=TRAIN, validation=VALID)["passed"]
    assert lc.export(mid, tmp_path / "export")["model"]["source_sha256"] == model["source_sha256"]


@pytest.mark.parametrize(
    "source",
    [
        CODE + "\nimport socket",
        CODE + "\nimport os",
        CODE + '\nopen("/etc/passwd")',
        CODE + '\neval("1")',
        CODE + "\nnp.__dict__",
        CODE.replace("def fit(", "def other("),
    ],
)
def test_unapproved_abi_or_imports_rejected(source):
    with pytest.raises(ValueError):
        guard_source(source)


def test_generated_code_cannot_read_host_files():
    source = CODE.replace("x=features(sequences)", "x=np.genfromtxt('/etc/passwd')")
    with pytest.raises(CodeExecutionError, match="failed"):
        run_code(source, {"operation": "fit", "sequences": ["AA"], "targets": [1.0]})


def test_infinite_generated_code_is_killed():
    source = "def fit(sequences, targets, seed):\n    while True: pass\ndef predict(model, sequences):\n    return []\n"
    with pytest.raises(CodeExecutionError, match="wall-clock"):
        run_code(source, {"operation": "fit", "sequences": ["AA"], "targets": [1.0]}, timeout_s=0.5)


def test_wrong_prediction_length_rejected():
    source = CODE.replace("return (features(sequences) @ np.array(model['weights'])).tolist()", "return []")
    model = fit_candidate(TRAIN, SCOPE, {"source_code": source, "seed": 1})
    with pytest.raises(ValueError, match="one finite number"):
        predict(model, VALID, SCOPE)


def test_frozen_code_transfers_but_fitted_model_scope_does_not():
    model = fit_candidate(TRAIN, SCOPE, {"source_code": CODE, "seed": 1})
    other = dict(SCOPE, protein="another protein")
    with pytest.raises(ValueError, match="applicability"):
        predict(model, VALID, other)
    refitted = fit_candidate(TRAIN, other, model["recipe"])
    assert refitted["source_sha256"] == model["source_sha256"]
    assert refitted["scope"] != model["scope"]
