import itertools
import json

import numpy as np
import pytest

from agent.rsi import generated_classifier as classifier
from agent.rsi.engine.experience import ModelExperienceStore
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.cases.nucb_case import reconstruct_sequence, temporal_partition
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json

SCOPE = {
    "protein": "fixture",
    "assay": "measured",
    "target": "class",
    "unit": "probability",
    "sequence_length": 3,
    "class_labels": ["low", "high"],
}
AA = "ACDEFGHIKLMNPQRSTVWY"
ROWS = [
    {"id": "".join(s), "sequence": "".join(s), "target": int(AA.index(s[0]) >= 10)}
    for s in itertools.product(AA, repeat=3)
]
ROWS.sort(key=lambda r: digest(r["sequence"]))
CODE = """import numpy as np
AA="ACDEFGHIKLMNPQRSTVWY"
def fit(sequences,targets,seed):
    x=np.array([AA.index(s[0]) for s in sequences],dtype=float)
    y=np.array(targets)
    return {"means":[float(x[y==k].mean()) for k in range(2)],"prior":[float((y==k).mean()) for k in range(2)]}
def predict(model,sequences):
    x=np.array([AA.index(s[0]) for s in sequences],dtype=float)
    logits=-((x[:,None]-np.array(model["means"])[None,:])**2)/16.0+np.log(np.array(model["prior"]))
    logits=logits-logits.max(axis=1,keepdims=True)
    p=np.exp(logits)
    return (p/p.sum(axis=1,keepdims=True)).tolist()
"""


def test_classifier_full_training_lifecycle_and_experience(tmp_path):
    train, val, future = ROWS[:720], ROWS[720:840], ROWS[840:900]
    recipe = {"source_code": CODE, "seed": 19}
    model = classifier.fit_candidate(train, SCOPE, recipe)
    p = classifier.predict(json.loads(json.dumps(model)), val, SCOPE)
    truth = np.eye(2)[[x["target"] for x in val]]
    expected = np.mean(np.sum((np.array([x["probabilities"] for x in p]) - truth) ** 2, axis=1))
    assert classifier.evaluate(model, val, SCOPE)["brier"] == pytest.approx(expected)
    plugin = get_model_plugin(classifier.PLUGIN_ID)
    policy = {
        "schema": 1,
        "scope": SCOPE,
        "plugin_id": plugin.id,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(train),
        "validation_sha256": digest(val),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }
    lc = ModelLifecycle(tmp_path / "registry", policy)
    atomic_json(tmp_path / "candidate.json", model)
    mid = lc.submit(tmp_path / "candidate.json", session_id="development", mode="science_agent")
    report = lc.assess(mid, training=train, validation=val)
    assert report["passed"] and report["acceptance_metric"] == classifier.ACCEPTANCE_METRIC
    inputs = [{k: r[k] for k in ("id", "sequence")} for r in future]
    preds = classifier.predict(model, inputs, SCOPE)
    assert (
        lc.record_reuse(
            mid, session_id="new-session", mode="science_expert", inputs=inputs, predictions=preds, scope=SCOPE
        )["rows"]
        == 60
    )
    experience = ModelExperienceStore(tmp_path / "experience")
    eid = experience.publish(lc, mid, summary="Training-estimated class centroids; full source retained")
    assert experience.project("source", [eid])["entries"][0]["donor_scope"]["class_labels"] == ["low", "high"]
    assert get_model_plugin("generated_python.v1").acceptance_metric == "mse_skill_vs_training_mean"


@pytest.mark.parametrize(
    "values", [[[0.1, 0.2]], [[-0.1, 1.1]], [[float("nan"), 0]], [[True, False]], [0.5], [[0.2, 0.3, 0.5]]]
)
def test_invalid_probability_contract_rejected(values):
    with pytest.raises(ValueError):
        classifier.validate_probabilities(values, 1, 2)


def test_class_order_scope_overlap_and_integer_labels_enforced():
    model = classifier.fit_candidate(ROWS[:30], SCOPE, {"source_code": CODE, "seed": 1})
    with pytest.raises(ValueError, match="overlap"):
        classifier.evaluate(model, ROWS[:2], SCOPE)
    with pytest.raises(ValueError, match="applicability"):
        classifier.predict(model, ROWS[30:32], dict(SCOPE, class_labels=["high", "low"]))
    with pytest.raises(ValueError, match="integer"):
        classifier.validate_rows([dict(ROWS[0], target=True)], SCOPE, labels=True)


def test_historical_training_does_not_use_later_duplicate_labels():
    def row(seq, label):
        return {"id": seq, "sequence": seq, "target": label}

    gens = {
        "g1": [row("AAA", 0), row("CCC", 0)],
        "g2": [row("AAA", 1), row("DDD", 0)],
        "g3": [row("AAA", 0), row("EEE", 1)],
        "g4": [row("DDD", 1), row("EEE", 0), row("FFF", 1)],
    }
    train, val, future = temporal_partition(gens)
    assert {r["sequence"]: r["target"] for r in train} == {"AAA": 1, "CCC": 0, "DDD": 0}
    assert val == [row("EEE", 1)] and future == [row("FFF", 1)]
    assert reconstruct_sequence("ACDE", [["C", 2, "F"]]) == "AFDE"
    with pytest.raises(ValueError):
        reconstruct_sequence("ACDE", [["A", 2, "F"]])
    with pytest.raises(ValueError):
        reconstruct_sequence("ACDE", [["C", 2, "F"], ["C", 2, "G"]])


def test_probability_artifact_contract_rejects_plausible_class_with_bad_distribution(tmp_path):
    from agent.rsi.engine.tool_contracts import validate_generated_result

    work, caps = tmp_path / "work", tmp_path / "caps"
    work.mkdir()
    (caps / "registered").mkdir(parents=True)
    rows = [{"id": "x", "sequence": "ACD"}]
    model = {"scope": SCOPE, "plugin_id": classifier.PLUGIN_ID}
    packet = {"model": model, "model_sha256": digest(model), "model_id": "registered"}
    atomic_json(caps / "registered/capability.json", packet)
    atomic_json(
        caps / "task.json",
        {
            "phase": "reuse",
            "scope": SCOPE,
            "plugin_id": classifier.PLUGIN_ID,
            "inputs": {"inputs.json": digest(rows)},
            "capability_sha256": digest(packet),
        },
    )
    atomic_json(work / "inputs.json", rows)
    for probabilities, prediction, valid in [([0.2, 0.8], 1, True), ([0.2, 0.2], 1, False), ([0.2, 0.8], 0, False)]:
        predictions = [{**rows[0], "probabilities": probabilities, "prediction": prediction}]
        atomic_json(work / "predictions.json", predictions)
        value = {
            "artifact": "predictions.json",
            "model_id": "registered",
            "rows": 1,
            "prediction_sha256": digest(predictions),
        }
        if valid:
            validate_generated_result("predict_with_generated_model", value, workspace=work, capability=caps)
        else:
            with pytest.raises(ValueError):
                validate_generated_result("predict_with_generated_model", value, workspace=work, capability=caps)
