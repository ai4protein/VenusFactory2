"""Real isolated computations on synthetic contextual fixtures, not acquired models."""

from copy import deepcopy

import numpy as np
import pytest

from agent.rsi import generated_context as api
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.numerical_evidence import archive_assessment
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json

SCOPE = {
    "protein": "Synthetic plant protein",
    "assay": "synthetic_response",
    "target": "relative_response",
    "unit": "dimensionless",
    "context": {"material": "synthetic"},
    "covariates": [{"name": "temperature", "unit": "degC", "minimum": 20.0, "maximum": 90.0}],
    "grouping": {"level": "biological_replicate", "evidence_sha256": "a" * 64},
}
ROWS = [
    {
        "id": f"g{group}_t{point}",
        "protein": SCOPE["protein"],
        "group": f"g{group}",
        "values": [30.0 + 5 * point],
        "target": 1.5 - 0.01 * (30 + 5 * point),
    }
    for group in range(6)
    for point in range(10)
]
CODE = """import numpy as np
def fit(features, targets, seed):
    x = np.column_stack([np.ones(len(features)), np.asarray(features)])
    weights = np.linalg.lstsq(x, np.asarray(targets), rcond=None)[0]
    return {'weights': weights.tolist()}
def predict(model, features):
    x = np.column_stack([np.ones(len(features)), np.asarray(features)])
    return (x @ np.asarray(model['weights'])).tolist()
"""


def fitted():
    return api.fit_candidate(ROWS[:40], SCOPE, {"source_code": CODE, "seed": 0})


def test_real_lifecycle_acceptance_export_and_fresh_group_reuse(tmp_path):
    model = fitted()
    assert "training_sequences" not in model
    plugin = get_model_plugin(api.PLUGIN_ID)
    policy = {
        "schema": 1,
        "scope": SCOPE,
        "plugin_id": plugin.id,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(ROWS[:40]),
        "validation_sha256": digest(ROWS[40:50]),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }
    lc = ModelLifecycle(tmp_path / "registry", policy)
    path = tmp_path / "candidate.json"
    atomic_json(path, model)
    mid = lc.submit(path, session_id="develop", mode="science_expert")
    report = lc.assess(mid, training=ROWS[:40], validation=ROWS[40:50])
    assert report["passed"] and report["training_replayed"]
    assert report["metrics"]["independent_groups"] == 1
    assert report["metrics"]["rmse"] < 1e-12
    packet = lc.export(mid, tmp_path / "packet")
    assert packet["model"]["plugin_id"] == api.PLUGIN_ID
    inputs = api.inference_inputs(ROWS[50:])
    predictions = api.predict(packet["model"], inputs, SCOPE)
    receipt = lc.record_reuse(
        mid, session_id="fresh", mode="science_agent", inputs=inputs, predictions=predictions, scope=SCOPE
    )
    assert receipt["rows"] == 10
    changed = deepcopy(inputs)
    changed[0]["values"][0] += 1
    with pytest.raises(ValueError, match="does not match"):
        lc.record_reuse(
            mid, session_id="other", mode="science_agent", inputs=changed, predictions=predictions, scope=SCOPE
        )
    for row in changed:
        row["group"] = "g0"
    with pytest.raises(ValueError, match="independent-group overlap"):
        lc.record_reuse(
            mid, session_id="other", mode="science_agent", inputs=changed, predictions=predictions, scope=SCOPE
        )


def test_grouped_feedback_keeps_technical_repeats_together():
    rows = deepcopy(ROWS[:40])
    for row in rows[10:20]:
        row["group"] = "g0"  # Another instrument run of the same physical sample.
    train, feedback = api.development_split(rows)
    assert not {r["group"] for r in train} & {r["group"] for r in feedback}
    assert len([r for r in train + feedback if r["group"] == "g0"]) == 20
    assert api.development_split(rows[::-1]) == (train, feedback)
    with pytest.raises(ValueError, match="two independent groups"):
        api.development_split(rows[:10])


@pytest.mark.parametrize("change", ["unit", "order", "protein", "context", "grouping"])
def test_applicability_cannot_silently_change(change):
    model = fitted()
    other = deepcopy(SCOPE)
    if change == "unit":
        other["covariates"][0]["unit"] = "kelvin"
    elif change == "order":
        other["covariates"][0]["name"] = "pH"
    elif change == "grouping":
        other["grouping"]["level"] = "workflow_replicate"
    elif change == "context":
        other["context"]["material"] = "another material"
    else:
        other["protein"] = "Another protein"
    with pytest.raises(ValueError, match="mismatch"):
        api.predict(model, api.inference_inputs(ROWS[40:]), other)


@pytest.mark.parametrize("bad", [[float("nan")], [True], [99.0], [], "not_a_vector"])
def test_reject_invalid_covariates(bad):
    row = deepcopy(ROWS[0])
    row["values"] = bad
    with pytest.raises(ValueError):
        api.validate_rows([row], SCOPE, labels=True)


def test_group_or_observation_reuse_rejected_even_if_values_change():
    model = fitted()
    rows = deepcopy(ROWS[40:50])
    rows[0]["group"] = "g0"
    with pytest.raises(ValueError, match="independent-group overlap"):
        api.evaluate(model, rows, SCOPE)
    rows = deepcopy(ROWS[40:50])
    rows[0]["id"] = "g0_t0"
    with pytest.raises(ValueError, match="observation overlap"):
        api.evaluate(model, rows, SCOPE)


def test_values_can_repeat_in_independent_samples_and_query_labels_are_ignored():
    model = fitted()
    first = api.predict(model, ROWS[40:50], SCOPE)
    changed = deepcopy(ROWS[40:50])
    for row in changed:
        row["target"] += 100
    assert api.predict(model, changed, SCOPE) == first
    assert api.evaluate(model, changed, SCOPE)["rmse"] > 99


def test_wrong_prediction_shape_and_nonnumeric_values_rejected():
    for expression in ["[]", "[True] * len(features)"]:
        code = CODE.replace("(x @ np.asarray(model['weights'])).tolist()", expression)
        model = api.fit_candidate(ROWS[:40], SCOPE, {"source_code": code, "seed": 0})
        with pytest.raises(ValueError, match="finite numeric prediction"):
            api.predict(model, api.inference_inputs(ROWS[40:50]), SCOPE)


@pytest.mark.parametrize("field", ["id", "protein", "group", "values_sha256"])
def test_numerical_evidence_binds_every_input_identity(tmp_path, field):
    model = fitted()
    metrics, predictions = api.evaluate_with_predictions(model, ROWS[40:50], SCOPE)
    predictions[0][field] = "tampered"
    metrics["predictions_sha256"] = digest(predictions)
    with pytest.raises(ValueError, match="Per-row evidence"):
        archive_assessment(
            tmp_path,
            "fixture",
            model,
            {"validation_sha256": digest(ROWS[40:50])},
            metrics,
            predictions,
            ROWS[40:50],
            [],
            {},
        )


def test_fit_is_deterministic_and_matches_independent_linear_solution():
    model = fitted()
    assert model == fitted()
    actual = [r["prediction"] for r in api.predict(model, ROWS[40:50], SCOPE)]
    np.testing.assert_allclose(actual, [r["target"] for r in ROWS[40:50]], atol=1e-12)
