"""Terminal success must not open future labels during a pre-barrier audit."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases.audit_reap_science_goal import audit_completed_goal

from agent.harness.contracts import digest
from agent.harness.host import write_json


class ClosedFuture:
    def __getitem__(self, key):
        raise AssertionError("Pre-barrier audit opened future labels")


def completed_fixture(root):
    (root / "reuse/workspace").mkdir(parents=True)
    (root / "experience/entries").mkdir(parents=True)
    inputs = [{"id": "new-sequence", "sequence": "ACDE"}]
    predictions = [{**inputs[0], "probabilities": [0.2, 0.8], "prediction": 1}]
    code = "fixture source"
    source_sha = hashlib.sha256(code.encode()).hexdigest()
    model = {"recipe": {"source_code": code}}
    scope = {"dataset": "fixture"}
    state = {"progress": {"reuse_receipt": "reuse/reuse_receipt.json"}}
    write_json(root / "reuse/workspace/predictions.json", predictions)
    write_json(
        root / "reuse/reuse_receipt.json", {"input_sha256": digest(inputs), "prediction_sha256": digest(predictions)}
    )
    entry = {"model_id": "model", "source_code": code, "source_sha256": source_sha, "donor_scope": scope, "parents": []}
    identity = digest(entry)
    write_json(root / "experience/entries" / (identity + ".json"), entry)
    write_json(root / "registered_experience.json", {"experience_id": identity, "source_sha256": source_sha})
    lifecycle = SimpleNamespace(_model=lambda mid: model)
    plugin = SimpleNamespace(
        lifecycle=lambda state: (lifecycle, "model"), inference_inputs=lambda: inputs, scope=scope, data=ClosedFuture()
    )
    calls = []

    def predict(received_model, rows, received_scope):
        assert rows == inputs and all(set(row) == {"id", "sequence"} for row in rows)
        assert received_model == model and received_scope == scope
        calls.append("predict_without_labels")
        return predictions

    def measure(*args):
        calls.append("scoring")
        raise AssertionError("Future scoring was called")

    return (
        state,
        plugin,
        inputs,
        predictions,
        calls,
        {"measure": measure, "compare": lambda a, b: None, "predict": predict},
    )


def test_completed_goal_replays_reuse_without_future_labels_or_metric_calls(tmp_path):
    state, plugin, inputs, predictions, calls, helpers = completed_fixture(tmp_path)
    result = audit_completed_goal(tmp_path, state, plugin, defer_future_scoring=True, **helpers)
    assert calls == ["predict_without_labels"]
    assert result == {"future_scoring_deferred": True, "reuse_replayed": True, "source_experience_verified": True}
    assert not (tmp_path / "future_metrics.json").exists()


def test_default_audit_rejects_missing_published_score_before_computing_it(tmp_path):
    state, plugin, inputs, predictions, calls, helpers = completed_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="future_metrics"):
        audit_completed_goal(tmp_path, state, plugin, defer_future_scoring=False, **helpers)
    assert calls == []


@pytest.mark.parametrize("filename", ["future_metrics.json", "selected_future_metrics.json"])
def test_deferred_audit_does_not_silently_accept_an_opened_holdout(tmp_path, filename):
    state, plugin, inputs, predictions, calls, helpers = completed_fixture(tmp_path)
    write_json(tmp_path / filename, {"already_scored": True})
    with pytest.raises(ValueError, match="published future score"):
        audit_completed_goal(tmp_path, state, plugin, defer_future_scoring=True, **helpers)
    assert calls == []


@pytest.mark.parametrize("mutation", ["predictions", "receipt", "source"])
def test_deferred_audit_still_rejects_forged_reuse_or_source(tmp_path, mutation):
    state, plugin, inputs, predictions, calls, helpers = completed_fixture(tmp_path)
    if mutation == "predictions":
        write_json(tmp_path / "reuse/workspace/predictions.json", [])
    elif mutation == "receipt":
        write_json(
            tmp_path / "reuse/reuse_receipt.json", {"input_sha256": "wrong", "prediction_sha256": digest(predictions)}
        )
    else:
        packet = json.loads((tmp_path / "registered_experience.json").read_text())
        packet["source_sha256"] = "wrong"
        write_json(tmp_path / "registered_experience.json", packet)
    with pytest.raises(ValueError):
        audit_completed_goal(tmp_path, state, plugin, defer_future_scoring=True, **helpers)


@pytest.mark.parametrize("field", ["target", "label", "undeclared"])
def test_reuse_prediction_path_cannot_receive_label_fields(tmp_path, field):
    state, plugin, inputs, predictions, calls, helpers = completed_fixture(tmp_path)
    inputs[0][field] = 1
    with pytest.raises(ValueError, match="label or undeclared"):
        audit_completed_goal(tmp_path, state, plugin, defer_future_scoring=True, **helpers)
    assert calls == []


def test_postbarrier_audit_recomputes_and_compares_the_published_score(tmp_path):
    state, plugin, inputs, predictions, calls, helpers = completed_fixture(tmp_path)
    plugin.data = {"future": [{**inputs[0], "target": 1}]}
    metrics = {"brier": 0.08}
    write_json(tmp_path / "future_metrics.json", metrics)

    def measure(model, rows, scope):
        assert rows[0]["target"] == 1
        calls.append("scoring")
        return predictions, metrics

    def compare(actual, expected):
        assert actual == expected == metrics
        calls.append("compared")

    helpers.update(measure=measure, compare=compare)
    result = audit_completed_goal(tmp_path, state, plugin, defer_future_scoring=False, **helpers)
    assert result["future_metrics"] == metrics and result["reuse_replayed"]
    assert calls == ["scoring", "compared"]
