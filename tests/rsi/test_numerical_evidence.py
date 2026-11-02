"""Per-row provenance, real isolated runtimes and tamper rejection at export/reuse."""

import copy
import json

import pytest
from test_generated_classifier import CODE as CLASSIFIER_CODE
from test_generated_classifier import ROWS as CLASSIFIER_ROWS
from test_generated_classifier import SCOPE as CLASSIFIER_SCOPE
from test_generated_features import CODE as FEATURE_CODE
from test_generated_features import ROWS as FEATURE_ROWS
from test_generated_features import SCOPE as FEATURE_SCOPE
from test_model_lifecycle import RECIPE, SCOPE, data

from agent.rsi.engine.code_sandbox import capture_code_runs, run_code
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.numerical_evidence import archive_assessment, verify_assessment
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json

SEQUENCE_CODE = """import numpy as np
AA = "ACDEFGHIKLMNPQRSTVWY"
def fit(sequences, targets, seed):
    x = np.array([[1., AA.index(s[0]), AA.index(s[1])] for s in sequences])
    return {"weights": np.linalg.lstsq(x, np.array(targets), rcond=None)[0].tolist()}
def predict(model, sequences):
    x = np.array([[1., AA.index(s[0]), AA.index(s[1])] for s in sequences])
    return (x @ np.array(model["weights"])).tolist()
"""


def setup(root, plugin_id):
    root.mkdir(exist_ok=True)
    plugin = get_model_plugin(plugin_id)
    if plugin_id == "generated_features.v1":
        train, validation, reuse = FEATURE_ROWS[:60], FEATURE_ROWS[60:80], FEATURE_ROWS[80:]
        scope, recipe = FEATURE_SCOPE, {"source_code": FEATURE_CODE, "seed": 19}
    elif plugin_id.startswith("generated_classifier"):
        train, validation, reuse = CLASSIFIER_ROWS[:60], CLASSIFIER_ROWS[60:80], CLASSIFIER_ROWS[80:90]
        scope, recipe = CLASSIFIER_SCOPE, {"source_code": CLASSIFIER_CODE, "seed": 19}
    else:
        train, validation, reuse = data()
        scope = SCOPE
        recipe = RECIPE if plugin_id == "aligned_site_ridge.v1" else {"source_code": SEQUENCE_CODE, "seed": 19}
    model = plugin.fit(train, scope, recipe)
    policy = {
        "schema": 1,
        "scope": scope,
        "plugin_id": plugin_id,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(train),
        "validation_sha256": digest(validation),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }
    controller = ModelLifecycle(root / "controller", policy)
    atomic_json(root / "model.json", model)
    mid = controller.submit(root / "model.json", session_id="develop", mode="science_agent")
    report = controller.assess(mid, training=train, validation=validation)
    return controller, mid, model, report, validation, reuse


@pytest.mark.parametrize(
    "plugin_id",
    [
        "aligned_site_ridge.v1",
        "generated_python.v1",
        "generated_features.v1",
        "generated_classifier.v1",
        "generated_classifier.large.v1",
    ],
)
def test_all_model_plugins_archive_the_actual_evaluated_rows_and_runtime(tmp_path, monkeypatch, plugin_id):
    monkeypatch.setenv("UNRELATED_PRIVATE_SECRET", "never-record-this")
    lc, mid, model, report, rows, reuse = setup(tmp_path, plugin_id)
    assert report["passed"]
    check = verify_assessment(lc.root, mid, model, lc.policy, report)
    assert check["status"] == "verified"
    assert check["predictions"] == get_model_plugin(plugin_id).predict(model, rows, lc.policy["scope"])
    witness = check["evidence"]
    assert witness["training_sha256"] == lc.policy["training_sha256"]
    assert witness["validation_sha256"] == lc.policy["validation_sha256"]
    assert witness["host_runtime"]["packages"]["numpy"]["version"]
    calls = witness["isolated_executions"]
    assert [c["operation"] for c in calls] == ([] if plugin_id == "aligned_site_ridge.v1" else ["fit", "predict"])
    for call in calls:
        assert call["source_sha256"] == model["source_sha256"]
        assert len(call["request_sha256"]) == len(call["result_sha256"]) == 64
        assert call["runtime"]["thread_settings"]["OPENBLAS_NUM_THREADS"] == "1"
        assert call["runtime"]["numeric_libraries"]
        assert call["runtime"]["python"]["executable"]["path"].startswith("/opt/venus-runtime/")
    assert "never-record-this" not in json.dumps(witness)
    assert all("target" not in row for row in check["predictions"])
    packet = lc.export(mid, tmp_path / "export")
    assert packet["assessment"]["numerical_evidence"] == report["numerical_evidence"]
    assert [p.name for p in (tmp_path / "export").iterdir()] == ["capability.json"]
    assert "predictions" not in packet and "predictions" not in packet["assessment"]["numerical_evidence"]
    inputs = [{k: v for k, v in row.items() if k != "target"} for row in reuse]
    predictions = get_model_plugin(plugin_id).predict(model, inputs, lc.policy["scope"])
    assert (
        lc.record_reuse(
            mid,
            session_id="fresh",
            mode="science_expert",
            inputs=inputs,
            predictions=predictions,
            scope=lc.policy["scope"],
        )["event"]
        == "reused"
    )


@pytest.mark.parametrize("damage", ["prediction", "runtime", "reference", "null", "symlink"])
def test_changed_evidence_blocks_both_export_and_reuse(tmp_path, damage):
    lc, mid, model, report, _, reuse = setup(tmp_path, "aligned_site_ridge.v1")
    folder = lc.root / "numerical_evidence/versions" / report["numerical_evidence"]["version_id"]
    if damage in {"prediction", "runtime", "symlink"}:
        assert folder.stat().st_mode & 0o222 == 0
        folder.chmod(0o755)  # Simulate external corruption only in this disposable test fixture.
        path = folder / ("predictions.json" if damage != "runtime" else "evidence.json")
        if damage == "symlink":
            other = tmp_path / "replacement.json"
            other.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(other)
        else:
            value = json.loads(path.read_text())
            if damage == "prediction":
                value[0]["prediction"] += 0.01
            else:
                value["host_runtime"]["python"]["version"] = "changed"
            atomic_json(path, value)
    else:
        if damage == "null":
            report["numerical_evidence"] = None
        else:
            report["numerical_evidence"]["evidence_sha256"] = "0" * 64
        atomic_json(lc.root / f"assessment_{mid}.json", report)
        state = lc._state()
        state["candidates"][mid]["assessment_sha256"] = digest(report)
        atomic_json(lc.root / "state.json", state)
    with pytest.raises(ValueError):
        lc.export(mid, tmp_path / "export")
    assert not (tmp_path / "export").exists()
    predictions = get_model_plugin("aligned_site_ridge.v1").predict(model, reuse, lc.policy["scope"])
    with pytest.raises(ValueError):
        lc.record_reuse(
            mid,
            session_id="fresh",
            mode="science_agent",
            inputs=reuse,
            predictions=predictions,
            scope=lc.policy["scope"],
        )
    assert not any(e["event"] == "reused" for e in lc._state()["events"])


def test_legacy_report_is_explicitly_identified_without_inventing_evidence(tmp_path):
    assert verify_assessment(tmp_path, "legacy", {}, {}, {"schema": 1}) == {"status": "legacy_without_per_row_evidence"}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("damage", ["order", "count", "hash"])
def test_archive_requires_same_order_and_values_as_the_evaluated_output(tmp_path, damage):
    lc, mid, model, report, rows, _ = setup(tmp_path, "aligned_site_ridge.v1")
    check = verify_assessment(lc.root, mid, model, lc.policy, report)
    predictions = copy.deepcopy(check["predictions"])
    metrics = copy.deepcopy(report["metrics"])
    if damage == "order":
        predictions.reverse()
        metrics["predictions_sha256"] = digest(predictions)
    elif damage == "count":
        predictions.pop()
    else:
        predictions[0]["prediction"] += 0.01
    with pytest.raises(ValueError, match="Per-row evidence"):
        archive_assessment(lc.root, mid, model, lc.policy, metrics, predictions, rows, [], {})


def test_isolated_execution_capture_does_not_leak_into_later_calls():
    request = {
        "operation": "fit",
        "input_kind": "sequences",
        "sequences": ["AA", "AC", "CA"],
        "targets": [0.0, 1.0, 3.0],
        "seed": 19,
    }
    with capture_code_runs() as records:
        run_code(SEQUENCE_CODE, request)
        with pytest.raises(RuntimeError, match="Nested"):
            with capture_code_runs():
                pass
    assert len(records) == 1
    result = run_code(SEQUENCE_CODE, request)
    assert set(result) == {"model"} and len(records) == 1


def test_corrupted_witness_cannot_enter_policy_revision_history(tmp_path):
    from scripts.rsi.evaluation import audit_improver_revision as auditor
    import run_improver_revision as producer
    from test_improver_history_preflight import donor

    root = donor(tmp_path)
    _, refs = producer.project_development_history([root])
    episode = root / "episodes/cycle_000"
    report = json.loads((episode / "acceptance.json").read_text())
    folder = episode / "controller/numerical_evidence/versions" / report["numerical_evidence"]["version_id"]
    folder.chmod(0o755)  # Disposable fixture, not a historical research record.
    path = folder / "predictions.json"
    values = json.loads(path.read_text())
    values[0]["prediction"] += 0.01
    atomic_json(path, values)
    with pytest.raises(ValueError):
        producer.project_development_history([root])
    with pytest.raises(ValueError):
        auditor.history_projection(refs, numerical=False)
