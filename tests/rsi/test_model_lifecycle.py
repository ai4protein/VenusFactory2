import json

import pytest

from agent.research_modes import get_research_mode
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.protein_model import PLUGIN_ID, digest, fit_candidate, predict, validate_rows
from agent.rsi.engine.runtime import safe_approval
from agent.rsi.engine.storage import atomic_json

SCOPE = {"protein": "fixture", "assay": "assay1", "target": "activity", "unit": "AU", "sequence_length": 2}
RECIPE = {"encoding": "onehot", "alphas": [0.001, 0.01]}


def data():
    rows = [
        {"id": a + b, "sequence": a + b, "target": float(i * 3 + j)}
        for i, a in enumerate("ACDEFG")
        for j, b in enumerate("ACDEFG")
    ]
    val = [r for n, r in enumerate(rows) if n % 7 == 0]
    reuse = [r for n, r in enumerate(rows) if n % 7 == 1]
    train = [r for n, r in enumerate(rows) if n % 7 not in (0, 1)]
    return train, val, reuse


def setup(tmp_path, minimum=0):
    train, val, reuse = data()
    plugin = get_model_plugin(PLUGIN_ID)
    policy = {
        "schema": 1,
        "scope": SCOPE,
        "plugin_id": PLUGIN_ID,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(train),
        "validation_sha256": digest(val),
        "minimum_skill": minimum,
        "max_candidates": 1,
    }
    lifecycle = ModelLifecycle(tmp_path / "controller", policy)
    model = fit_candidate(train, SCOPE, RECIPE)
    artifact = tmp_path / "model.json"
    atomic_json(artifact, model)
    mid = lifecycle.submit(artifact, session_id="develop", mode="science_agent")
    return lifecycle, mid, model, train, val, reuse


def test_real_lifecycle_blocks_unvalidated_then_reuses_in_fresh_mode(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path)
    with pytest.raises(ValueError, match="validated"):
        lc.export(mid, tmp_path / "early")
    report = lc.assess(mid, training=train, validation=val)
    assert report["passed"] and report["training_replayed"]
    lc = ModelLifecycle(lc.root, lc.policy)  # new controller object, persisted state
    packet = lc.export(mid, tmp_path / "published")
    assert packet["model_sha256"] == digest(model)
    predictions = predict(model, reuse, SCOPE)
    event = lc.record_reuse(
        mid, session_id="later", mode="science_expert", inputs=reuse, predictions=predictions, scope=SCOPE
    )
    assert event["event"] == "reused" and event["rows"] == len(reuse)
    assert [e["event"] for e in lc._state()["events"]] == ["candidate", "validated", "registered", "reused"]
    # This tests the shared contract, not actual execution of the Scientist graph.


def test_rejected_candidate_never_becomes_available(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path, 0.9999999999)
    assert not lc.assess(mid, training=train, validation=val)["passed"]
    assert lc._state()["published"] == []
    with pytest.raises(ValueError, match="validated"):
        lc.export(mid, tmp_path / "bad")


def test_agent_written_metrics_cannot_bypass_replay(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path)
    # Tamper before submission in another independent controller.
    model["intercept"] += 100
    artifact = tmp_path / "forged.json"
    atomic_json(artifact, model)
    other = ModelLifecycle(tmp_path / "other", lc.policy)
    forged = other.submit(artifact, session_id="agent", mode="science_agent")
    with pytest.raises(ValueError, match="replayed training"):
        other.assess(forged, training=train, validation=val)
    assert other._state()["published"] == []


def test_hidden_labels_and_policy_cannot_change(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path)
    changed = [dict(r) for r in val]
    changed[0]["target"] += 1
    with pytest.raises(ValueError, match="data changed"):
        lc.assess(mid, training=train, validation=changed)
    with pytest.raises(ValueError, match="immutable"):
        ModelLifecycle(lc.root, dict(lc.policy, minimum_skill=0.5))


def test_no_assessment_overlap_even_when_file_hashes_differ(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path)
    overlapping = [dict(r, id="new_" + r["id"]) for r in train[:3]]
    other = ModelLifecycle(tmp_path / "other", dict(lc.policy, validation_sha256=digest(overlapping)))
    other_mid = other.submit(tmp_path / "model.json", session_id="agent", mode="science_agent")
    with pytest.raises(ValueError, match="overlap"):
        other.assess(other_mid, training=train, validation=overlapping)


def test_candidate_budget_and_double_assessment_are_enforced(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        lc.submit(tmp_path / "model.json", session_id="retry", mode="science_agent")
    lc.assess(mid, training=train, validation=val)
    with pytest.raises(ValueError, match="already been assessed"):
        lc.assess(mid, training=train, validation=val)


def test_wrong_assay_bad_predictions_and_same_session_cannot_claim_reuse(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path)
    lc.assess(mid, training=train, validation=val)
    preds = predict(model, reuse, SCOPE)
    with pytest.raises(ValueError, match="distinct session"):
        lc.record_reuse(mid, session_id="develop", mode="science_agent", inputs=reuse, predictions=preds, scope=SCOPE)
    with pytest.raises(ValueError, match="applicability"):
        lc.record_reuse(
            mid,
            session_id="new",
            mode="science_agent",
            inputs=reuse,
            predictions=preds,
            scope=dict(SCOPE, assay="other"),
        )
    preds[0]["prediction"] += 1
    with pytest.raises(ValueError, match="inference"):
        lc.record_reuse(mid, session_id="new", mode="science_agent", inputs=reuse, predictions=preds, scope=SCOPE)


def test_tampered_assessment_and_retirement_block_export(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path)
    lc.assess(mid, training=train, validation=val)
    path = lc.root / f"assessment_{mid}.json"
    record = json.loads(path.read_text())
    atomic_json(path, dict(record, passed=False))
    with pytest.raises(ValueError, match="assessment"):
        lc.export(mid, tmp_path / "tampered")
    atomic_json(path, record)
    lc.retire(mid, "assay calibration changed")
    with pytest.raises(ValueError, match="validated"):
        lc.export(mid, tmp_path / "retired")


@pytest.mark.parametrize(
    "change", [lambda r: r.update(target=float("nan")), lambda r: r.update(sequence="XX"), lambda r: r.update(id="")]
)
def test_invalid_protein_data_fails_closed(change):
    train, _, _ = data()
    change(train[0])
    with pytest.raises(ValueError):
        validate_rows(train, SCOPE, labels=True)


def test_runtime_plugin_tools_are_opt_in_and_do_not_enable_shell():
    allowed = frozenset({"develop_protein_model"})
    assert not safe_approval({"toolName": "mcp__venusfactory__develop_protein_model"})
    assert safe_approval({"toolName": "mcp__venusfactory__develop_protein_model"}, allowed)
    assert not safe_approval({"toolName": "mcp__venusfactory__validate_linear_fixture"}, allowed)
    assert not safe_approval({"toolName": "Bash"}, allowed)


def test_modes_name_real_engines_and_unknown_mode_fails():
    assert get_research_mode("science_expert").engine == "graph"
    assert get_research_mode("science_expert").roles == ("PI", "CB", "MLS", "SC")
    assert get_research_mode("science_agent").engine == "kimi-code"
    with pytest.raises(ValueError):
        get_research_mode("invented")


def test_model_artifact_tampering_is_detected(tmp_path):
    lc, mid, model, train, val, reuse = setup(tmp_path)
    artifact = lc.root / "artifacts/versions" / mid / "model.json"
    artifact.chmod(0o644)
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="file changed"):
        lc.assess(mid, training=train, validation=val)


def test_pairwise_plugin_survives_json_roundtrip_and_preserves_order():
    train, val, _ = data()
    model = fit_candidate(train, SCOPE, {"encoding": "onehot_pairwise", "alphas": [0.1, 1.0]})
    assert len(model["coefficients"]) == 440
    portable = json.loads(json.dumps(model, allow_nan=False))
    inputs = list(reversed(val))
    assert predict(model, inputs, SCOPE) == predict(portable, inputs, SCOPE)
    assert [r["id"] for r in predict(portable, inputs, SCOPE)] == [r["id"] for r in inputs]


def test_untrusted_plugin_cannot_enter_controller(tmp_path):
    lc, _, _, _, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="Unsupported trusted"):
        ModelLifecycle(tmp_path / "untrusted", dict(lc.policy, plugin_id="arbitrary_python"))


def test_positive_skill_cannot_publish_when_same_input_predictions_change(tmp_path, monkeypatch):
    from dataclasses import replace
    from agent.rsi.engine.numerical_evidence import verify_assessment
    import agent.rsi.engine.model_lifecycle as lifecycle

    lc, mid, model, train, validation, reuse = setup(tmp_path)
    plugin = get_model_plugin(PLUGIN_ID)
    received = []
    def drifting(model, rows, scope):
        assert all("target" not in row for row in rows)
        received.extend(rows)
        result = plugin.predict(model, rows, scope)
        result[0]["prediction"] += 0.5
        return result
    monkeypatch.setattr(lifecycle, "get_model_plugin", lambda _: replace(plugin, predict=drifting))
    report = lc.assess(mid, training=train, validation=validation)
    assert report["metrics"][plugin.acceptance_metric] > 0
    assert not report["passed"]
    assert report["prediction_replay"]["exact_match"] is False
    assert report["rejection_reasons"] == ["prediction_not_repeatable"]
    assert [r["id"] for r in received] == [r["id"] for r in validation]
    assert not lc._state()["published"]
    assert lc._state()["candidates"][mid]["status"] == "rejected"
    check = verify_assessment(lc.root, mid, model, lc.policy, report)
    assert check["evidence"]["prediction_replay"] == report["prediction_replay"]
    with pytest.raises(ValueError, match="validated"):
        lc.export(mid, tmp_path / "rejected")
    inputs = [{k: v for k, v in row.items() if k != "target"} for row in reuse]
    with pytest.raises(ValueError):
        lc.record_reuse(mid, session_id="later", mode="science_agent", inputs=inputs,
                        predictions=plugin.predict(model, inputs, SCOPE), scope=SCOPE)
