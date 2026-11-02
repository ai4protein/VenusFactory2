import json

import pytest
from test_generated_models import CODE, SCOPE, TRAIN, VALID

from agent.rsi.engine.experience import LIMITATION, ModelExperienceStore, validate_projection
from agent.rsi.engine.generated_model import fit_candidate
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json


@pytest.fixture
def accepted(tmp_path):
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
    lifecycle = ModelLifecycle(tmp_path / "controller", policy)
    atomic_json(tmp_path / "candidate.json", fit_candidate(TRAIN, SCOPE, {"source_code": CODE, "seed": 1}))
    mid = lifecycle.submit(tmp_path / "candidate.json", session_id="generated", mode="science_agent")
    lifecycle.assess(mid, training=TRAIN, validation=VALID)
    return lifecycle, mid


def test_projection_controls_source_access_and_excludes_weights_and_labels(accepted, tmp_path):
    lc, mid = accepted
    bank = ModelExperienceStore(tmp_path / "bank")
    eid = bank.publish(lc, mid, summary="Least squares over encoded amino-acid sites.")
    assert bank.publish(lc, mid, summary="Least squares over encoded amino-acid sites.") == eid
    assert bank.project("none")["entries"] == []
    summary = bank.project("summary")["entries"][0]
    source = bank.project("source")["entries"][0]
    assert "source_code" not in summary and source.pop("source_code") == CODE
    assert summary == source
    assert not {"model_state", "training_sequences", "metrics", "assessment", "model_id"} & summary.keys()


def test_unpublished_retired_and_modified_assessments_cannot_enter_library(accepted, tmp_path):
    lc, mid = accepted
    bank = ModelExperienceStore(tmp_path / "bank")
    with pytest.raises(ValueError, match="validated"):
        bank.publish(lc, "not-published", summary="test")
    path = lc.root / f"assessment_{mid}.json"
    report = json.loads(path.read_text())
    report["metrics"]["rmse"] = 999
    atomic_json(path, report)
    with pytest.raises(ValueError, match="changed"):
        bank.publish(lc, mid, summary="test")
    lc.retire(mid, "test retirement")
    with pytest.raises(ValueError, match="validated"):
        bank.publish(lc, mid, summary="test")


def test_lineage_and_content_integrity(accepted, tmp_path):
    lc, mid = accepted
    bank = ModelExperienceStore(tmp_path / "bank")
    with pytest.raises(ValueError, match="parent"):
        bank.publish(lc, mid, summary="test", parents=["a" * 64])
    eid = bank.publish(lc, mid, summary="first")
    child = bank.publish(lc, mid, summary="second", parents=[eid])
    assert bank.project("source", [child])["entries"][0]["parents"] == [eid]
    path = bank.root / "entries" / (eid + ".json")
    entry = json.loads(path.read_text())
    entry["source_code"] += "# tampered"
    atomic_json(path, entry)
    with pytest.raises(ValueError, match="changed"):
        bank.project("source")


def test_projection_rejects_leaked_fields(accepted, tmp_path):
    lc, mid = accepted
    bank = ModelExperienceStore(tmp_path / "bank")
    bank.publish(lc, mid, summary="test")
    packet = bank.project("summary")
    packet["entries"][0]["model_state"] = {"training_labels": [100]}
    with pytest.raises(ValueError, match="unapproved"):
        validate_projection(packet)
    with pytest.raises(ValueError, match="cold-start"):
        validate_projection({**bank.project("summary"), "condition": "none"})
    packet = bank.project("source")
    packet["entries"][0]["source_code"] += "# changed"
    with pytest.raises(ValueError, match="source changed"):
        validate_projection(packet)


def test_empty_projection_contract():
    assert validate_projection({"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION})
