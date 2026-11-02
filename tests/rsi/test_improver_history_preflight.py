"""Numerical donor identity must be checked before spending policy-revision calls."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import audit_improver_revision as audit
import run_improver_revision as revision
from test_reap_goal_comparison import candidate
from test_reap_science_goal import setup_goal

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import file_hash
from agent.harness.host import write_json
from agent.rsi import generated_features


def donor(tmp_path):
    store = setup_goal(tmp_path)
    protocol = json.loads((store.root / "protocol.json").read_text())
    protocol["split"] = {"family": "fixture"}
    write_json(store.root / "protocol.json", protocol)
    cycle, _, _ = candidate(store.root, 0, 1000.0)
    store.mutate(
        lambda s: s.update(
            status="budget_exhausted",
            inflight=None,
            cycles=[{**cycle, "phase": "develop", "status": "candidate_rejected"}],
        )
    )
    return store.root


def corrupt_replay(monkeypatch, damage):
    if damage == "training":
        original = generated_features.fit_candidate

        def changed(*args):
            model = copy.deepcopy(original(*args))
            model["training_mean"] += 1
            return model

        monkeypatch.setattr(generated_features, "fit_candidate", changed)
    elif damage == "predictions":
        original = audit.independent_metrics

        def changed(*args):
            predictions, metrics = original(*args)
            return predictions, {**metrics, "predictions_sha256": "0" * 64}

        monkeypatch.setattr(audit, "independent_metrics", changed)


@pytest.mark.parametrize("damage", [None, "training", "predictions"])
def test_numerical_history_rejects_drift_and_never_reads_future(tmp_path, monkeypatch, damage):
    root = donor(tmp_path)
    projected, refs = revision.project_development_history([root])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original = audit.read

    def development_only(path):
        assert "future" not in str(path)
        return original(path)

    monkeypatch.setattr(audit, "read", development_only)
    corrupt_replay(monkeypatch, damage)
    if damage:
        with pytest.raises(ValueError, match="identity"):
            audit.history_projection(refs, numerical=True)
    else:
        assert audit.history_projection(refs, numerical=True) == projected
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [None, "training", "predictions"])
async def test_revision_preflight_precedes_credentials_and_execute(tmp_path, monkeypatch, damage):
    root = donor(tmp_path)
    profile = Profile(
        "science_agent",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=120, model_calls=8, tool_calls=4),
    )
    profile_path = tmp_path / "profile.json"
    write_json(profile_path, profile.public())
    monkeypatch.setattr(revision, "builtin_registry", lambda: SimpleNamespace(preflight=lambda *a: {"ready": True}))
    monkeypatch.setattr(revision, "default_runtimes", lambda: {})
    output = tmp_path / "revision"
    protocol = revision.prepare(
        SimpleNamespace(output=output, profile=profile_path, parent_policy=None, history_goal=[root])
    )
    assert {"scripts/rsi/audit_improver_revision.py", "scripts/rsi/audit_reap_generated.py"} <= protocol[
        "implementation"
    ].keys()
    calls = []

    def credential(*args):
        assert json.loads((output / "history_preflight.json").read_text())["status"] == "passed"
        calls.append("credential")
        return "fixture"

    async def execute(*args, **kwargs):
        calls.append("execute")
        return {"status": "execution_failed", "audit": {"passed": False}}

    monkeypatch.setattr(revision, "credential_from_toml", credential)
    monkeypatch.setattr(revision, "execute", execute)
    monkeypatch.setattr(revision, "audit_events", lambda *a: {"passed": False})
    corrupt_replay(monkeypatch, damage)
    if damage:
        with pytest.raises(ValueError, match="identity"):
            await revision.run(output, None)
        assert calls == []
        report = json.loads((output / "report.json").read_text())
        assert report["status"] == "history_preflight_failed" and report["external_model_calls"] == 0
        assert report["promoted"] is False and report["recursive_effectiveness_verified"] is False
    else:
        result = await revision.run(output, None)
        assert calls == ["credential", "execute"] and result["status"] == "revision_failed"
    report = json.loads((output / "report.json").read_text())
    receipt = output / "history_preflight.json"
    assert report["history_preflight_sha256"] == file_hash(receipt)
    assert json.loads(receipt.read_text())["external_model_calls"] == 0
    before = (output / "report.json").read_bytes()
    with pytest.raises(ValueError, match="Already attempted"):
        await revision.run(output, None)
    assert (output / "report.json").read_bytes() == before
