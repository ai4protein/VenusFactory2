"""Reject the observed roundoff lineage while preserving frozen older protocols."""

import asyncio
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_science_evolution import EvolvingModelGoal
from run_science_goal import GeneratedModelGoal
from test_science_evolution import lineage as lineage

from agent.rsi.engine.lineage_promotion import gain_error, new_evolution, validate_evolution
from agent.rsi.datasets.source_lineage import program_hash

BEFORE = 0.14655056844396863
AFTER = 0.14655056844396874


@pytest.mark.parametrize("current", [BEFORE, AFTER, BEFORE - 0.01, BEFORE + 5e-11])
def test_equal_roundoff_regression_and_subtolerance_are_not_promotions(current):
    assert gain_error({"evolution": new_evolution(2)}, BEFORE, current)


def test_positive_gain_and_first_model_margin():
    protocol = {"evolution": new_evolution(2)}
    assert gain_error(protocol, BEFORE, BEFORE + 1e-6) is None
    assert gain_error(protocol, 0.0, 1e-16)
    assert gain_error(protocol, 0.0, 0.1) is None
    assert gain_error(protocol, math.nextafter(1.0, 0.0), 1.0)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "0.5", None])
def test_invalid_scores_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        gain_error({"evolution": new_evolution(2)}, BEFORE, bad)


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_roundoff_policy_is_preserved_for_historical_replay(schema):
    protocol = {"evolution": {"schema": schema, "generations": 2}}
    assert gain_error(protocol, BEFORE, AFTER) is None


@pytest.mark.parametrize("attack", ["missing", "zero_tolerance", "downgrade", "unknown", "extra"])
def test_guard_cannot_be_silently_removed_or_changed(attack):
    evolution = new_evolution(2)
    if attack == "missing":
        evolution.pop("promotion_guard")
    elif attack == "zero_tolerance":
        evolution["promotion_guard"]["atol"] = 0.0
    elif attack == "downgrade":
        evolution["schema"] = 2
    elif attack == "unknown":
        evolution["schema"] = 4
    else:
        evolution["promotion_guard"]["ignored"] = True
    with pytest.raises(ValueError):
        validate_evolution({"evolution": evolution})


def test_guard_dictionaries_are_not_shared():
    one = new_evolution(2)
    one["promotion_guard"]["atol"] = 0.0
    assert new_evolution(2)["promotion_guard"]["atol"] == 1e-10


def test_real_training_acceptance_and_reuse_satisfy_new_gate(lineage):
    goal, state = lineage
    goal.protocol["evolution"] = new_evolution(2)
    for record in state["progress"]["generations"]:
        lc, mid = goal.lifecycle({"progress": {"accepted_model": record}})
        model = lc._model(mid)
        record.update(source_sha256=model["source_sha256"], program_sha256=program_hash(model["recipe"]["source_code"]))
    assert all(v["passed"] for v in goal.evaluate(state)["criteria"].values())


def test_roundoff_candidate_is_retired_before_experience_or_reuse(tmp_path, monkeypatch):
    goal = object.__new__(EvolvingModelGoal)
    goal.store = SimpleNamespace(root=tmp_path)
    goal.protocol = {"evolution": new_evolution(2)}
    parent = {"controller": "parent", "model_id": "old", "reuse_receipt": "receipt.json"}
    current = {"controller": "current", "model_id": "new"}
    state = {"progress": {"generations": [parent]}}
    retired = []
    for name, mid, score in [("parent", "old", BEFORE), ("current", "new", AFTER)]:
        root = tmp_path / name
        root.mkdir()
        (root / ("assessment_" + mid + ".json")).write_text(
            json.dumps({"metrics": {"mse_skill_vs_training_mean": score}})
        )
    parent_lc = SimpleNamespace(root=tmp_path / "parent")
    current_lc = SimpleNamespace(
        root=tmp_path / "current", _model=lambda mid: {}, retire=lambda mid, reason: retired.append((mid, reason))
    )
    monkeypatch.setattr(
        goal,
        "lifecycle",
        lambda s: (parent_lc, "old") if s["progress"]["accepted_model"]["model_id"] == "old" else (current_lc, "new"),
    )
    monkeypatch.setattr(goal, "promotion_error", lambda model, records: None)

    async def host_cycle(self, index, profile, current_state, context):
        current_state["progress"]["accepted_model"] = current
        return {"status": "completed", "progress": current_state["progress"]}

    monkeypatch.setattr(GeneratedModelGoal, "cycle", host_cycle)
    (tmp_path / "episodes/cycle_002").mkdir(parents=True)
    unchanged = copy.deepcopy(state)
    outcome = asyncio.run(goal.cycle(2, None, state, "Improve the model"))
    assert outcome["status"] == "candidate_rejected"
    assert "tolerance" in outcome["summary"]
    assert retired[0][0] == "new"
    assert outcome["progress"]["generations"] == [parent]
    assert "accepted_model" not in outcome["progress"]
    assert state == unchanged
    assert not (tmp_path / "experience").exists()


@pytest.mark.parametrize("case", ["valid", "missing_guard", "roundoff"])
def test_library_independently_enforces_new_promotion_guard(tmp_path, monkeypatch, case):
    import test_scientist_model_library as numerical_tests
    from test_generation_publication import numerical_checkpoint

    from agent.rsi import generation_publication as generation
    from agent.rsi import scientist_model_library as library
    from agent.rsi.engine.storage import atomic_json

    # Real numerical fixture; native expert execution remains explicitly mocked.
    (tmp_path / "numeric").mkdir()
    numerical = numerical_tests.numerical.__wrapped__(tmp_path / "numeric")
    root, state, record, catalog, lifecycle = numerical_checkpoint(numerical, tmp_path, monkeypatch)
    protocol = generation.read(root / "protocol.json")
    protocol["evolution"] = new_evolution(2)
    if case == "missing_guard":
        protocol["evolution"].pop("promotion_guard")
    elif case == "roundoff":
        protocol["model_task_minimum_skill"] = BEFORE
        # Even an upstream acceptance response must pass the library's own gate.
        monkeypatch.setattr(library, "_model", lambda *args: {"skill": AFTER})
    atomic_json(root / "protocol.json", protocol)
    snapshot = generation.freeze(catalog, root, state)
    if case == "valid":
        result = library.validate_model_goal(snapshot, record["model_id"], replay_training=True)
        assert result["packet"]["model"] == lifecycle._model(record["model_id"])
    else:
        with pytest.raises(ValueError, match="promotion guard"):
            library.validate_model_goal(snapshot, record["model_id"], replay_training=True)
    assert not (catalog.root / "index").exists()
