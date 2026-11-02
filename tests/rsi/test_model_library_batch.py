"""Real shared fits and inference; native fixtures remain explicitly synthetic."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_scientist_model_library as single_fixtures
import test_shared_model_publication as shared_fixtures

from agent.rsi import model_library_batch as batch
from agent.rsi import model_tasks
from agent.rsi import scientist_model_library as library_module
from agent.rsi.scientist_model_library import ScientistModelLibrary
from agent.rsi.storage import atomic_json, sha256_file

completed = shared_fixtures.completed
storage = single_fixtures.storage
synthetic_native_seam = shared_fixtures.synthetic_native_seam


def ready(completed, tmp_path, monkeypatch):
    store, driver, _ = completed
    synthetic_native_seam(monkeypatch)
    library = ScientistModelLibrary(tmp_path / "batch_library")
    published = library.publish_goal(store.root)
    requests = []
    for index, row in enumerate(published["outcomes"]):
        task_id = row["task_id"]
        definition = driver.by_id[task_id]
        requests.append(
            {
                "entry_id": row["entry_id"],
                "inputs": model_tasks.inference_inputs(driver.data[task_id]["future"], definition["plugin_id"]),
                "scope": definition["scope"],
                "session_id": f"independent_agent_{index}",
            }
        )
    return store, library, requests


@pytest.mark.asyncio
async def test_batch_matches_individual_inference_and_verifies_shared_goal_twice(completed, tmp_path, monkeypatch):
    store, library, requests = ready(completed, tmp_path, monkeypatch)
    prior = sha256_file(store.root / "goal.json")
    expected = [library.predict(**r)["predictions"] for r in requests]
    calls = []
    original = batch.validate_shared_goal

    def counted(folder, **kwargs):
        calls.append(folder)
        return original(folder, **kwargs)

    monkeypatch.setattr(batch, "validate_shared_goal", counted)
    result = library.predict_many(requests)
    assert [r["predictions"] for r in result["results"]] == expected
    assert len(calls) == 2 and len(set(calls)) == 1
    assert result["batch_receipt"]["verified_shared_goals"] == 1
    assert not result["batch_receipt"]["persistent_validation_cache"]
    assert len({r["receipt"]["session_id"] for r in result["results"]}) == len(requests)
    assert len({r["receipt_id"] for r in result["results"]}) == len(requests)
    assert library._object("usage_batches", result["batch_receipt_id"]) == result["batch_receipt"]
    assert all(library._object("usage", r["receipt_id"]) == r["receipt"] for r in result["results"])
    assert sha256_file(store.root / "goal.json") == prior


@pytest.mark.asyncio
async def test_unpatched_native_gate_rejects_synthetic_batch(completed, tmp_path, monkeypatch):
    original = library_module._episodes
    _, library, requests = ready(completed, tmp_path, monkeypatch)
    monkeypatch.setattr(library_module, "_episodes", original)
    with pytest.raises(ValueError, match="publication auditor|Fixture execution"):
        library.predict_many(requests)
    assert not (library.root / "usage").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["predictions", "index", "goal", "training"])
async def test_change_during_inference_is_rejected_without_receipts(completed, tmp_path, monkeypatch, attack):
    store, library, requests = ready(completed, tmp_path, monkeypatch)
    paths = {
        "predictions": store.root / "episodes/cycle_001/workspace/targets/synthetic_0/predictions.json",
        "index": next((library.root / "index").glob("*.json")),
        "goal": store.root / "goal.json",
        "training": store.root / "targets/synthetic_0/data/training.json",
    }
    path = paths[attack]
    original_bytes = path.read_bytes()
    original_plugin = batch.get_model_plugin
    changed = []

    def plugin(name):
        actual = original_plugin(name)

        def predict(model, inputs, scope):
            result = actual.predict(model, inputs, scope)
            if not changed:
                atomic_json(path, {})
                changed.append(True)
            return result

        return SimpleNamespace(predict=predict)

    monkeypatch.setattr(batch, "get_model_plugin", plugin)
    try:
        with pytest.raises((ValueError, KeyError, TypeError)):
            library.predict_many(requests)
        assert changed
        assert not (library.root / "usage").exists()
        assert not (library.root / "usage_batches").exists()
    finally:
        path.write_bytes(original_bytes)


@pytest.mark.asyncio
async def test_second_request_revalidates_artifacts(completed, tmp_path, monkeypatch):
    store, library, requests = ready(completed, tmp_path, monkeypatch)
    library.predict_many(requests)
    usages = set((library.root / "usage").glob("*.json"))
    path = store.root / "episodes/cycle_000/shared_training_history.json"
    original_bytes = path.read_bytes()
    atomic_json(path, {})
    try:
        with pytest.raises((ValueError, KeyError)):
            library.predict_many(requests)
        assert set((library.root / "usage").glob("*.json")) == usages
    finally:
        path.write_bytes(original_bytes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem", ["empty", "duplicate_session", "unknown_field", "wrong_scope", "invalid_mode", "oversized"]
)
async def test_invalid_requests_do_not_write_receipts(completed, tmp_path, monkeypatch, problem):
    _, library, requests = ready(completed, tmp_path, monkeypatch)
    if problem == "empty":
        requests = []
    elif problem == "duplicate_session":
        requests[1]["session_id"] = requests[0]["session_id"]
    elif problem == "unknown_field":
        requests[0]["skip_verification"] = True
    elif problem == "wrong_scope":
        requests[0]["scope"] = {**requests[0]["scope"], "target": "another endpoint"}
    elif problem == "invalid_mode":
        requests[0]["mode"] = "unknown"
    else:
        requests = requests * (batch.MAX_BATCH_REQUESTS + 1)
    with pytest.raises((ValueError, KeyError)):
        library.predict_many(requests)
    assert not (library.root / "usage").exists()


def test_legacy_resolver_is_retained_and_duplicate_entry_can_serve_distinct_sessions(storage, monkeypatch):
    library, root, verified, calls = storage
    entry = library.publish(root, verified["packet"]["model_id"])["entry_id"]
    calls.clear()
    request = {
        "entry_id": entry,
        "inputs": [{"id": "x", "sequence": "AC"}],
        "scope": verified["packet"]["model"]["scope"],
        "session_id": "first",
    }
    monkeypatch.setattr(
        batch,
        "get_model_plugin",
        lambda _: SimpleNamespace(
            predict=lambda model, rows, scope: [{"id": r["id"], "prediction": 1.0} for r in rows]
        ),
    )
    result = library.predict_many([request, {**deepcopy(request), "session_id": "second"}])
    assert calls == [False, False]
    assert result["batch_receipt"]["unique_entries"] == 1
    assert result["batch_receipt"]["verified_shared_goals"] == 0
    assert result["batch_receipt"]["requests"] == 2
    assert result["results"][0]["predictions"] == result["results"][1]["predictions"]
