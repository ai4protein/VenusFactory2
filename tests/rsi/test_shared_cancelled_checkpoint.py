"""Offline cancellation followed by real synthetic fits; not autonomous evidence."""

import asyncio

import pytest
import test_shared_science_goal as fixture
from test_shared_model_publication import synthetic_native_seam

from agent.rsi.protein_model import digest
from agent.rsi.scientist_model_library import ScientistModelLibrary, validate_model_goal
from agent.rsi.storage import atomic_json
from agent.science_goal import run_goal


def cancellation_executor(monkeypatch, *, corruption=None):
    fixture.synthetic_executor(monkeypatch)
    delegate = fixture.shared.execute
    count = 0

    async def execute(profile, task, runtimes, *, credential):
        nonlocal count
        count += 1
        lock = {"profile": profile.public(), "task": task.binding()}
        lock["sha256"] = digest(lock)
        atomic_json(task.output / "profile.lock.json", lock)
        if count > 1:
            run = await delegate(profile, task, runtimes, credential=credential)
            run["profile_sha256"] = lock["sha256"]
            atomic_json(task.output / "run.json", run)
            return run
        run = {
            "status": "interrupted",
            "phase": task.phase,
            "error_type": "CancelledError",
            "session_id": "cancelled_fixture",
            "host_wall_seconds": 0.0,
            "binding_intact": True,
            "cancellation_finalized": True,
            "profile_sha256": lock["sha256"],
            "audit": {"requests": {"model_calls": 1}, "events": {"tool_calls": 0}},
            "llm_calls": 1,
            "tool_calls": 0,
        }
        if corruption == "binding":
            run["binding_intact"] = False
        elif corruption == "marker":
            run.pop("cancellation_finalized")
        elif corruption == "success":
            run["status"] = "completed"
        elif corruption == "profile":
            run["profile_sha256"] = "changed"
        if corruption != "missing":
            atomic_json(task.output / "run.json", run)
        (task.output / "events.jsonl").write_text("")
        raise asyncio.CancelledError

    monkeypatch.setattr(fixture.shared, "execute", execute)
    monkeypatch.setattr(
        fixture.shared,
        "audit_events",
        lambda output, phase, run, *args: {"passed": run["status"] == "completed", "synthetic_executor": True},
    )


@pytest.mark.asyncio
async def test_cancelled_shared_episode_has_no_models_and_later_valid_models_survive(tmp_path, monkeypatch):
    store, driver = fixture.prepared(tmp_path)
    cancellation_executor(monkeypatch)
    state = await run_goal(store, driver.cycle, driver.evaluate, stop_on_uncheckpointed_failure=True)
    assert state["status"] == "completed" and len(state["cycles"]) == 3
    failed = state["cycles"][0]
    assert failed["status"] == "failed" and failed["execution_cancelled"] is True
    assert failed["progress"] == {"targets": {}} and failed["target_outcomes"] == {}
    assert state["usage"]["model_calls"] == 3
    saved = fixture.read(store.root / "episodes/cycle_000/outcome.json")
    assert {k: v for k, v in failed.items() if k not in {"index", "usage"}} == {
        k: v for k, v in saved.items() if k != "usage"
    }
    assert all(
        v["development_episode"] == "episodes/cycle_001" and v["reuse_episode"] == "episodes/cycle_002"
        for v in state["progress"]["targets"].values()
    )
    identity = state["progress"]["targets"]["synthetic_0"]["model_id"]
    with pytest.raises(ValueError, match="publication auditor|Fixture execution"):
        validate_model_goal(store.root, identity)
    # Only the explicit offline native seam is patched; task/history/model
    # checks and independent numerical replay remain the existing validators.
    synthetic_native_seam(monkeypatch)
    published = ScientistModelLibrary(tmp_path / "library").publish_goal(store.root)
    assert published["all_published"] and published["verified_target_scopes"] == 2
    assert published["shared_usage"]["model_calls"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["binding", "marker", "success", "profile", "missing"])
async def test_cancelled_episode_without_bound_host_receipt_still_stops(tmp_path, monkeypatch, corruption):
    store, driver = fixture.prepared(tmp_path)
    cancellation_executor(monkeypatch, corruption=corruption)
    state = await run_goal(store, driver.cycle, driver.evaluate, stop_on_uncheckpointed_failure=True)
    assert state["status"] == "interrupted" and len(state["cycles"]) == 1
    assert not state["progress"]
    assert state["usage"]["model_calls"] == 8  # Unknown usage retains the original reservation.
    assert not (store.root / "episodes/cycle_000/outcome.json").exists()
