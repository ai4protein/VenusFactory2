"""Multi-batch program memory; native proof is stubbed only in marked fixtures."""

import copy
import json
import sys
from pathlib import Path

import pytest
import test_shared_science_goal as goal_fixture
from test_context_transfer import task as context_task
from test_generated_context import CODE
from test_shared_model_campaign import batch, create
from test_shared_model_publication import synthetic_native_seam

from agent.harness import Profile
from agent.harness.contracts import digest
from agent.rsi import shared_campaign_experience as memory
from agent.rsi import shared_library_experience
from agent.rsi.engine.context_transfer import POLICY as OVERLAP_POLICY
from agent.rsi.engine.context_transfer import SOURCE_ADAPTATION
from agent.rsi.engine.storage import atomic_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
import run_shared_model_campaign as cli
import run_shared_science_goal as shared


def declaration():
    return {
        "policy": memory.POLICY,
        "condition": "source",
        "max_candidates": 5,
        "keep": 2,
        "ranking": "training_holdout_mean_skill",
        "overlap_policy": OVERLAP_POLICY,
        "adaptation_policy": SOURCE_ADAPTATION,
    }


def item(root, index):
    value = batch(root, index)
    tasks = []
    for offset in range(2):
        task, _ = context_task(root / f"study_{index}_{offset}", f"10.1234/study{index}", f"P{index}_{offset}")
        task["plugin_id"] = "generated_context.v2"
        task["scope"]["interface_version"] = task["plugin_id"]
        tasks.append(task)
    value["tasks"] = tasks
    value[memory.FIELD] = declaration()
    return value


@pytest.mark.parametrize("change", ["unknown", "capacity", "keep", "ranking", "missing_overlap", "extra"])
def test_campaign_rejects_invalid_program_memory_before_creation(tmp_path, change):
    value = item(tmp_path, 0)
    policy = value[memory.FIELD]
    if change == "unknown":
        policy["policy"] = "unknown"
    elif change == "capacity":
        policy["max_candidates"] = 100
    elif change == "keep":
        policy["keep"] = True
    elif change == "ranking":
        policy["ranking"] = "future_score"
    elif change == "missing_overlap":
        policy.pop("overlap_policy")
    else:
        policy["unbound_sources"] = []
    with pytest.raises(ValueError):
        create(tmp_path / "run", [value])
    assert not (tmp_path / "run/campaign").exists()


def test_first_batch_has_no_historical_program_and_unfinished_prefix_is_rejected(tmp_path):
    first, second = item(tmp_path, 0), item(tmp_path, 1)
    store = create(tmp_path / "run", [first, second])
    request = memory.select_for_batch(store, first)
    assert not request["entry_ids"]
    record = memory.verify_selection(store, first, replay=True)
    assert record["screen"]["candidate_source_tests"] == 0
    assert not record["pool"]["prefix_publications"]
    with pytest.raises(ValueError, match="Earlier batch is not terminal"):
        memory.select_for_batch(store, second)


@pytest.mark.asyncio
async def test_real_two_batch_campaign_selects_and_exposes_prior_published_code(tmp_path, monkeypatch):
    first, second = item(tmp_path, 0), item(tmp_path, 1)
    resources = {
        key: first["goal_budget"][key] + second["goal_budget"][key]
        for key in ["model_calls", "tool_calls", "wall_seconds"]
    }
    resources["wall_seconds"] += 600
    store = create(tmp_path / "run", [first, second], full_source=True, resources=resources)
    monkeypatch.setattr(goal_fixture, "CODE", CODE)
    calls = goal_fixture.synthetic_executor(monkeypatch)
    delegate = shared.execute

    async def bound(profile, task, runtimes, *, credential):
        assert credential == "synthetic-not-an-api-key"
        lock = {"profile": profile.public(), "task": task.binding()}
        lock["sha256"] = digest(lock)
        result = await delegate(profile, task, runtimes, credential=None)
        result["profile_sha256"] = lock["sha256"]
        atomic_json(task.output / "profile.lock.json", lock)
        atomic_json(task.output / "run.json", result)
        return result

    monkeypatch.setattr(shared, "execute", bound)
    monkeypatch.setattr(cli, "authorize_profile", lambda profile: None)
    monkeypatch.setenv(Profile.from_dict(first["profile"]).model.credential_env, "synthetic-not-an-api-key")
    synthetic_native_seam(monkeypatch)
    state = await cli.execute_campaign(store)
    assert state["status"] == "needs_tasks" and state["definition"]["target_scopes"] == 10000
    assert all(row["status"] == "completed" for row in state["batches"].values())
    assert [row["phase"] for row in calls] == ["develop", "reuse", "develop", "reuse"]
    assert state["usage"]["model_calls"] == 4
    first_record = memory.verify_selection(store, first, goal=store.goal(first["id"]), replay=True)
    second_record = memory.verify_selection(store, second, goal=store.goal(second["id"]), replay=True)
    assert not first_record["screen"]["selected"]["entry_ids"]
    assert second_record["pool"]["prior_entry_records"] == 2
    assert second_record["pool"]["prior_distinct_sources"] == 1
    assert len(second_record["screen"]["selected"]["entry_ids"]) == 1
    assert second_record["screen"]["candidate_source_tests"] == 2
    protocol = json.loads((store.goal(second["id"]) / "protocol.json").read_text())
    frozen = shared_library_experience.read_frozen(
        store.goal(second["id"]), json.loads((store.goal(second["id"]) / "goal.json").read_text()), protocol
    )
    projection = frozen["projection"]
    assert len(projection["entries"]) == 1
    assert json.dumps(projection, ensure_ascii=False) in calls[2]["prompt"]
    assert "Verified historical program precedents" not in calls[3]["prompt"]
    audit = json.loads((store.root / "audits" / (state["last_audit"] + ".json")).read_text())
    assert audit["verified_scope_count"] == 4 and not audit["target_met"]
    path = memory.selection_path(store, second)
    changed = copy.deepcopy(second_record)
    changed["screen"]["candidates"][0]["mean_training_holdout_skill"] += 1
    path.chmod(0o644)
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="screening replay differs"):
        store.audit_publications()
