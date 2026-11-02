"""Shared campaign scheduling and real local fits; no autonomous fixture claims."""

import copy
import json
import sys
from pathlib import Path

import pytest
import test_shared_science_goal as goal_fixture
from test_shared_model_program import make_task
from test_shared_model_publication import synthetic_native_seam

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi import shared_model_campaign as campaign
from agent.rsi.engine.scientist_model_library import ScientistModelLibrary
from agent.rsi.engine.source_snapshot import research_implementation_paths
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.science_goal import GoalSpec, GoalStore, run_goal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
import run_shared_science_goal as shared


def read(path):
    return json.loads(path.read_text())


def batch(root, index, *, reject=False):
    tasks = []
    for offset in range(2):
        task, _, _ = make_task(root / f"data_{index}_{offset}", protein=f"Synthetic plant {index}_{offset}")
        task.update(id=f"plant_{index}_{offset}", plugin_id="generated_context.v2")
        task["scope"]["interface_version"] = task["plugin_id"]
        if reject and offset:
            path = Path(task["data"]["validation"]["path"])
            rows = read(path)
            for row in rows:
                row["target"] *= -1
            atomic_json(path, rows)
            task["data"]["validation"]["sha256"] = sha256_file(path)
        tasks.append(task)
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test", "synthetic_no_api", "https://example.org/v1"),
        Budget(model_calls=8, tool_calls=8, wall_seconds=120, request_seconds=60),
    )
    return {
        "id": f"batch_{index}",
        "tasks": tasks,
        "profile": profile.public(),
        "goal_budget": {"max_cycles": 4, "model_calls": 32, "tool_calls": 32, "wall_seconds": 900},
        "training_baseline_feedback": "group_baselines.v2",
        "shared_training_history": "bounded_training_program_history.v2",
    }


def create(root, batches, *, target=10000, resources=None, full_source=False, excluded=(), required=()):
    paths = research_implementation_paths(ROOT) if full_source else [ROOT / "src/agent/rsi/campaigns/shared_model_campaign.py"]
    implementation = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    limits = {key: sum(row["goal_budget"][key] for row in batches) for key in campaign.RESOURCES}
    return campaign.SharedCampaignStore.create(
        root / "campaign",
        objective="Acquire independently accepted plant models and fresh reuse",
        target_scopes=target,
        library_root=root / "library",
        repository=ROOT,
        implementation=implementation,
        batches=batches,
        resources=limits if resources is None else resources,
        excluded_scopes=excluded,
        required_stages=required,
    )


def executor(store, monkeypatch, *, native_seam=False):
    calls = goal_fixture.synthetic_executor(monkeypatch)
    delegate = shared.execute

    async def bound(profile, task, runtimes, *, credential):
        lock = {"profile": profile.public(), "task": task.binding()}
        lock["sha256"] = campaign.digest(lock)
        result = await delegate(profile, task, runtimes, credential=credential)
        result["profile_sha256"] = lock["sha256"]
        atomic_json(task.output / "profile.lock.json", lock)
        atomic_json(task.output / "run.json", result)
        return result

    monkeypatch.setattr(shared, "execute", bound)
    if native_seam:
        synthetic_native_seam(monkeypatch)

    async def run(item, folder):
        if (folder / "goal.json").exists():
            child = GoalStore(folder)
        else:
            goal = GoalSpec(store.read()["definition"]["objective"], campaign.CRITERIA, **item["goal_budget"])
            child = shared.prepare(
                folder,
                item["tasks"],
                Profile.from_dict(item["profile"]),
                goal,
                training_baseline_feedback=item["training_baseline_feedback"],
                shared_training_history=item["shared_training_history"],
            )
        driver = shared.SharedScientistGoal(child, None)
        return await run_goal(child, driver.cycle, driver.evaluate)

    return run, calls


def test_plan_deduplicates_scopes_and_exclusions_before_writing(tmp_path):
    first = batch(tmp_path, 0)
    repeated = copy.deepcopy(first)
    repeated["id"] = "different_batch_name"
    with pytest.raises(ValueError, match="Scientific scope"):
        create(tmp_path / "duplicate", [first, repeated])
    with pytest.raises(ValueError, match="Scientific scope"):
        create(tmp_path / "excluded", [first], excluded=[campaign.task_scope(first["tasks"][0])])
    assert not (tmp_path / "duplicate/campaign").exists()
    changed = copy.deepcopy(first["tasks"][0])
    changed["id"] = "renamed"
    changed["scope"]["grouping"]["evidence_sha256"] = "0" * 64
    assert campaign.task_scope(changed) == campaign.task_scope(first["tasks"][0])


@pytest.mark.parametrize("change", ["profile", "history", "budget", "fields", "name", "data"])
def test_admission_rejects_invalid_batches(tmp_path, change):
    item = batch(tmp_path, 0)
    if change == "profile":
        item["profile"]["harness"] = "langgraph"
    elif change == "history":
        item["shared_training_history"] = "unbound"
    elif change == "budget":
        item["goal_budget"]["max_cycles"] = 1
    elif change == "fields":
        item["invented"] = True
    elif change == "name":
        item["id"] = "../escape"
    else:
        item["tasks"][0]["data"]["training"]["sha256"] = "0" * 64
    with pytest.raises((ValueError, TypeError)):
        create(tmp_path / "invalid", [item])


def test_extension_preserves_target_chain_and_prior_allocations(tmp_path):
    first = batch(tmp_path, 0)
    store = create(tmp_path / "run", [first])
    definition = store.read()["definition_sha256"]
    store.mutate(lambda s: s.update(status="needs_tasks"))
    second = batch(tmp_path, 1)
    store.extend([second], resources={key: second["goal_budget"][key] for key in campaign.RESOURCES})
    state = store.read()
    assert state["definition_sha256"] == definition and state["definition"]["target_scopes"] == 10000
    assert (
        state["status"] == "ready"
        and len(state["plans"]) == 2
        and campaign.digest(store.specifications()["batch_0"]) == campaign.digest(first)
    )
    with pytest.raises(ValueError, match="Duplicate"):
        store.extend([second], resources=dict.fromkeys(campaign.RESOURCES, 0))
    store.pause()
    store.resume()
    assert store.read()["status"] == "ready"


@pytest.mark.asyncio
async def test_budget_shortfall_does_not_start_a_batch(tmp_path):
    store = create(
        tmp_path / "run", [batch(tmp_path, 0)], resources={"model_calls": 31, "tool_calls": 32, "wall_seconds": 900}
    )

    async def forbidden(*args):
        pytest.fail("No external execution permitted")

    state = await campaign.run_campaign(store, forbidden)
    assert state["status"] == "budget_exhausted" and not any(state["usage"].values()) and state["owner"] is None
    assert not store.goal("batch_0").exists()


@pytest.mark.asyncio
async def test_execution_failure_charges_reservation_once_and_preserves_large_target(tmp_path):
    store = create(tmp_path / "run", [batch(tmp_path, 0), batch(tmp_path, 1)])
    calls = []

    async def fail(item, folder):
        calls.append(item["id"])
        raise RuntimeError("Synthetic unknown execution cost")

    state = await campaign.run_campaign(store, fail)
    assert calls == ["batch_0", "batch_1"] and state["usage"] == state["limits"]
    assert state["status"] == "needs_tasks" and state["definition"]["target_scopes"] == 10000
    audit = read(store.root / "audits" / (state["last_audit"] + ".json"))
    assert audit["verified_scope_count"] == 0 and not audit["target_met"]
    assert all(not row["attempts"][0]["usage_known"] for row in state["batches"].values())


def test_interrupted_reservation_requires_dead_owner_and_is_never_requeued(tmp_path, monkeypatch):
    store = create(tmp_path / "run", [batch(tmp_path, 0)])
    limits = store.read()["limits"]
    store.mutate(
        lambda s: s.update(
            inflight={"batch": "batch_0", "owner": campaign.process_identity(), "reservation": limits},
            usage=limits,
            status="active",
        )
    )
    with pytest.raises(ValueError, match="No dead"):
        store.recover()
    monkeypatch.setattr(campaign, "process_live", lambda owner: False)
    state = store.recover()
    assert state["usage"] == limits and state["inflight"] is None
    assert state["batches"]["batch_0"]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_two_real_shared_goals_publish_four_scopes_with_native_gate_seam(tmp_path, monkeypatch):
    store = create(tmp_path / "run", [batch(tmp_path, 0), batch(tmp_path, 1)], target=4, full_source=True)
    run, calls = executor(store, monkeypatch, native_seam=True)
    state = await campaign.run_campaign(store, run)
    assert state["status"] == "completed" and state["owner"] is None
    assert [row["phase"] for row in calls] == ["develop", "reuse", "develop", "reuse"]
    assert state["usage"]["model_calls"] == 4
    audit = read(store.root / "audits" / (state["last_audit"] + ".json"))
    assert audit["verified_scope_count"] == 4 and audit["all_batches_terminal"]
    assert len(ScientistModelLibrary(tmp_path / "run/library").catalog()["entries"]) == 4
    path = store.goal("batch_0") / "targets/plant_0_0/model_task.json"
    value = read(path)
    value["stage"] = "expression" if value["stage"] != "expression" else "solubility"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="task changed"):
        store.audit_publications()


@pytest.mark.asyncio
async def test_partial_goal_registers_only_qualified_reused_scope_without_completing_campaign(tmp_path, monkeypatch):
    store = create(tmp_path / "run", [batch(tmp_path, 0, reject=True)], target=2, full_source=True)
    run, _ = executor(store, monkeypatch, native_seam=True)
    state = await campaign.run_campaign(store, run)
    assert state["status"] == "needs_tasks" and state["batches"]["batch_0"]["status"] == "budget_exhausted"
    audit = read(store.root / "audits" / (state["last_audit"] + ".json"))
    assert audit["verified_scope_count"] == 1 and not audit["target_met"]


@pytest.mark.asyncio
async def test_unpatched_native_gate_prevents_fixture_publication_and_counting(tmp_path, monkeypatch):
    store = create(tmp_path / "run", [batch(tmp_path, 0)], target=2, full_source=True)
    run, _ = executor(store, monkeypatch)
    state = await campaign.run_campaign(store, run)
    assert state["status"] == "needs_tasks" and state["batches"]["batch_0"]["status"] == "failed"
    assert not (tmp_path / "run/library/index").exists()
    audit = read(store.root / "audits" / (state["last_audit"] + ".json"))
    assert audit["verified_scope_count"] == 0


@pytest.mark.asyncio
async def test_model_access_resume_retains_child_and_charges_only_remaining_budget(tmp_path):
    item = batch(tmp_path, 0)
    # The campaign wall ceiling also includes host preparation/verification.
    # Reserve explicit host overhead while retaining the frozen child ceiling.
    resources = {key: item["goal_budget"][key] for key in campaign.RESOURCES}
    resources["wall_seconds"] += 100
    store = create(tmp_path / "run", [item], full_source=True, resources=resources)
    observations = []

    async def access_checkpoint(allocation, folder):
        observations.append(copy.deepcopy(store.read()["inflight"]))
        if not (folder / "goal.json").exists():
            spec = GoalSpec(store.read()["definition"]["objective"], campaign.CRITERIA, **allocation["goal_budget"])
            child = shared.prepare(
                folder,
                allocation["tasks"],
                Profile.from_dict(allocation["profile"]),
                spec,
                training_baseline_feedback=allocation["training_baseline_feedback"],
                shared_training_history=allocation["shared_training_history"],
            )
            return child.mutate(
                lambda state: state.update(
                    status="waiting_for_input", usage={"model_calls": 1, "tool_calls": 0, "wall_seconds": 0.1}
                )
            )
        child = GoalStore(folder)
        assert child.read()["status"] == "active"
        return child.mutate(lambda state: state.update(status="budget_exhausted"))

    state = await campaign.run_campaign(store, access_checkpoint)
    assert state["status"] == "waiting_for_model" and state["usage"]["model_calls"] == 1
    goal_hash = state["batches"]["batch_0"]["attempts"][0]["goal_sha256"]
    assert goal_hash == sha256_file(store.goal("batch_0") / "goal.json")
    store.resume()
    state = await campaign.run_campaign(store, access_checkpoint)
    assert state["status"] == "needs_tasks" and state["usage"]["model_calls"] == 1
    assert observations[0]["reservation"]["model_calls"] == 32
    assert observations[1]["reservation"]["model_calls"] == 31
    assert len(state["batches"]["batch_0"]["attempts"]) == 2


@pytest.mark.asyncio
async def test_cli_honors_pause_received_before_child_preparation(tmp_path, monkeypatch):
    import run_shared_model_campaign as entry

    store = create(tmp_path / "run", [batch(tmp_path, 0)], full_source=True)
    original = entry.prepare

    def paused_prepare(*args, **kwargs):
        store.pause()
        return original(*args, **kwargs)

    monkeypatch.setattr(entry, "prepare", paused_prepare)
    monkeypatch.setattr(entry, "authorize_profile", lambda profile: None)
    monkeypatch.setenv("VENUS_RESEARCH_API_KEY", "synthetic-no-network")

    async def forbidden(*args, **kwargs):
        pytest.fail("Pause must prevent a new native/model call")

    monkeypatch.setattr(shared, "execute", forbidden)
    state = await entry.execute_campaign(store)
    assert state["status"] == "paused" and state["usage"]["model_calls"] == 0
    assert GoalStore(store.goal("batch_0")).read()["status"] == "paused"


@pytest.mark.asyncio
async def test_missing_declared_credential_does_not_consume_a_batch(tmp_path, monkeypatch):
    import run_shared_model_campaign as entry

    store = create(tmp_path / "run", [batch(tmp_path, 0)], full_source=True)
    monkeypatch.setattr(entry, "authorize_profile", lambda profile: None)
    monkeypatch.delenv("VENUS_RESEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="before reservation"):
        await entry.execute_campaign(store)
    state = store.read()
    assert state["status"] == "ready" and state["inflight"] is None
    assert not any(state["usage"].values()) and not store.goal("batch_0").exists()


@pytest.mark.asyncio
async def test_scope_count_does_not_replace_required_domain_coverage(tmp_path, monkeypatch):
    item = batch(tmp_path, 0)
    assert all(task["stage"] != "expression" for task in item["tasks"])
    store = create(tmp_path / "run", [item], target=2, full_source=True, required=["expression"])
    run, _ = executor(store, monkeypatch, native_seam=True)
    state = await campaign.run_campaign(store, run)
    audit = read(store.root / "audits" / (state["last_audit"] + ".json"))
    assert audit["verified_scope_count"] == 2 and audit["all_batches_terminal"]
    assert not audit["target_met"] and state["status"] == "needs_tasks"


@pytest.mark.parametrize("changed", ["counter", "settled", "reservation"])
def test_budget_counter_is_reconstructed_from_attempts_and_live_reservation(tmp_path, changed):
    store = create(tmp_path / "run", [batch(tmp_path, 0)])
    state = store.read()
    charge = {"model_calls": 1, "tool_calls": 0, "wall_seconds": 0.1}
    state["batches"]["batch_0"]["attempts"] = [
        {"status": "failed", "usage_known": True, "charged": charge, "publication": None}
    ]
    state["usage"] = copy.deepcopy(charge)
    atomic_json(store.root / "campaign.json", state)
    assert store.read()["usage"]["model_calls"] == 1
    if changed == "counter":
        state["usage"]["model_calls"] = 0
    elif changed == "settled":
        state["batches"]["batch_0"]["attempts"][0]["charged"]["model_calls"] = 0
    else:
        state["inflight"] = {
            "batch": "batch_0",
            "reservation": copy.deepcopy(charge),
            "owner": campaign.process_identity(),
        }
    atomic_json(store.root / "campaign.json", state)
    with pytest.raises(ValueError, match="usage differs"):
        store.read()
