"""Synthetic scheduler checks; simulated publications are never scientific models."""

import asyncio
import copy
import importlib
from pathlib import Path

import pytest
from test_generated_models import SCOPE, TRAIN, VALID

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi import model_tasks
from agent.rsi import scientist_model_campaign as campaign
from agent.rsi import training_attempt_memory as memory
from agent.rsi.scientist_model_library import ScientistModelLibrary as RealLibrary
from agent.rsi.storage import atomic_json
from agent.science_goal import GoalSpec, GoalStore


@pytest.fixture
def task(tmp_path):
    reference = tmp_path / "reference.txt"
    reference.write_text("Synthetic scheduler fixture")
    data = {
        "training": TRAIN,
        "validation": VALID,
        "future": [{"id": "HI", "sequence": "HI", "target": 19.0}, {"id": "IH", "sequence": "IH", "target": 20.0}],
    }
    bindings = {}
    for phase, rows in data.items():
        p = tmp_path / (phase + ".json")
        atomic_json(p, rows)
        bindings[phase] = {"path": str(p), "sha256": file_hash(p)}
    return {
        "schema": 1,
        "id": "synthetic_activity",
        "stage": "activity",
        "plugin_id": "generated_python.v1",
        "scope": SCOPE,
        "paper": {"title": "Synthetic fixture", "doi": "test:fixture"},
        "references": [{"path": str(reference), "sha256": file_hash(reference)}],
        "data": bindings,
        "minimum_skill": 0.0,
    }


@pytest.fixture
def setup(tmp_path, task, monkeypatch):
    repo = tmp_path / "repository"
    repo.mkdir()
    source = repo / "worker.py"
    source.write_text("# synthetic frozen executor\n")
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("dmx", "glm-4-flash", "https://www.dmxapi.cn/v1"),
        Budget(wall_seconds=10, request_seconds=5, model_calls=3, tool_calls=2, output_tokens=4096, thinking=None),
    ).public()
    base = {
        "id": "first",
        "task": task,
        "profile": profile,
        "generations": 1,
        "goal_budget": {"max_cycles": 4, "model_calls": 12, "tool_calls": 8, "wall_seconds": 60},
    }
    models, resolved, settings = {}, [], {"duplicate": False, "external": False, "retired": False}

    class Library:
        def __init__(self, root):
            self.real = RealLibrary(root)
            self.root = self.real.root

        def goal(self, path):
            return self.real.goal(path)

        def publish_goal(self, goal):
            identity = digest("duplicate" if settings["duplicate"] else goal.name)
            origin = str(tmp_path / "unrelated_goal") if settings["external"] else str(goal)
            models.setdefault(
                identity, {"origin_goal": origin, "origin": {"stage": campaign.read(goal / "model_task.json")["stage"]}}
            )
            return {"all_published": True, "outcomes": [{"status": "published", "entry_id": identity}]}

        def resolve(self, identity, *, replay_training):
            assert replay_training
            if settings["retired"]:
                raise ValueError("Synthetic retired entry")
            resolved.append(identity)
            return models[identity]

    monkeypatch.setattr(campaign, "ScientistModelLibrary", Library)

    def create(*, count=2, target=2, calls=24, stages=(), execution=campaign.COVERAGE_TARGET):
        allocations = [{**copy.deepcopy(base), "id": "allocation_" + str(i)} for i in range(count)]
        store = campaign.ModelCampaignStore.create(
            tmp_path / "campaign",
            counting="program_entry.v1",  # Preserve the historical protocol in these fixtures.
            objective="Synthetic bounded campaign",
            target_models=target,
            library_root=tmp_path / "library",
            repository=repo,
            implementation={"worker.py": file_hash(source)},
            allocations=allocations,
            resources={"model_calls": calls, "tool_calls": 16 * count, "wall_seconds": 120 * count},
            required_stages=stages,
            execution=execution,
        )
        return store

    return create, base, settings, models, resolved


def checkpoint(allocation, goal, *, status="completed", calls=1):
    if not (goal / "goal.json").exists():
        criteria = ["accepted_model", "fresh_session_reuse"] + (
            ["verified_lineage"] if allocation["generations"] > 1 else []
        )
        store = GoalStore.create(
            goal,
            GoalSpec("Synthetic child", tuple(criteria), **allocation["goal_budget"]),
            Profile.from_dict(allocation["profile"]),
            admission=model_tasks.admission(allocation["task"]),
        )
        atomic_json(goal / "model_task.json", allocation["task"])
        atomic_json(goal / "protocol.json", {"evolution": {"generations": allocation["generations"]}})
    else:
        store = GoalStore(goal)

    def update(state):
        state["status"] = status
        state["usage"]["model_calls"] += calls
        state["usage"]["tool_calls"] += 1
        state["usage"]["wall_seconds"] += 0.01

    return store.mutate(update)


async def complete(allocation, goal):
    return checkpoint(allocation, goal)


@pytest.mark.asyncio
async def test_completion_requires_actual_verified_unique_count(setup):
    create, _, _, _, resolved = setup
    store = create(target=1000)
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "needs_tasks"
    assert len(state["verified_entry_ids"]) == 2
    assert state["usage"]["model_calls"] == 2
    assert state["usage"]["tool_calls"] == 2
    assert len(resolved) >= 2
    assert state["owner"] is None and state["inflight"] is None


@pytest.mark.asyncio
async def test_target_stops_unused_allocations(setup):
    create, *_ = setup
    store = create(target=1)
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "completed"
    assert [x["status"] for x in state["allocations"].values()] == ["completed", "queued"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, count", [("duplicate", 1), ("external", 0)])
async def test_duplicates_and_other_campaigns_never_inflate_count(setup, kind, count):
    create, _, settings, *_ = setup
    settings[kind] = True
    store = create(target=2)
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "needs_tasks" and len(state["verified_entry_ids"]) == count


@pytest.mark.asyncio
async def test_stage_coverage_is_required_even_when_count_is_met(setup):
    create, *_ = setup
    store = create(target=1, stages=["activity", "stability"])
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "needs_tasks"
    assert all(x["status"] == "completed" for x in state["allocations"].values())
    audit = campaign.read(store.root / state["last_publication_audit"])
    assert audit["verified_stages"] == ["activity"] and not audit["target_met"]


@pytest.mark.asyncio
async def test_aggregate_budget_reservation_prevents_another_goal(setup):
    create, *_ = setup
    store = create(calls=12)
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "budget_exhausted"
    assert state["usage"]["model_calls"] == 1
    assert state["allocations"]["allocation_1"]["status"] == "queued"


@pytest.mark.asyncio
async def test_provider_failure_stops_allocation_and_explicit_resume_keeps_usage(setup):
    create, *_ = setup
    store = create()

    async def access_failed(allocation, goal):
        return checkpoint(allocation, goal, status="waiting_for_input")

    state = await campaign.run_campaign(store, access_failed)
    assert state["status"] == "waiting_for_model"
    assert state["usage"]["model_calls"] == 1
    assert not store.goal("allocation_1").exists()
    with pytest.raises(ValueError, match="not runnable"):
        await campaign.run_campaign(store, complete)
    store.resume()
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "completed" and state["usage"]["model_calls"] == 3
    assert len(state["allocations"]["allocation_0"]["attempts"]) == 2


@pytest.mark.asyncio
async def test_pause_does_not_start_next_task(setup):
    create, *_ = setup
    store = create(target=1)

    async def pause(allocation, goal):
        checkpoint(allocation, goal, status="active")
        store.pause()
        return GoalStore(goal).read()

    state = await campaign.run_campaign(store, pause)
    assert state["status"] == "paused" and not store.goal("allocation_1").exists()
    store.resume()
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "completed" and state["usage"]["model_calls"] == 2


@pytest.mark.asyncio
async def test_unknown_failure_retains_reservation_without_fabricated_success(setup):
    create, *_ = setup
    store = create(count=1, target=1, calls=12)

    async def fail(*args):
        raise RuntimeError("Synthetic unknown failure")

    state = await campaign.run_campaign(store, fail)
    assert state["status"] == "needs_tasks" and state["usage"]["model_calls"] == 12
    assert not state["verified_entry_ids"]
    assert state["allocations"]["allocation_0"]["attempts"][0]["usage_known"] is False


@pytest.mark.asyncio
async def test_process_loss_retains_reservation_and_never_restarts_same_task(setup, monkeypatch):
    create, *_ = setup
    store = create()

    async def interrupted(*args):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await campaign.run_campaign(store, interrupted)
    assert store.read()["inflight"] is not None
    with pytest.raises(ValueError, match="still alive"):
        store.recover()
    monkeypatch.setattr(campaign, "process_live", lambda _: False)
    recovered = store.recover()
    assert recovered["usage"]["model_calls"] == 12
    assert recovered["allocations"]["allocation_0"]["status"] == "interrupted"
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "needs_tasks" and state["usage"]["model_calls"] == 13
    assert len(state["verified_entry_ids"]) == 1


@pytest.mark.asyncio
async def test_single_runner_lease_prevents_duplicate_execution(setup):
    create, *_ = setup
    store = create(target=1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def worker(allocation, goal):
        entered.set()
        await release.wait()
        return checkpoint(allocation, goal)

    running = asyncio.create_task(campaign.run_campaign(store, worker))
    await entered.wait()
    with pytest.raises(RuntimeError, match="already owns"):
        await campaign.run_campaign(store, worker)
    release.set()
    state = await running
    assert state["usage"]["model_calls"] == 1


@pytest.mark.asyncio
async def test_append_plan_keeps_prior_records_and_restarts_only_new_allocations(setup):
    create, base, *_ = setup
    store = create(count=1, target=2, calls=12)
    first = await campaign.run_campaign(store, complete)
    first_plan = first["plans"][0]
    before = file_hash(store.root / "plans" / (first_plan + ".json"))
    store.extend([{**base, "id": "second"}], resources={"model_calls": 12, "tool_calls": 8, "wall_seconds": 60})
    state = await campaign.run_campaign(store, complete)
    assert state["status"] == "completed" and state["usage"]["model_calls"] == 2
    assert file_hash(store.root / "plans" / (first_plan + ".json")) == before
    assert len(state["allocations"]["allocation_0"]["attempts"]) == 1


@pytest.mark.asyncio
async def test_changed_publication_cannot_satisfy_completion(setup):
    create, _, settings, *_ = setup
    store = create(target=1)
    await campaign.run_campaign(store, complete)
    settings["retired"] = True
    with pytest.raises(ValueError, match="retired"):
        store.audit_publications()
    assert store.read()["status"] == "publication_invalid"
    assert store.read()["verified_entry_ids"] == []


@pytest.mark.parametrize("attack", ["source", "plan", "definition", "inventory"])
def test_frozen_campaign_tampering_is_rejected(setup, attack):
    create, *_ = setup
    store = create()
    state = store.read()
    if attack == "source":
        (Path(state["definition"]["repository"]) / "worker.py").write_text("changed")
    elif attack == "plan":
        path = store.root / "plans" / (state["plans"][0] + ".json")
        path.chmod(0o644)
        atomic_json(path, {"changed": True})
    else:
        if attack == "definition":
            state["definition"]["target_models"] = 1
        else:
            del state["allocations"]["allocation_1"]
        atomic_json(store.root / "campaign.json", state)
    with pytest.raises(ValueError):
        store.read()


def test_capacity_is_not_reported_as_model_publication(setup):
    create, *_ = setup
    store = create(count=1001, target=1001, calls=12012)
    state = store.read()
    assert len(state["allocations"]) == 1001 and state["verified_entry_ids"] == []
    assert state["usage"]["model_calls"] == 0 and state["status"] == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["profile", "task", "budget", "result"])
async def test_executor_cannot_substitute_a_different_goal(setup, attack):
    create, *_ = setup
    store = create(count=1, target=1, calls=12)

    async def unbound(allocation, goal):
        state = checkpoint(allocation, goal)
        if attack == "result":
            state["usage"]["model_calls"] = 0
            return state
        if attack == "profile":
            state["definition"]["profile"]["model"]["model"] = "another-model"
        elif attack == "task":
            state["definition"]["admission"] = {}
        else:
            state["definition"]["goal"]["model_calls"] += 1
        state["definition_sha256"] = digest(state["definition"])
        atomic_json(goal / "goal.json", state)
        return state

    state = await campaign.run_campaign(store, unbound)
    assert state["status"] == "needs_tasks" and not state["verified_entry_ids"]
    assert state["usage"]["model_calls"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("http_status", [401, 403])
async def test_real_goal_loop_stops_after_one_auth_failure(setup, monkeypatch, http_status):
    create, *_ = setup
    store = create()
    store.mutate(lambda s: s.update(status="active"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    runner = importlib.import_module("run_scientist_model_campaign")
    calls = []

    class Driver:
        def __init__(self, child, credential):
            self.store = child

        def evaluate(self, state):
            return {
                "criteria": {key: {"passed": False, "evidence": []} for key in state["definition"]["goal"]["criteria"]}
            }

        async def cycle(self, index, profile, state, context):
            calls.append(index)
            folder = self.store.root / "episodes" / f"cycle_{index:03d}"
            (folder / "execution").mkdir(parents=True)
            (folder / "execution/model_requests.jsonl").write_text(
                '{"event":"response","status":' + str(http_status) + "}\n"
            )
            return {
                "status": "failed",
                "summary": "Synthetic access failure",
                "progress": {},
                "usage": {"model_calls": 1, "tool_calls": 0, "wall_seconds": 0.01},
            }

    monkeypatch.setattr(runner, "GeneratedModelGoal", Driver)
    monkeypatch.setattr(runner, "credential_from_toml", lambda *args: "synthetic-credential")
    allocation = store.specifications()["allocation_0"]
    state = await runner.execute_allocation(allocation, store.goal("allocation_0"), Path("unused"))
    assert state["status"] == "waiting_for_input" and calls == [0]
    assert state["cycles"][0]["external_blocker"]["http_status"] == http_status


def test_process_identity_uses_live_pid_start_and_boot():
    identity = campaign.process_identity()
    assert campaign.process_live(identity)
    assert not campaign.process_live({**identity, "start_ticks": "0"})
    assert not campaign.process_live({**identity, "boot_id": "wrong-boot"})


@pytest.mark.asyncio
async def test_cli_missing_credential_leaves_allocation_and_budget_untouched(setup, monkeypatch):
    from types import SimpleNamespace

    create, *_ = setup
    store = create()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    runner = importlib.import_module("run_scientist_model_campaign")
    monkeypatch.setattr(runner, "credential_from_toml", lambda *args: None)
    args = SimpleNamespace(action="run", output=store.root, credential_toml=Path("unused"))
    with pytest.raises(ValueError, match="no resources reserved"):
        await runner.main(args)
    assert store.read()["usage"]["model_calls"] == 0
    assert all(x["status"] == "queued" for x in store.read()["allocations"].values())
    assert not (store.root / "goals").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["run", "resume"])
async def test_paid_model_is_rejected_before_credentials_or_budget(setup, monkeypatch, action):
    from types import SimpleNamespace

    from agent.rsi.model_access import ModelAccessDenied

    create, base, *_ = setup
    base["profile"]["model"]["model"] = "glm-5.2"
    store = create()
    before = store.read()
    runner = importlib.import_module("run_scientist_model_campaign")

    def credentials_must_not_be_read(*args):
        pytest.fail("Rejected allocation attempted credential resolution")

    monkeypatch.setattr(runner, "credential_from_toml", credentials_must_not_be_read)
    with pytest.raises(ModelAccessDenied):
        await runner.main(SimpleNamespace(action=action, output=store.root, credential_toml=Path("unused")))
    assert store.read() == before
    assert not (store.root / "goals").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", memory.POLICIES)
async def test_campaign_memory_selection_reaches_prepared_goal_and_binding(setup, monkeypatch, policy):
    create, base, *_ = setup
    base[memory.FIELD] = policy
    store = create(count=1, target=1, calls=12)
    store.mutate(lambda s: s.update(status="active"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    runner = importlib.import_module("run_scientist_model_campaign")
    monkeypatch.setattr(runner, "credential_from_toml", lambda *args: "synthetic-credential")
    called = []

    async def inspect(child, *args, **kwargs):
        protocol = campaign.read(child.root / "protocol.json")
        assert memory.verify_goal(child.read(), protocol) == policy
        called.append(True)
        return child.read()

    monkeypatch.setattr(runner, "run_goal", inspect)
    allocation = store.specifications()["allocation_0"]
    root = store.goal("allocation_0")
    state = await runner.execute_allocation(allocation, root, Path("unused"))
    assert called == [True] and state["usage"]["model_calls"] == 0
    assert campaign.bind_goal(allocation, root) == state
    if policy == memory.CUMULATIVE_POLICY:
        undeclared = {k: v for k, v in allocation.items() if k != memory.FIELD}
        with pytest.raises(ValueError, match="explicit campaign allocation"):
            campaign.bind_goal(undeclared, root)
    protocol = campaign.read(root / "protocol.json")
    protocol[memory.FIELD] = "none" if policy != "none" else memory.POLICY
    atomic_json(root / "protocol.json", protocol)
    with pytest.raises(ValueError, match="memory.*changed"):
        campaign.bind_goal(allocation, root)


def test_unknown_campaign_memory_is_rejected_during_admission(setup):
    create, base, *_ = setup
    base[memory.FIELD] = "unknown-policy"
    with pytest.raises(ValueError, match="Unknown training-attempt"):
        create()
