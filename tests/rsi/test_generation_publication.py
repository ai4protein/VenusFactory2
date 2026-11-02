"""Synthetic checkpoint tests; none of these fixtures are autonomous models."""

import copy
from pathlib import Path

import pytest
import test_science_goal as goal_tests
import test_scientist_model_library as library_tests

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi import generation_publication as generation
from agent.rsi import scientist_model_library as library
from agent.rsi.engine.experience import ModelExperienceStore
from agent.rsi.datasets.source_lineage import program_hash
from agent.rsi.engine.storage import atomic_json
from agent.science_goal import GoalSpec, GoalStore, run_goal

numerical = library_tests.numerical


def checkpoint(tmp_path):
    profile = Profile(
        "science_expert", "expert-kimi-code", ModelBinding("test_fixture", "model", "https://example.org/v1"), Budget()
    )
    store = GoalStore.create(
        tmp_path / "goal",
        GoalSpec("Synthetic two-generation goal", ("accepted_model", "fresh_session_reuse", "verified_lineage")),
        profile,
    )
    record = {"model_id": "b" * 64, "reuse_receipt": "episodes/cycle_001/reuse_receipt.json"}

    def update(state):
        state["progress"] = {"generations": [record]}
        state["cycles"] = [
            {"index": 0, "status": "completed", "phase": "develop"},
            {"index": 1, "status": "completed", "phase": "reuse", "progress": copy.deepcopy(state["progress"])},
        ]
        state["assessment"] = {
            "criteria": {k: {"passed": False, "evidence": []} for k in state["definition"]["goal"]["criteria"]}
        }

    state = store.mutate(update)
    atomic_json(store.root / "protocol.json", {"evolution": {"schema": 2, "generations": 2}})
    for cycle in state["cycles"]:
        atomic_json(store.root / f"cycle_{cycle['index']:03d}.json", cycle)
    return store, library.ScientistModelLibrary(tmp_path / "library")


def test_snapshot_preserves_unfinished_goal_and_survives_later_failure(tmp_path):
    store, catalog = checkpoint(tmp_path)
    before = generation.files(store.root)
    snapshot = generation.freeze(catalog, store.root, store.read())
    manifest = generation.verify_snapshot(snapshot)
    assert generation.files(store.root) == before
    frozen = generation.read(snapshot / "goal.json")
    assert frozen == store.read() and frozen["status"] == "active"
    assert manifest["requested_generations"] == 2 and len(manifest["model_ids"]) == 1
    assert not any(c["passed"] for c in frozen["assessment"]["criteria"].values())

    def fail(state):
        state["cycles"].append({"index": 2, "status": "failed", "phase": "develop"})
        state["status"] = "budget_exhausted"

    store.mutate(fail)
    assert generation.bind_parent(snapshot, store.root) == manifest
    assert generation.records_at_checkpoint(store.read(), generation.read(store.root / "protocol.json")) == []
    assert generation.read(snapshot / "goal.json") == frozen


@pytest.mark.parametrize(
    "attack", ["inflight", "no_reuse", "extra_generation", "changed_progress", "wrong_episode", "symlink"]
)
def test_incomplete_or_unsafe_checkpoint_cannot_be_frozen(tmp_path, attack):
    store, catalog = checkpoint(tmp_path)
    state = store.read()
    if attack == "inflight":
        state["inflight"] = {"index": 2}
    elif attack == "no_reuse":
        state["progress"]["generations"][0].pop("reuse_receipt")
    elif attack == "extra_generation":
        state["progress"]["generations"] *= 3
    elif attack == "changed_progress":
        state["progress"]["changed"] = True
    elif attack == "wrong_episode":
        state["progress"]["generations"][0]["reuse_receipt"] = "episodes/cycle_000/reuse_receipt.json"
        state["cycles"][-1]["progress"] = copy.deepcopy(state["progress"])
    else:
        (store.root / "alias").symlink_to(store.root / "protocol.json")
    atomic_json(store.root / "goal.json", state)
    with pytest.raises(ValueError):
        generation.freeze(catalog, store.root, state)
    assert not (catalog.root / "index").exists()


@pytest.mark.parametrize(
    "attack",
    ["file", "extra_file", "manifest", "parent_prefix", "parent_definition", "parent_protocol", "other_parent"],
)
def test_snapshot_and_parent_tampering_fails(tmp_path, attack):
    store, catalog = checkpoint(tmp_path)
    snapshot = generation.freeze(catalog, store.root, store.read())
    if attack == "file":
        atomic_json(snapshot / "goal.json", {})
    elif attack == "extra_file":
        (snapshot / "injected.py").write_text("pass\n")
    elif attack == "manifest":
        value = generation.read(snapshot / generation.MANIFEST)
        value["requested_generations"] = 1
        atomic_json(snapshot / generation.MANIFEST, value)
    elif attack == "parent_prefix":
        store.mutate(lambda s: s["cycles"][0].update(status="failed"))
    elif attack == "parent_definition":
        store.mutate(lambda s: s["definition"]["goal"].update(objective="Changed"))
    elif attack == "parent_protocol":
        atomic_json(store.root / "protocol.json", {"evolution": {"generations": 1}})
    with pytest.raises(ValueError):
        generation.bind_parent(snapshot, tmp_path / "unrelated" if attack == "other_parent" else store.root)


def test_native_fixture_remains_unpublishable_with_a_valid_snapshot(tmp_path):
    store, catalog = checkpoint(tmp_path)
    atomic_json(store.root / "implementation/source.py", {})
    protocol = generation.read(store.root / "protocol.json")
    protocol.update(implementation={"source.py": file_hash(store.root / "implementation/source.py")}, data_hashes={})
    for phase in ("training", "validation", "future"):
        atomic_json(store.root / "data" / (phase + ".json"), [])
        protocol["data_hashes"][phase] = digest([])
    atomic_json(store.root / "protocol.json", protocol)
    snapshot = generation.freeze(catalog, store.root, store.read())
    with pytest.raises(ValueError, match="Fixture execution"):
        library.validate_model_goal(snapshot, "b" * 64)
    assert not (catalog.root / "index").exists()


def numerical_checkpoint(numerical, tmp_path, monkeypatch):
    root, record, protocol, data, episodes, lifecycle = numerical
    holder, catalog = checkpoint(tmp_path / "metadata")
    state = holder.read()
    mid = record["model_id"]
    model = lifecycle._model(mid)
    experience = ModelExperienceStore(root / "experience")
    identity = experience.publish(lifecycle, mid, summary="Synthetic numerical fixture", parents=[])
    record.update(
        generation=1,
        experience_id=identity,
        source_sha256=model["source_sha256"],
        program_sha256=program_hash(model["recipe"]["source_code"]),
    )
    state["progress"] = {"generations": [record]}
    state["cycles"][-1]["progress"] = copy.deepcopy(state["progress"])
    atomic_json(root / "goal.json", state)
    atomic_json(root / "implementation/source.py", {})
    protocol.update(
        evolution={"schema": 2, "generations": 2},
        case="synthetic",
        source={"synthetic": True},
        implementation={"source.py": file_hash(root / "implementation/source.py")},
        data_hashes={p: digest(rows) for p, rows in data.items()},
    )
    for phase, rows in data.items():
        atomic_json(root / "data" / (phase + ".json"), rows)
    atomic_json(root / "protocol.json", protocol)
    # Only native execution is a fixture. The model's fit, strict validation,
    # source lineage and fresh-session numerical receipt are actually replayed.
    monkeypatch.setattr(library, "_episodes", lambda *args: (episodes, {"model_calls": 0, "tool_calls": 0}))
    catalog = library.ScientistModelLibrary(tmp_path / "numerical_library")
    return root, state, record, catalog, lifecycle


def test_partial_generation_replays_training_lineage_and_reuse(numerical, tmp_path, monkeypatch):
    root, state, record, catalog, _ = numerical_checkpoint(numerical, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="completed"):
        library.validate_model_goal(root, record["model_id"])
    before = generation.files(root)
    report = generation.publish_checkpoint(catalog, root, state)
    assert report["all_published"] and report["new_entries"] == 1
    assert report["new_autonomous_executions"] == 0
    resolved = catalog.resolve(report["outcomes"][0]["entry_id"], replay_training=True)
    assert resolved["origin"]["generation_milestone"]["requested_generations"] == 2
    assert resolved["origin"]["generation_milestone"]["verified_generations"] == 1
    assert generation.files(root) == before
    again = generation.publish_checkpoint(catalog, root, state)
    assert report == again and catalog.catalog()["indexed_entries"] == 1
    assert len(list((catalog.root / "publication_attempts").glob("*.json"))) == 1

    def advance(s):
        s["status"] = "budget_exhausted"
        s["cycles"].append({"index": 2, "status": "failed", "phase": "develop"})

    GoalStore(root).mutate(advance)
    assert catalog.resolve(report["outcomes"][0]["entry_id"], replay_training=True)["packet"] == resolved["packet"]
    snapshot = Path(resolved["origin_goal"])
    snapshot_before = generation.files(snapshot)
    prediction = catalog.predict(
        report["outcomes"][0]["entry_id"],
        [{"id": "HI", "sequence": "HI", "target": 999999}],
        library_tests.SCOPE,
        session_id="synthetic-agent-after-parent-failure",
        mode="science_agent",
    )
    assert prediction["receipt"]["mode"] == "science_agent"
    assert len(prediction["predictions"]) == 1
    assert generation.files(snapshot) == snapshot_before


@pytest.mark.parametrize("attack", ["retired", "threshold", "reuse", "ancestry", "source"])
def test_partial_publication_does_not_relax_model_checks(numerical, tmp_path, monkeypatch, attack):
    root, state, record, catalog, lifecycle = numerical_checkpoint(numerical, tmp_path, monkeypatch)
    if attack == "retired":
        lifecycle.retire(record["model_id"], "Synthetic rejection")
    elif attack == "threshold":
        policy = generation.read(lifecycle.root / "policy.json")
        policy["minimum_skill"] = -1
        atomic_json(lifecycle.root / "policy.json", policy)
    elif attack == "reuse":
        atomic_json(root / record["reuse_receipt"], {"forged": True})
    else:
        record["experience_id" if attack == "ancestry" else "program_sha256"] = "a" * 64
        state["cycles"][-1]["progress"] = copy.deepcopy(state["progress"])
        atomic_json(root / "goal.json", state)
    report = generation.publish_checkpoint(catalog, root, state)
    assert not report["all_published"] and report["rejected_entries"] == 1
    assert catalog.catalog()["indexed_entries"] == 0
    assert not (catalog.root / "generation_index").exists()
    assert report["new_autonomous_executions"] == 0


@pytest.mark.asyncio
async def test_goal_callback_is_after_durable_checkpoint_and_before_next_cycle(tmp_path):
    store = goal_tests.create(tmp_path)
    calls = []

    def callback(state):
        index = len(state["cycles"]) - 1
        assert generation.read(store.root / f"cycle_{index:03d}.json") == state["cycles"][-1]
        assert state["inflight"] is None and state["status"] == "active"
        calls.append(index)
        state["status"] = "completed"  # Detached input cannot complete the saved goal.

    async def cycle(index, *args):
        assert calls == list(range(index))
        (store.root / (["accepted", "reused"][index] + ".json")).write_text("{}")
        return goal_tests.outcome()

    state = await run_goal(store, cycle, lambda s: goal_tests.assessment(store, s), on_checkpoint=callback)
    assert state["status"] == "completed" and calls == [0, 1]
    assert state["usage"]["model_calls"] == 2


@pytest.mark.asyncio
async def test_failed_publication_stops_next_cycle_and_resume_keeps_progress(tmp_path):
    store = goal_tests.create(tmp_path)
    calls, fail = [], [True]

    async def cycle(index, *args):
        calls.append(index)
        (store.root / (["accepted", "reused"][index] + ".json")).write_text("{}")
        return goal_tests.outcome()

    def callback(state):
        if fail[0]:
            raise ValueError("Synthetic publication failure")

    first = await run_goal(store, cycle, lambda s: goal_tests.assessment(store, s), on_checkpoint=callback)
    assert first["status"] == "interrupted" and calls == [0]
    assert first["events"][-1]["event"] == "checkpoint_callback_error"
    fail[0] = False
    store.resume()
    final = await run_goal(store, cycle, lambda s: goal_tests.assessment(store, s), on_checkpoint=callback)
    assert final["status"] == "completed" and calls == [0, 1]
    assert final["usage"]["model_calls"] == 2


def test_campaign_counts_partial_generation_and_evaluates_its_own_packet(numerical, tmp_path, monkeypatch):
    from agent.rsi import campaign_evaluation as evaluation
    from agent.rsi import model_tasks
    from agent.rsi.campaigns.scientist_model_campaign import ModelCampaignStore

    root, state, record, catalog, _ = numerical_checkpoint(numerical, tmp_path, monkeypatch)
    data = {p: generation.read(root / "data" / (p + ".json")) for p in model_tasks.PHASES}
    references = tmp_path / "reference.txt"
    references.write_text("Synthetic campaign integration only")
    assets = tmp_path / "assets"
    bindings = {}
    for phase, rows in data.items():
        path = assets / (phase + ".json")
        atomic_json(path, rows)
        bindings[phase] = {"path": str(path), "sha256": file_hash(path)}
    task = {
        "schema": 1,
        "id": "synthetic_activity",
        "stage": "activity",
        "plugin_id": "generated_python.v1",
        "scope": library_tests.SCOPE,
        "paper": {"title": "Synthetic fixture", "doi": "test:fixture"},
        "references": [{"path": str(references), "sha256": file_hash(references)}],
        "data": bindings,
        "minimum_skill": 0.0,
    }
    goal_budget = {
        k: state["definition"]["goal"][k] for k in ("max_cycles", "model_calls", "tool_calls", "wall_seconds")
    }
    allocation = {
        "id": "case",
        "task": task,
        "profile": state["definition"]["profile"],
        "generations": 2,
        "goal_budget": goal_budget,
    }
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / "worker.py").write_text("# synthetic fixture\n")
    campaign = ModelCampaignStore.create(
        tmp_path / "campaign",
        counting="program_entry.v1",  # Preserve the historical protocol in these fixtures.
        objective="Synthetic generation campaign",
        target_models=2,
        library_root=catalog.root,
        repository=repo,
        implementation={"worker.py": file_hash(repo / "worker.py")},
        allocations=[allocation],
        resources={k: goal_budget[k] for k in ("model_calls", "tool_calls", "wall_seconds")},
    )
    new_root = campaign.goal("case")
    new_root.parent.mkdir(parents=True)
    root.rename(new_root)
    root = new_root
    state["definition"]["admission"] = model_tasks.admission(task)
    state["definition_sha256"] = digest(state["definition"])
    atomic_json(root / "goal.json", state)
    protocol = generation.read(root / "protocol.json")
    protocol.update(
        model_task=model_tasks.admission(task), model_plugin_id=task["plugin_id"], model_task_minimum_skill=0.0
    )
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "model_task.json", task)
    campaign.mutate(lambda s: s["allocations"]["case"].update(status="running"))
    report = generation.publish_checkpoint(catalog, root, state)
    assert report["all_published"]
    campaign.record_generation("case", report)
    campaign.record_generation("case", report)
    current = campaign.read()
    assert len(current["verified_entry_ids"]) == 1
    assert len(current["allocations"]["case"]["generation_publications"]) == 1
    assert current["allocations"]["case"]["status"] == "running"
    assert not campaign.audit_publications()["target_met"]

    def fail(s):
        s["cycles"].append({"index": 2, "status": "failed", "phase": "develop"})
        s["status"] = "budget_exhausted"

    GoalStore(root).mutate(fail)
    campaign.mutate(lambda s: s["allocations"]["case"].update(status="budget_exhausted"))
    assert len(campaign.audit_publications()["verified_entry_ids"]) == 1
    models = evaluation.published_models(campaign, "case", allocation, campaign.read()["allocations"]["case"])
    assert len(models) == 1 and models[0]["model_id"] == record["model_id"]
    assert models[0]["canonical_origin_is_this_goal"]
    assert campaign.read()["allocations"]["case"]["status"] == "budget_exhausted"
    # A valid independent snapshot cannot conceal a changed parent history when
    # crediting the originating campaign.
    GoalStore(root).mutate(lambda s: s["cycles"][0].update(status="failed"))
    with pytest.raises(ValueError, match="prefix"):
        campaign.audit_publications()
    assert campaign.read()["status"] == "publication_invalid"
    assert campaign.read()["verified_entry_ids"] == []
