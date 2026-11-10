"""Real fits/MCP and storage, with an explicit synthetic native-audit seam.

The unpatched native gate must reject these fixtures. No external API is used,
and no fixture publication represents an autonomous scientific model.
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import pytest_asyncio
import test_shared_science_goal as goal_fixture
from test_shared_model_program import CODE, task_pair

from agent.rsi import context_transfer, library_experience
from agent.rsi import scientist_model_library as library
from agent.rsi.engine.model_origin_view import origin_view
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.shared_model_publication import validate_shared_goal
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.science_goal import run_goal


def read(path):
    return json.loads(path.read_text())


def set_doi(task, doi):
    task["paper"]["doi"] = doi
    reference = task["references"][1]
    path = Path(reference["path"])
    value = read(path)
    value["study_doi"] = doi
    atomic_json(path, value)
    reference["sha256"] = sha256_file(path)
    task["scope"]["grouping"]["evidence_sha256"] = reference["sha256"]


async def run_bound_goal(tmp_path, monkeypatch, **options):
    original_pair = goal_fixture.task_pair

    def pair(path):
        tasks = original_pair(path)
        for task in tasks:
            set_doi(task, "10.9999/synthetic_shared_donor")
        return tasks

    monkeypatch.setattr(goal_fixture, "task_pair", pair)
    store, driver = goal_fixture.prepared(tmp_path, **options)
    calls = goal_fixture.synthetic_executor(monkeypatch)
    delegate = goal_fixture.shared.execute

    async def bound_execute(profile, task, runtimes, *, credential):
        lock = {"profile": profile.public(), "task": task.binding()}
        lock["sha256"] = digest(lock)
        result = await delegate(profile, task, runtimes, credential=credential)
        result["profile_sha256"] = lock["sha256"]
        atomic_json(task.output / "profile.lock.json", lock)
        atomic_json(task.output / "run.json", result)
        return result

    monkeypatch.setattr(goal_fixture.shared, "execute", bound_execute)
    state = await run_goal(store, driver.cycle, driver.evaluate)
    assert len(calls) == len(state["cycles"])
    return store, driver, state


@pytest_asyncio.fixture
async def completed(tmp_path, monkeypatch):
    result = await run_bound_goal(tmp_path, monkeypatch)
    assert result[2]["status"] == "completed" and len(result[2]["cycles"]) == 2
    return result


def synthetic_native_seam(monkeypatch):
    """Only the native authenticity gate is stubbed; all shared/model checks run."""

    def episodes(root, state, protocol):
        records = {
            f"episodes/cycle_{c['index']:03d}": read(root / f"episodes/cycle_{c['index']:03d}/execution/run.json")
            for c in state["cycles"]
            if c["status"] == "completed"
        }
        usage = {name: sum(c["usage"][name] for c in state["cycles"]) for name in ["model_calls", "tool_calls"]}
        return records, usage

    monkeypatch.setattr(library, "_episodes", episodes)


@pytest.mark.asyncio
async def test_native_proof_cannot_be_replaced_by_synthetic_executor(completed):
    store, _, state = completed
    identity = state["progress"]["targets"]["synthetic_0"]["model_id"]
    with pytest.raises(ValueError, match="publication auditor|Fixture execution"):
        library.validate_model_goal(store.root, identity)
    assert not (store.root.parent / "library/index").exists()


@pytest.mark.asyncio
async def test_shared_publication_resolution_agent_prediction_and_donor_projection(completed, tmp_path, monkeypatch):
    store, driver, state = completed
    synthetic_native_seam(monkeypatch)
    shared = library.ScientistModelLibrary(tmp_path / "library")
    before = sha256_file(store.root / "goal.json")
    result = shared.publish_goal(store.root)
    assert result["all_published"] and result["new_entries"] == 2 and result["verified_shared_goals"] == 1
    assert result["shared_usage"]["model_calls"] == 2
    assert shared.catalog()["distinct_scopes"] == 2 and shared.catalog()["distinct_programs"] == 1
    again = shared.publish_goal(store.root)
    assert again["new_entries"] == 0 and again["duplicate_entries"] == 2
    views = []
    for outcome in result["outcomes"]:
        verified = shared.resolve(outcome["entry_id"], replay_training=True)
        assert verified["origin"]["shared_execution_id"] == result["shared_execution_id"]
        assert verified["origin"]["usage_attribution"] == "shared_goal_total_count_once"
        view = origin_view(verified)
        views.append(view)
        task_id = verified["origin"]["task_id"]
        assert view["task"] == driver.by_id[task_id] and view["data"] == driver.data[task_id]
        prediction = shared.predict(
            outcome["entry_id"],
            driver.data[task_id]["future"],
            view["task"]["scope"],
            session_id="synthetic_agent_new_session",
        )
        assert prediction and sha256_file(store.root / "goal.json") == before
    target = task_pair(tmp_path / "recipient")[0]
    target["plugin_id"] = "generated_context.v2"
    target["scope"]["interface_version"] = "generated_context.v2"
    set_doi(target, "10.9999/synthetic_separate_recipient")
    request = {
        "condition": "source",
        "library_root": str(shared.root),
        "entry_ids": [x["entry_id"] for x in result["outcomes"]],
        "overlap_policy": context_transfer.POLICY,
    }
    projection = library_experience.snapshot(request, target)
    assert len(projection["projection"]["entries"]) == 2
    assert {x["source_code"] for x in projection["projection"]["entries"]} == {CODE}
    serialized = json.dumps(projection["projection"])
    assert "model_state" not in serialized and "1234567" not in serialized and "training.json" not in serialized
    assert [x["data_hashes"] for x in projection["manifest"]["donors"]] == [v["data_hashes"] for v in views]
    set_doi(target, "10.9999/synthetic_shared_donor")
    with pytest.raises(library_experience.IneligibleDonor, match="study overlaps"):
        library_experience.snapshot(request, target)


@pytest.mark.asyncio
async def test_shared_audit_rejects_modified_scope_history_predictions_and_requests(completed, monkeypatch):
    store, _, state = completed
    synthetic_native_seam(monkeypatch)
    identity = state["progress"]["targets"]["synthetic_0"]["model_id"]
    verified = validate_shared_goal(store.root, identity)
    assert verified["origin"]["task_id"] == "synthetic_0"
    paths = [
        "episodes/cycle_000/workspace/candidate.json",
        "episodes/cycle_001/workspace/targets/synthetic_0/predictions.json",
        "episodes/cycle_000/shared_training_history.json",
        "targets/synthetic_0/data/training.json",
        "episodes/cycle_000/shared_prompt.json",
    ]
    for relative in paths:
        path = store.root / relative
        original = path.read_bytes()
        value = read(path)
        if isinstance(value, dict):
            value["tampered"] = True
        elif isinstance(value, list):
            value.pop()
        else:
            value += " unrecorded prompt change"
        atomic_json(path, value)
        try:
            with pytest.raises((ValueError, KeyError)):
                validate_shared_goal(store.root, identity)
        finally:
            path.write_bytes(original)
    path = store.root / "episodes/cycle_000/execution/events.jsonl"
    original = path.read_bytes()
    events = [json.loads(x) for x in path.read_text().splitlines()]
    for event in events:
        if event.get("tool_name", "").endswith("evaluate_model_code"):
            event["args"]["source_code"] += "\n# not actually executed\n"
    path.write_text("".join(json.dumps(x) + "\n" for x in events))
    with pytest.raises(ValueError, match="actual request"):
        validate_shared_goal(store.root, identity)
    path.write_bytes(original)
    path = store.root / "goal.json"
    original = path.read_bytes()
    changed = deepcopy(state)
    changed["progress"]["targets"]["invented"] = deepcopy(changed["progress"]["targets"]["synthetic_0"])
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="omitted allocated"):
        validate_shared_goal(store.root, identity)
    path.write_bytes(original)


@pytest.mark.asyncio
async def test_resolved_shared_origin_rejects_later_artifact_changes(completed, tmp_path, monkeypatch):
    store, _, _ = completed
    synthetic_native_seam(monkeypatch)
    shared = library.ScientistModelLibrary(tmp_path / "library")
    published = shared.publish_goal(store.root)
    path = store.root / "episodes/cycle_001/workspace/targets/synthetic_0/predictions.json"
    rows = read(path)
    rows[0]["prediction"] += 1
    atomic_json(path, rows)
    with pytest.raises(ValueError):
        shared.resolve(published["outcomes"][0]["entry_id"])


@pytest.mark.asyncio
async def test_exhausted_shared_goal_publishes_only_qualified_reused_targets(tmp_path, monkeypatch):
    store, _, state = await run_bound_goal(tmp_path, monkeypatch, reject_second=True, max_cycles=4)
    assert state["status"] == "budget_exhausted"
    synthetic_native_seam(monkeypatch)
    shared = library.ScientistModelLibrary(tmp_path / "library")
    before = sha256_file(store.root / "goal.json")
    result = shared.publish_goal(store.root)
    assert result["all_published"] and not result["all_allocated_scopes_published"]
    assert result["new_entries"] == result["verified_target_scopes"] == 1
    assert result["allocated_scopes"] == 2 and result["shared_usage"]["model_calls"] == 4
    assert result["parent_goal_status"] == "budget_exhausted"
    assert result["unpublished_targets"] == [{"task_id": "synthetic_1", "status": "not_accepted_within_budget"}]
    verified = shared.resolve(result["outcomes"][0]["entry_id"], replay_training=True)
    assert verified["origin"]["shared_target_count"] == 2
    assert verified["origin"]["shared_published_target_count"] == 1
    assert verified["origin"]["parent_goal_status"] == "budget_exhausted"
    assert sha256_file(store.root / "goal.json") == before
    rejected_outcome = state["cycles"][0]["target_outcomes"]["synthetic_1"]
    assert rejected_outcome["rejection_reasons"] == ["performance_threshold"]
    original = (store.root / "goal.json").read_bytes()
    changed_reasons = deepcopy(state)
    changed_reasons["cycles"][0]["target_outcomes"]["synthetic_1"]["rejection_reasons"] = [
        "prediction_not_repeatable"
    ]
    atomic_json(store.root / "goal.json", changed_reasons)
    try:
        with pytest.raises(ValueError, match="Shared per-target outcomes changed"):
            validate_shared_goal(store.root)
    finally:
        (store.root / "goal.json").write_bytes(original)
    rejected = rejected_outcome["model_id"]
    with pytest.raises(ValueError, match="not a qualified"):
        validate_shared_goal(store.root, rejected)
    changed = deepcopy(state)
    changed["definition"]["goal"]["max_cycles"] = 5
    changed["definition_sha256"] = digest(changed["definition"])
    atomic_json(store.root / "goal.json", changed)
    with pytest.raises(ValueError, match="frozen budget"):
        validate_shared_goal(store.root)


@pytest.mark.asyncio
async def test_accepted_target_without_fresh_reuse_is_not_publishable(tmp_path, monkeypatch):
    store, _, state = await run_bound_goal(tmp_path, monkeypatch, max_cycles=1)
    assert state["status"] == "budget_exhausted"
    assert len(state["progress"]["targets"]) == 2
    synthetic_native_seam(monkeypatch)
    shared = library.ScientistModelLibrary(tmp_path / "library")
    with pytest.raises(ValueError, match="No independently accepted and freshly reused"):
        shared.publish_goal(store.root)
    assert shared.catalog()["distinct_scopes"] == 0


@pytest.mark.asyncio
async def test_compact_inspection_and_scientific_receipts_must_remain_complete(completed, monkeypatch):
    store, _, _ = completed
    synthetic_native_seam(monkeypatch)
    path = store.root / "episodes/cycle_000/execution/events.jsonl"
    original = path.read_bytes()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    names = {row["tool_call_id"]: row["tool_name"] for row in events if row["kind"] == "tool_call_start"}
    for row in events:
        if row["kind"] == "tool_result" and names[row["tool_call_id"]].endswith("inspect_model_code_task"):
            row["output"] = row["output"][:100]
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    with pytest.raises(ValueError):
        validate_shared_goal(store.root)
    events = [json.loads(line) for line in original.decode().splitlines()]
    for row in events:
        if row["kind"] == "tool_result" and names[row["tool_call_id"]].endswith("evaluate_model_code"):
            row["output"] = row["output"][:100]
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    try:
        with pytest.raises(ValueError):
            validate_shared_goal(store.root)
    finally:
        path.write_bytes(original)


@pytest.mark.asyncio
async def test_legacy_shared_history_retains_its_original_publication_semantics(tmp_path, monkeypatch):
    from agent.rsi.engine.shared_goal_history import LEGACY_POLICY

    store, _, state = await run_bound_goal(tmp_path, monkeypatch, history_policy=LEGACY_POLICY)
    assert state["status"] == "completed"
    synthetic_native_seam(monkeypatch)
    history = read(store.root / "episodes/cycle_000/shared_training_history.json")
    assert history == {"policy": LEGACY_POLICY, "attempts": []}
    prompt = read(store.root / "episodes/cycle_000/shared_prompt.json")
    assert "bounded training-only projection" not in prompt
    result = library.ScientistModelLibrary(tmp_path / "library").publish_goal(store.root)
    assert result["all_published"] and result["new_entries"] == 2


@pytest.mark.asyncio
async def test_harness_repeat_notice_is_separate_from_validated_scientific_receipt(completed, tmp_path, monkeypatch):
    from agent.harness.events import KIMI_REPEAT_NOTICE, audit_trace, normalize_kimi_tool_result

    store, _, _ = completed
    synthetic_native_seam(monkeypatch)
    path = store.root / "episodes/cycle_001/execution/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    starts = {e["tool_call_id"]: e for e in events if e["kind"] == "tool_call_start"}
    changed = 0
    for row in events:
        if row["kind"] == "tool_result" and starts[row["tool_call_id"]]["tool_name"].endswith(
            "discover_generated_models"
        ):
            raw = row["output"] + KIMI_REPEAT_NOTICE
            row.update(normalize_kimi_tool_result(raw, is_error=row["is_error"]))
            assert row["raw_output"] == raw and json.loads(row["output"])
            changed += 1
    assert changed > 0
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    allowed = {row["tool_name"].removeprefix("mcp__venusfactory__") for row in starts.values()}
    assert audit_trace(path, allowed, len(starts))["passed"]
    result = library.ScientistModelLibrary(tmp_path / "library").publish_goal(store.root)
    assert result["all_published"] and result["new_entries"] == 2
