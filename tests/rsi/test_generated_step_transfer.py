"""Cross-goal capability tests with simulated models and actual isolated code."""

import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from test_paper_step_goal import system as system
from test_paper_step_tools import lifecycle as lifecycle
from test_paper_step_tools import simulated_session

from agent.harness import Profile
from agent.rsi import paper_step_goal as goal
from agent.rsi import paper_step_transfer as transfer
from agent.rsi.generated_step_library import GeneratedStepLibrary, validate_completed_step
from agent.rsi.paper_step_sessions import read
from agent.rsi.storage import atomic_json, sha256_file
from agent.science_goal import GoalSpec, run_goal


@pytest_asyncio.fixture
async def donor(system, monkeypatch):
    store, driver, profile = system
    calls = []

    async def execute(folder, profile, runtimes, credential):
        calls.append(folder)
        declaration = read(folder / "phase.json")
        task = simulated_session(
            folder,
            phase=declaration["phase"],
            capability_id=declaration["capability_id"],
            session_id="simulated-" + folder.parent.parent.name + "-" + folder.name,
            profile=profile,
        )
        return read(task.output / "run.json")

    monkeypatch.setattr(goal, "execute_phase", execute)
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "completed"
    library = GeneratedStepLibrary(store.root.parent / "library")
    entry = library.publish(store.root)
    return SimpleNamespace(store=store, driver=driver, profile=profile, library=library, entry=entry, calls=calls)


def definition(donor):
    value = deepcopy(donor.driver.definition)
    value["paper"] = {"title": "Recipient numerical paper fixture", "doi": "test:recipient"}
    ref = donor.store.root.parent / "recipient_reference.txt"
    if not ref.exists():
        ref.write_text("Recipient independent sum reference")
    value["references"] = {str(ref): sha256_file(ref)}
    for cases in value["cases"].values():
        for case in cases:
            case["inputs"] = [x + 100 for x in case["inputs"]]
            case["expected"] = {"sum": sum(case["inputs"])}
    return value


def recipient(donor, *, value=None, name="recipient"):
    profile = Profile.from_dict({**donor.profile.public(), "harness": "deepseek-harness"})
    spec = GoalSpec(
        "Import immutable source and independently verify fresh inputs",
        transfer.CRITERIA,
        max_cycles=6,
        model_calls=96,
        tool_calls=32,
        wall_seconds=7200,
    )
    store = transfer.create_transfer_goal(
        donor.store.root.parent / name,
        value or definition(donor),
        profile,
        spec,
        donor.library,
        condition="persistent_agent_ablation",
    )
    return store, transfer.TransferredStepGoal(store, "unused-test-credential", {})


@pytest.mark.asyncio
async def test_source_accumulates_across_goals_with_recipient_acceptance_and_dedup(donor):
    original = donor.library.resolve(donor.entry)
    assert original["origin_record_kind"] == "simulated_model_records"
    assert not original["autonomous_scientific_success"]
    store, driver = recipient(donor)
    initial = driver.evaluate(store.read())["criteria"]
    assert initial["verified_source_imported"]["passed"]
    assert not initial["recipient_acceptance_passed"]["passed"]
    assert not initial["fresh_inputs_reused"]["passed"]
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "completed", state
    assert len(state["cycles"]) == 3 and state["usage"]["model_calls"] == 2 and state["usage"]["tool_calls"] == 6
    assert state["cycles"][0]["usage"]["model_calls"] == 0
    packet = driver.lifecycle.capability(state["progress"]["capability_id"])
    assert packet["source_code"] == original["packet"]["source_code"]
    assert packet["capability_id"] != original["packet"]["capability_id"]
    assert packet["declared_origin"] == "library_import:" + donor.entry
    imported = read(store.root / state["progress"]["development_folder"] / "host_acceptance.json")
    assert imported["imported_source"] and not imported["new_code_generated"]
    assert imported["assessment"]["passed"]
    for relative in state["progress"]["reuse"].values():
        folder = store.root / relative
        events = [json.loads(line) for line in (folder / "runtime/events.jsonl").read_text().splitlines()]
        assert not any(e.get("tool_name", "").endswith("evaluate_paper_step") for e in events)
    assert donor.library.publish(store.root) == donor.entry
    assert len(list((donor.library.root / "generated_entries").glob("*.json"))) == 1
    assert len(donor.calls) == 5


@pytest.mark.asyncio
async def test_failed_recipient_acceptance_retains_source_without_model_retry(donor):
    value = definition(donor)
    value["cases"]["acceptance"][0]["expected"]["sum"] += 1
    store, driver = recipient(donor, value=value)
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "waiting_for_input" and state["progress"]["hold"]
    assert state["usage"]["model_calls"] == 0 and len(donor.calls) == 3
    assert not state["assessment"]["criteria"]["recipient_acceptance_passed"]["passed"]
    receipt = read(store.root / "episodes/cycle_000/host_acceptance.json")
    assert not receipt["assessment"]["passed"] and "capability_id" not in receipt
    store.resume()
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "waiting_for_input" and len(donor.calls) == 3
    assert len(list((driver.lifecycle.root / "attempts").glob("*/reservation.json"))) == 1


@pytest.mark.parametrize("change", ["inputs", "interface", "scope", "tolerance"])
def test_incompatible_transfer_rejected_before_any_execution(donor, change):
    value = definition(donor)
    if change == "inputs":
        value["cases"]["reuse"][0]["inputs"] = donor.driver.definition["cases"]["development"][0]["inputs"]
    elif change == "interface":
        value["interface"]["description"] = "Different output meaning"
    elif change == "scope":
        value["scope"]["covered_steps"] = ["Another scientific computation"]
    else:
        value["tolerance"]["atol"] = 1e-8
    assert donor.library.find(value) == []
    with pytest.raises(ValueError):
        recipient(donor, value=value)
    assert len(donor.calls) == 3


@pytest.mark.parametrize("change", ["goal", "source", "requests", "reference", "archive", "entry", "missing_admission"])
def test_changed_origin_cannot_be_resolved_or_silently_repaired(donor, change):
    root = donor.store.root
    state = donor.store.read()
    development = root / state["progress"]["development_folder"]
    if change == "goal":
        donor.store.mutate(lambda s: s["usage"].update(model_calls=99))
    elif change == "source":
        p = development / "workspace/step_candidate.json"
        value = read(p)
        value["source_code"] += "\n# changed"
        atomic_json(p, value)
    elif change == "requests":
        (development / "runtime/model_requests.jsonl").unlink()
    elif change == "reference":
        Path(next(iter(donor.driver.definition["references"]))).write_text("changed")
    elif change == "archive":
        (root / "implementation/src/agent/rsi/paper_step.py").write_text("changed")
    elif change == "missing_admission":
        p = root / "host_private/admission.json"
        p.chmod(0o644)
        p.unlink()
    else:
        p = donor.library.root / "generated_entries" / f"{donor.entry}.json"
        p.chmod(0o644)
        value = read(p)
        value["origin_record_kind"] = "recorded_model_session"
        atomic_json(p, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        donor.library.resolve(donor.entry)
    if change == "missing_admission":
        assert not (root / "host_private/admission.json").exists()


def test_owner_scope_and_symlinks_are_rejected(donor, tmp_path):
    with pytest.raises(ValueError):
        donor.library.goal(donor.store.root.parent.parent / "outside_owner")
    alias = donor.store.root.parent / "alias"
    alias.symlink_to(donor.store.root, target_is_directory=True)
    with pytest.raises(ValueError):
        donor.library.publish(alias)
    library_alias = donor.store.root.parent / "library_alias"
    library_alias.symlink_to(donor.library.root, target_is_directory=True)
    with pytest.raises(ValueError):
        GeneratedStepLibrary(library_alias)


def test_incomplete_source_goal_cannot_publish(donor):
    donor.store.mutate(lambda s: s.update(status="active"))
    with pytest.raises(ValueError, match="completed original"):
        validate_completed_step(donor.store.root)


@pytest.mark.asyncio
async def test_import_reservation_blocks_recomputation_after_lost_checkpoint(donor):
    store, driver = recipient(donor)
    folder = store.root / "episodes/cycle_009"
    folder.mkdir(parents=True)
    transfer.accept_import(store, folder)
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "waiting_for_input" and state["usage"]["model_calls"] == 0
    assert state["progress"]["hold"]["path"] == "episodes/cycle_009/host_acceptance_reservation.json"
    assert len(list((driver.lifecycle.root / "attempts").glob("*/reservation.json"))) == 1


@pytest.mark.asyncio
async def test_cancellation_reaps_import_worker_without_credentials(donor, monkeypatch, tmp_path):
    store, driver = recipient(donor)
    folder = tmp_path / "cancel_worker"
    folder.mkdir()
    original = asyncio.create_subprocess_exec
    processes = []

    async def spawn(*args, **kwargs):
        assert set(kwargs["env"]) == {"PATH", "LANG", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"}
        proc = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(transfer.import_worker(store, folder))
    for _ in range(500):
        if (folder / "host_worker.json").exists():
            break
        await asyncio.sleep(0.01)
    assert processes
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode == -9
    assert read(folder / "host_worker_termination.json") == {"interrupted": True, "returncode": -9}


@pytest.mark.asyncio
async def test_independent_regeneration_of_identical_code_keeps_one_capability(donor):
    spec = GoalSpec(
        "Independently generate the same numerical fixture",
        goal.CRITERIA,
        max_cycles=6,
        model_calls=96,
        tool_calls=32,
        wall_seconds=7200,
    )
    store = goal.create_step_goal(
        donor.store.root.parent / "another_generator",
        donor.driver.definition,
        donor.profile,
        spec,
        condition="persistent_agent_ablation",
    )
    driver = goal.PaperStepGoal(store, "unused-test-credential", {})
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "completed"
    assert donor.library.publish(store.root) == donor.entry
    assert donor.library.publish(store.root) == donor.entry
    assert len(list((donor.library.root / "generated_entries").glob("*.json"))) == 1
    supporting = list((donor.library.root / "supporting_origins" / donor.entry).glob("*.json"))
    assert len(supporting) == 1
    assert read(supporting[0])["origin_goal"] == "another_generator"
    assert len(donor.calls) == 6


@pytest.mark.asyncio
async def test_recipient_cannot_reuse_a_donor_session_identity(donor, monkeypatch):
    store, driver = recipient(donor)
    donor_session = donor.library.resolve(donor.entry)["sessions"][1]

    async def execute(folder, profile, runtimes, credential):
        declaration = read(folder / "phase.json")
        task = simulated_session(
            folder, phase="reuse", capability_id=declaration["capability_id"], session_id=donor_session, profile=profile
        )
        return read(task.output / "run.json")

    monkeypatch.setattr(goal, "execute_phase", execute)
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "interrupted"
    assert state["events"][-1]["event"] == "assessment_error"
    assert not state["assessment"]["criteria"]["fresh_inputs_reused"]["passed"]
    with pytest.raises(ValueError):
        donor.library.publish(store.root)
