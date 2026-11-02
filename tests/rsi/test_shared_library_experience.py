"""Real grouped screening with explicit provenance fixtures; no acquisition claims."""

import copy
import json
import sys
from pathlib import Path

import pytest
from test_context_transfer import fixture as context_fixture  # noqa: F401
from test_context_transfer import task as context_task
from test_library_experience import fixture as sequence_fixture  # noqa: F401

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi import library_experience
from agent.rsi import shared_library_experience as shared
from agent.rsi.code_sandbox import CandidateCodeError
from agent.rsi.storage import atomic_json
from agent.science_goal import GoalSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
import run_shared_science_goal as goal


@pytest.fixture
def fixture(tmp_path, request):
    selection, target, donor, _, rows = request.getfixturevalue("context_fixture")
    other, _ = context_task(tmp_path / "second", target["paper"]["doi"], "Second")
    other["id"] = target["id"] + "_second"
    return selection, [target, other], donor, rows


def test_common_projection_contains_one_complete_program_for_two_targets(fixture):
    selection, tasks, donor, _ = fixture
    result = shared.snapshot(selection, tasks)
    assert len(result["manifest"]["target_manifests"]) == 2
    assert len(result["projection"]["entries"]) == 1
    entry = result["projection"]["entries"][0]
    assert entry["source_code"] == donor["packet"]["model"]["recipe"]["source_code"]
    assert shared.projection_bytes(result["projection"]) <= shared.MAX_PROMPT_BYTES
    public = json.dumps(result["projection"])
    assert "private_fitted_weight" not in public and "917.123" not in public
    assert selection["library_root"] not in public


def test_source_ineligible_for_one_target_is_not_exposed_to_any(tmp_path, request):
    selection, target, donor, donor_rows, _ = request.getfixturevalue("sequence_fixture")
    other = copy.deepcopy(target)
    other["id"] += "_second"
    other["scope"]["protein"] += " second"
    future = json.loads(Path(target["data"]["future"]["path"]).read_text())
    future[0]["sequence"] = donor_rows["training"][0]["sequence"]
    path = tmp_path / "second_future.json"
    atomic_json(path, future)
    other["data"]["future"] = {"path": str(path), "sha256": file_hash(path)}
    with pytest.raises(library_experience.IneligibleDonor, match="overlap"):
        shared.snapshot(selection, [target, other])
    report, packet = shared.select_sources(selection, [target, other], keep=1, ranking="declared_order")
    assert report["candidates"][0]["status"] == "ineligible"
    assert not packet["projection"]["entries"] and report["candidate_source_tests"] == 0


def test_grouped_training_screen_refits_each_target_without_publishing(fixture):
    selection, tasks, _, _ = fixture
    report, packet = shared.select_sources(selection, tasks, keep=1)
    assert report["candidate_source_tests"] == 2
    assert len(report["candidates"][0]["feedback"]) == 2
    assert all(row["status"] == "screened" for row in report["candidates"][0]["feedback"])
    assert report["selected"] == selection and report["selected_snapshot_sha256"] == digest(packet)
    assert not report["validation_scored"] and not report["future_scored"]
    assert report["external_model_calls"] == report["new_acquired_models"] == 0


def test_hidden_targets_do_not_change_training_screen(fixture):
    selection, tasks, _, _ = fixture
    before, _ = shared.select_sources(selection, tasks, keep=1)
    for task in tasks:
        for phase in ["validation", "future"]:
            path = Path(task["data"][phase]["path"])
            rows = json.loads(path.read_text())
            for row in rows:
                row["target"] += 10000
            atomic_json(path, rows)
            task["data"][phase]["sha256"] = file_hash(path)
    after, _ = shared.select_sources(selection, tasks, keep=1)
    assert before["candidates"][0]["feedback"] == after["candidates"][0]["feedback"]
    assert before["selected"] == after["selected"]
    assert before["target_admissions"] != after["target_admissions"]


def test_identical_sources_are_screened_once(fixture):
    selection, tasks, _, _ = fixture
    selection["entry_ids"] = ["a" * 64, "b" * 64]
    report, packet = shared.select_sources(selection, tasks, keep=2)
    assert [row["status"] for row in report["candidates"]] == ["eligible", "duplicate_source"]
    assert report["candidate_source_tests"] == 2 and len(packet["projection"]["entries"]) == 1


def test_failed_source_kept_and_corrupt_evidence_is_fatal(fixture, monkeypatch):
    selection, tasks, _, _ = fixture

    def broken(*args):
        raise CandidateCodeError("arbitrary raw private output")

    monkeypatch.setattr(shared.library_program_screen, "_evaluate", broken)
    report, packet = shared.select_sources(selection, tasks, keep=1)
    assert report["candidates"][0]["status"] == "execution_failed"
    assert report["candidate_source_tests"] == 2 and not packet["projection"]["entries"]
    assert "arbitrary raw private output" not in json.dumps(report)
    Path(tasks[0]["data"]["training"]["path"]).write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        shared.select_sources(selection, tasks, keep=1)


def test_context_limit_omits_whole_sources_without_clipping(fixture, monkeypatch):
    selection, tasks, _, _ = fixture
    original = shared.snapshot(selection, tasks)
    size = shared.projection_bytes(original["projection"])
    monkeypatch.setattr(shared, "MAX_PROMPT_BYTES", size - 1)
    with pytest.raises(ValueError, match="prompt budget"):
        shared.snapshot(selection, tasks)
    report, packet = shared.select_sources(selection, tasks, keep=1)
    assert report["candidates"][0]["status"] == "context_budget"
    assert not packet["projection"]["entries"] and report["candidate_source_tests"] == 0


@pytest.mark.parametrize("condition", ["none", "summary", "source"])
def test_freeze_and_dependency_reader_preserve_conditions(tmp_path, fixture, condition):
    selection, tasks, _, _ = fixture
    selection["condition"] = condition
    value = shared.snapshot(selection, tasks)
    root = tmp_path / "goal"
    root.mkdir()
    binding = shared.freeze(root, value)
    state = {"definition": {"admission": {shared.FIELD: binding}}}
    protocol = {"schema": goal.SCHEMA, shared.FIELD: binding}
    assert library_experience.read_frozen(root, state, protocol) == value
    assert shared.verify_goal(root, state, protocol, tasks) == value["projection"]
    if condition == "none":
        assert not value["projection"]["entries"]
    path = root / shared.FIELD / "projection.json"
    changed = json.loads(path.read_text())
    changed["entries"] = [] if changed["entries"] else [{"invented": True}]
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="Frozen shared experience changed"):
        library_experience.read_frozen(root, state, protocol)


def test_real_shared_goal_prepares_bound_history_and_keeps_reuse_clean(tmp_path, fixture):
    selection, tasks, _, _ = fixture
    profile = Profile(
        "science_expert", "expert-kimi-code", ModelBinding("test", "no_api", "https://example.org/v1"), Budget()
    )
    spec = GoalSpec("Synthetic historical-source integration", goal.CRITERIA, max_cycles=4)
    store = goal.prepare(tmp_path / "goal", tasks, profile, spec, library_selection=selection)
    driver = goal.SharedScientistGoal(store, None)
    assert len(driver.library_experience["entries"]) == 1
    assert store.read()["usage"]["model_calls"] == 0
    prompt = shared.append_prompt("base", "develop", driver.library_experience)
    assert (
        driver.library_experience["entries"][0]["source_code"]
        in json.loads(prompt.split("\n")[-1])["entries"][0]["source_code"]
    )
    assert shared.append_prompt("base", "reuse", None) == "base"
    with pytest.raises(ValueError, match="cannot enter model reuse"):
        shared.append_prompt("base", "reuse", driver.library_experience)
    shared.verify_episode(
        store.root,
        store.read(),
        driver.protocol,
        {"phase": "develop", shared.FIELD + "_sha256": digest(driver.library_experience)},
    )
    with pytest.raises(ValueError, match="experience changed"):
        shared.verify_episode(store.root, store.read(), driver.protocol, {"phase": "develop"})
    with pytest.raises(ValueError, match="leaked"):
        shared.verify_episode(
            store.root,
            store.read(),
            driver.protocol,
            {"phase": "reuse", shared.FIELD + "_sha256": digest(driver.library_experience)},
        )


@pytest.mark.parametrize(
    "keep,ranking", [(True, "declared_order"), (0, "declared_order"), (6, "declared_order"), (1, "unknown")]
)
def test_invalid_selection_policy_rejected(fixture, keep, ranking):
    selection, tasks, _, _ = fixture
    with pytest.raises(ValueError):
        shared.select_sources(selection, tasks, keep=keep, ranking=ranking)


@pytest.mark.asyncio
async def test_real_mcp_development_and_reuse_bind_source_prompt_and_native_gate(tmp_path, fixture, monkeypatch):
    import test_shared_science_goal as executor_fixture
    from test_shared_model_publication import synthetic_native_seam

    from agent.rsi.context_transfer import SOURCE_ADAPTATION
    from agent.rsi.shared_model_publication import validate_shared_goal
    from agent.science_goal import run_goal

    selection, tasks, donor, _ = fixture
    selection["adaptation_policy"] = SOURCE_ADAPTATION
    monkeypatch.setattr(executor_fixture, "CODE", donor["packet"]["model"]["recipe"]["source_code"])
    for task in tasks:
        task["plugin_id"] = "generated_context.v2"
        task["scope"]["interface_version"] = task["plugin_id"]
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test", "no_api", "https://example.org/v1"),
        Budget(model_calls=8, tool_calls=8, wall_seconds=120, request_seconds=60),
    )
    spec = GoalSpec(
        "Synthetic shared-source execution",
        goal.CRITERIA,
        max_cycles=4,
        model_calls=32,
        tool_calls=32,
        wall_seconds=900,
    )
    store = goal.prepare(tmp_path / "running_goal", tasks, profile, spec, library_selection=selection)
    calls = executor_fixture.synthetic_executor(monkeypatch)
    delegate = goal.execute

    async def bound(profile, task, runtimes, *, credential):
        lock = {"profile": profile.public(), "task": task.binding()}
        lock["sha256"] = digest(lock)
        result = await delegate(profile, task, runtimes, credential=credential)
        result["profile_sha256"] = lock["sha256"]
        atomic_json(task.output / "profile.lock.json", lock)
        atomic_json(task.output / "run.json", result)
        return result

    monkeypatch.setattr(goal, "execute", bound)
    driver = goal.SharedScientistGoal(store, None)
    state = await run_goal(store, driver.cycle, driver.evaluate)
    assert state["status"] == "completed" and len(state["progress"]["targets"]) == 2
    assert [call["phase"] for call in calls] == ["develop", "reuse"]
    assert "Verified historical program precedents" in calls[0]["prompt"]
    assert "Verified historical program precedents" not in calls[1]["prompt"]
    with pytest.raises(ValueError, match="Fixture execution"):
        validate_shared_goal(store.root)
    synthetic_native_seam(monkeypatch)
    assert len(validate_shared_goal(store.root)) == 2
    path = store.root / "episodes/cycle_000/shared_prompt.json"
    atomic_json(path, json.loads(path.read_text()) + " altered historical instruction")
    with pytest.raises(ValueError, match="prompt changed"):
        validate_shared_goal(store.root)
