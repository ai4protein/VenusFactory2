"""Real retrieval/coverage integration on synthetic source-bound tasks, without LLM calls."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_context_transfer import fixture as context_fixture  # noqa: F401
from test_research_project_catalog import save

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import file_hash
from agent.rsi import library_program_screen, model_tasks
from agent.rsi import screened_research_batch as batch
from agent.rsi.code_sandbox import CandidateCodeError
from agent.rsi.model_coverage import resolve_scope
from agent.rsi.research_project_catalog import SCHEMA, build_catalog
from agent.rsi.scientist_model_campaign import ALL_ALLOCATIONS, ModelCampaignStore
from agent.rsi.storage import atomic_json


@pytest.fixture
def fixture(tmp_path, request):
    selection, target, _, _, parts = request.getfixturevalue("context_fixture")
    target = deepcopy(target)
    group_ref = next(x for x in target["references"] if Path(x["path"]).name == "groups.json")
    evidence = json.loads(Path(group_ref["path"]).read_text())
    assignments = {}
    for phase, rows in parts.items():
        for row in rows:
            row["group"] = row["id"]
            assignments[row["id"]] = {"group": row["group"], "phase": phase}
        target["data"][phase] = save(Path(target["data"][phase]["path"]), rows)
    evidence["observations"] = assignments
    changed_ref = save(Path(group_ref["path"]), evidence)
    target["references"] = [changed_ref if x == group_ref else x for x in target["references"]]
    target["scope"]["grouping"]["evidence_sha256"] = changed_ref["sha256"]
    task_ref = save(tmp_path / "receiver_task.json", target)
    source = target["references"][0]
    manifest = {
        "schema": SCHEMA,
        "target_models": 10000,
        "projects": [
            {
                "id": "project_one",
                "title": "Synthetic source",
                "url": "https://example.org/fixture",
                "study_doi": target["paper"]["doi"],
                "source_evidence": source,
                "license_evidence": source,
                "license": "Synthetic fixture",
            }
        ],
        "candidates": [
            {
                "id": "candidate_one",
                "project_id": "project_one",
                "task": task_ref,
                "protein_scope": {"kind": "protein", "identifier": target["scope"]["protein"]},
                "property": target["scope"]["target"],
                "scenario": target["scope"]["context"],
                "question": "Synthetic preparation integration",
                "groups": save(tmp_path / "catalog_groups.json", assignments),
                "group_basis": "replicate_group",
                "scenario_evidence": source,
            }
        ],
    }
    catalog = build_catalog(manifest)
    assert catalog["summary"]["statuses"] == {"admitted": 1}
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("dmx", "unisound-u2-flash", "https://www.dmxapi.cn/v1", credential_env="DMXAPI_API_KEY"),
        Budget(wall_seconds=10, request_seconds=5, model_calls=3, tool_calls=2),
    )
    declaration = {
        "schema": batch.SCHEMA,
        "parent_manifest": save(tmp_path / "manifest.json", manifest),
        "parent_catalog": save(tmp_path / "catalog.json", catalog),
        "candidate_ids": ["candidate_one"],
        "selection": selection,
        "keep": 1,
        "profile": profile.public(),
        "goal_budget": {"max_cycles": 2, "model_calls": 6, "tool_calls": 4, "wall_seconds": 30},
        "generations": 1,
        "training_baseline_feedback": "none",
        "training_attempt_memory": "cumulative_training_attempts.v2",
    }
    return declaration, target, catalog


def test_real_screen_creates_admissible_campaign_with_exact_scope_binding(fixture, tmp_path):
    request, original, prior = fixture
    saved = deepcopy(request)
    out = tmp_path / "batch"
    receipt = batch.export(request, out)
    assert request == saved
    assert receipt["status"] == "ready" and receipt["allocated_tasks"] == 1
    assert receipt["external_model_calls"] == receipt["new_acquired_models"] == receipt["new_published_models"] == 0
    assert receipt["zero_donor_tasks"] == 0 and receipt["candidate_source_tests"] == 1
    plan = json.loads((out / "campaign_plan.json").read_text())
    allocation = plan["allocations"][0]
    assert plan["execution_policy"] == ALL_ALLOCATIONS
    assert plan["resources"] == {k: request["goal_budget"][k] for k in ("model_calls", "tool_calls", "wall_seconds")}
    assert allocation["id"] == "candidate_one" != allocation["task"]["id"]
    assert allocation["task"]["scope"] == original["scope"] and allocation["task"]["data"] == original["data"]
    assert allocation["task"]["minimum_skill"] == original["minimum_skill"]
    assert allocation["experience"]["entry_ids"] == request["selection"]["entry_ids"]
    assert allocation["task"]["references"][:-2] == original["references"]
    assert allocation["task"]["references"][-2] == receipt["outputs"]["screen/screen.json"]
    original_allocation = {
        "task": {**original, "references": original["references"] + [request["parent_manifest"]]},
        "coverage": {"manifest": request["parent_manifest"], "candidate_id": "candidate_one"},
    }
    assert resolve_scope(allocation)["scope_key"] == resolve_scope(original_allocation)["scope_key"]
    exported_catalog = json.loads((out / "catalog.json").read_text())
    assert exported_catalog["entries"][0]["scope_key"] == prior["entries"][0]["scope_key"]
    for ref in receipt["outputs"].values():
        model_tasks.bound_file(ref)
    root = Path(__file__).resolve().parents[2]
    impl = "src/agent/rsi/screened_research_batch.py"
    store = ModelCampaignStore.create(
        tmp_path / "campaign",
        objective="Synthetic Scientist batch integration",
        target_models=1,
        library_root=request["selection"]["library_root"],
        repository=root,
        implementation={impl: file_hash(root / impl)},
        allocations=plan["allocations"],
        resources=plan["resources"],
        counting=plan["counting_policy"],
        execution=plan["execution_policy"],
    )
    assert store.read()["status"] == "ready"


def test_no_usable_source_preserves_recipient_for_new_code(fixture, tmp_path, monkeypatch):
    request, _, _ = fixture

    def broken(*args):
        raise CandidateCodeError("synthetic candidate failure")

    monkeypatch.setattr(library_program_screen, "_evaluate", broken)
    out = tmp_path / "batch"
    receipt = batch.export(request, out)
    assert receipt["allocated_tasks"] == receipt["zero_donor_tasks"] == 1
    allocation = json.loads((out / "campaign_plan.json").read_text())["allocations"][0]
    assert allocation["experience"]["entry_ids"] == []


@pytest.mark.parametrize("change", ["empty", "too_many", "duplicate", "unknown", "paid", "memory", "generations"])
def test_invalid_requests_fail_before_output(fixture, tmp_path, change):
    request, _, _ = fixture
    if change == "empty":
        request["candidate_ids"] = []
    elif change == "too_many":
        request["candidate_ids"] = [f"task_{i}" for i in range(33)]
    elif change == "duplicate":
        request["candidate_ids"] *= 2
    elif change == "unknown":
        request["extra"] = True
    elif change == "paid":
        request["profile"]["model"]["model"] = "glm-5.2"
    elif change == "memory":
        request["training_attempt_memory"] = "unsupported"
    else:
        request["generations"] = 2
    with pytest.raises((ValueError, PermissionError)):
        batch.export(request, tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


@pytest.mark.parametrize("attack", ["task", "parent"])
def test_mid_preparation_changes_leave_failure_evidence_and_no_plan(fixture, tmp_path, monkeypatch, attack):
    request, _, _ = fixture
    original = library_program_screen.export

    def altered(*args):
        report = original(*args)
        if attack == "parent":
            Path(request["parent_catalog"]["path"]).write_text("{}")
        else:
            handoff_path = Path(args[1]) / "handoff.json"
            handoff = json.loads(handoff_path.read_text())
            record = handoff["tasks"][0]
            task = json.loads(Path(record["task"]["path"]).read_text())
            task["minimum_skill"] = 0.25
            record["task"] = save(Path(record["task"]["path"]), task)
            atomic_json(handoff_path, handoff)
        return report

    monkeypatch.setattr(library_program_screen, "export", altered)
    out = tmp_path / "batch"
    with pytest.raises(ValueError):
        batch.export(request, out)
    assert not (out / "campaign_plan.json").exists() and not (out / "preparation.json").exists()
    assert json.loads((out / "failure.json").read_text())["status"] == "preparation_failed"


def test_existing_output_is_never_replaced(fixture, tmp_path):
    request, _, _ = fixture
    out = tmp_path / "batch"
    out.mkdir()
    (out / "marker").write_text("existing run")
    with pytest.raises(ValueError, match="exists"):
        batch.export(request, out)
    assert (out / "marker").read_text() == "existing run"


def test_selected_source_change_after_screening_is_rejected(fixture, tmp_path, monkeypatch):
    from agent.rsi import library_experience

    request, _, _ = fixture
    original_export = library_program_screen.export
    original_snapshot = library_experience.snapshot
    completed = False

    def export_then_change(*args):
        nonlocal completed
        result = original_export(*args)
        completed = True
        return result

    def changed_snapshot(*args):
        result = original_snapshot(*args)
        if completed:
            result["manifest"]["donors"][0]["source_sha256"] = "f" * 64
        return result

    monkeypatch.setattr(library_program_screen, "export", export_then_change)
    monkeypatch.setattr(library_experience, "snapshot", changed_snapshot)
    with pytest.raises(ValueError, match="Selected source changed"):
        batch.export(request, tmp_path / "batch")
    assert not (tmp_path / "batch/campaign_plan.json").exists()
