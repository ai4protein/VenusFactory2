"""Coverage protocol tests; simulated scheduler publications are not real models."""

import copy
import json
from pathlib import Path

import pytest
from test_research_project_catalog import manifest as manifest
from test_research_project_catalog import save, task_of
from test_scientist_model_campaign import checkpoint

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi import scientist_model_campaign as campaign
from agent.rsi.engine.model_coverage import (
    PROGRAM_ENTRIES,
    SCIENTIFIC_SCOPES,
    counting_policy,
    resolve_scope,
    summarize_scopes,
)
from agent.rsi.datasets.research_project_catalog import build_catalog, campaign_plan
from agent.rsi.engine.scientist_model_library import ScientistModelLibrary


def export(manifest, tmp_path, name="manifest"):
    reference = save(tmp_path / f"{name}.json", manifest)
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("dmx", "glm-4-flash", "https://www.dmxapi.cn/v1"),
        Budget(wall_seconds=10, request_seconds=5, model_calls=3, tool_calls=2, output_tokens=4096),
    ).public()
    goal_budget = {"max_cycles": 4, "model_calls": 12, "tool_calls": 8, "wall_seconds": 60}
    plan = campaign_plan(manifest, build_catalog(manifest), profile, goal_budget, reference)
    assert plan["counting_policy"] == SCIENTIFIC_SCOPES
    return plan["allocations"][0]


def variant(manifest, tmp_path, dimension):
    changed = copy.deepcopy(manifest)
    candidate = changed["candidates"][0]
    task = task_of(manifest)
    if dimension == "protein":
        task["scope"]["protein"] = candidate["protein_scope"]["identifier"] = "Another synthetic protein"
    elif dimension == "property":
        task["scope"]["target"] = candidate["property"] = "expression"
        task["stage"] = "expression"
    elif dimension == "scenario":
        candidate["scenario"]["medium"] = "second measured context"
    elif dimension == "unit":
        task["scope"]["unit"] = "fraction"
    elif dimension == "task_name":
        task["id"] = "renamed_task"
        task["scope"]["assay"] = "renamed_assay"
    # Distinct observations permit a new scope where it is scientifically distinct;
    # otherwise different training labels/versions still must not inflate coverage.
    for phase, binding in task["data"].items():
        rows = json.loads(Path(binding["path"]).read_text())
        for row in rows:
            row["target"] += 0.5
            if dimension == "protein":
                row["sequence"] = row["sequence"][:-1] + "C"
        task["data"][phase] = save(tmp_path / f"{dimension}_{phase}.json", rows)
    candidate["task"] = save(tmp_path / f"{dimension}_task.json", task)
    return changed


def test_bound_scope_is_independent_of_algorithm_budget_and_task_program(manifest, tmp_path):
    allocation = export(manifest, tmp_path)
    scope = resolve_scope(allocation)
    other = copy.deepcopy(allocation)
    other.update(id="renamed_allocation", generations=3)
    other["profile"]["model"]["model"] = "unisound-u2-flash"
    assert resolve_scope(other) == scope
    records = [{"entry_id": digest(i), **resolve_scope(a)} for i, a in enumerate([allocation, other])]
    result = summarize_scopes(records)
    assert result["primary_count"] == 1 and result["verified_program_entries"] == 2


@pytest.mark.parametrize(
    "dimension,expected", [("protein", 2), ("property", 2), ("scenario", 2), ("unit", 1), ("task_name", 1)]
)
def test_count_user_defined_axes_but_not_units_or_renamed_tasks(manifest, tmp_path, dimension, expected):
    first = export(manifest, tmp_path)
    second = export(variant(manifest, tmp_path, dimension), tmp_path, name=dimension)
    result = summarize_scopes([{"entry_id": digest(i), **resolve_scope(a)} for i, a in enumerate([first, second])])
    assert result["primary_count"] == expected


def test_transitive_scope_and_measurement_duplicates_cannot_inflate_coverage():
    records = [
        {"entry_id": digest(1), "scope_key": digest("scope1"), "measurement_key": digest("data1")},
        {"entry_id": digest(2), "scope_key": digest("scope2"), "measurement_key": digest("data1")},
        {"entry_id": digest(3), "scope_key": digest("scope2"), "measurement_key": digest("data2")},
        {"entry_id": digest(4), "scope_key": digest("scope3"), "measurement_key": digest("data3")},
    ]
    result = summarize_scopes(records)
    assert result["primary_count"] == 2 and result["collapsed_additional_entries"] == 2
    assert summarize_scopes(records[::-1]) == result
    assert summarize_scopes([])["primary_count"] == 0


@pytest.mark.parametrize(
    "attack",
    [
        "missing_binding",
        "unknown_candidate",
        "changed_manifest",
        "changed_task",
        "missing_task_reference",
        "rebound_data",
    ],
)
def test_reject_unbound_or_mismatched_scope_claims(manifest, tmp_path, attack):
    allocation = export(manifest, tmp_path)
    if attack == "missing_binding":
        allocation.pop("coverage")
    elif attack == "unknown_candidate":
        allocation["coverage"]["candidate_id"] = "unknown"
    elif attack == "changed_manifest":
        Path(allocation["coverage"]["manifest"]["path"]).write_text("{}")
    elif attack == "changed_task":
        allocation["task"]["scope"]["protein"] = "Unrelated protein"
    elif attack == "missing_task_reference":
        allocation["task"]["references"].pop()
    else:
        allocation["task"]["data"]["training"] = allocation["task"]["data"]["validation"]
    with pytest.raises(ValueError):
        resolve_scope(allocation)


def make_store(tmp_path, allocations, monkeypatch, *, target=2):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "worker.py"
    source.write_text("# frozen synthetic executor\n")
    values = {}

    class SimulatedLibrary:
        def __init__(self, root):
            self.real = ScientistModelLibrary(root)
            self.root = self.real.root

        def goal(self, path):
            return self.real.goal(path)

        def publish_goal(self, goal):
            entry = digest(str(goal))
            task = campaign.read(goal / "model_task.json")
            values[entry] = {"origin_goal": str(goal), "origin": {"stage": task["stage"]}}
            return {"all_published": True, "outcomes": [{"status": "published", "entry_id": entry}]}

        def resolve(self, entry, *, replay_training):
            assert replay_training
            return values[entry]

    monkeypatch.setattr(campaign, "ScientistModelLibrary", SimulatedLibrary)
    return campaign.ModelCampaignStore.create(
        tmp_path / "campaign",
        objective="Synthetic scope-counting test",
        target_models=target,
        library_root=tmp_path / "library",
        repository=repository,
        implementation={"worker.py": file_hash(source)},
        allocations=allocations,
        resources={
            "model_calls": 12 * len(allocations),
            "tool_calls": 8 * len(allocations),
            "wall_seconds": 60 * len(allocations),
        },
    )


@pytest.mark.asyncio
async def test_completion_requires_distinct_scopes_not_distinct_programs(manifest, tmp_path, monkeypatch):
    original = export(manifest, tmp_path)
    other = export(variant(manifest, tmp_path, "protein"), tmp_path, "other_manifest")
    allocations = [
        {**copy.deepcopy(a), "id": f"allocation_{i}"} for i, a in enumerate([original, original, other, other])
    ]
    store = make_store(tmp_path, allocations, monkeypatch)

    async def execute(allocation, goal):
        return checkpoint(allocation, goal)

    result = await campaign.run_campaign(store, execute)
    assert result["status"] == "completed"
    assert len(result["verified_entry_ids"]) == 3 and result["verified_scope_count"] == 2
    assert [a["status"] for a in result["allocations"].values()] == ["completed", "completed", "completed", "queued"]
    assert store.audit_publications()["collapsed_additional_entries"] == 1


@pytest.mark.asyncio
async def test_same_scope_remains_incomplete_and_changed_provenance_invalidates_count(manifest, tmp_path, monkeypatch):
    allocation = export(manifest, tmp_path)
    store = make_store(tmp_path, [{**allocation, "id": f"allocation_{i}"} for i in range(2)], monkeypatch)

    async def execute(allocation, goal):
        return checkpoint(allocation, goal)

    result = await campaign.run_campaign(store, execute)
    assert result["status"] == "needs_tasks" and result["verified_scope_count"] == 1
    Path(allocation["coverage"]["manifest"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        store.audit_publications()
    result = store.read()
    assert (
        result["status"] == "publication_invalid"
        and result["verified_scope_count"] == 0
        and not result["verified_entry_ids"]
    )


def test_default_new_campaign_rejects_missing_coverage_and_old_definition_stays_legacy(manifest, tmp_path, monkeypatch):
    allocation = export(manifest, tmp_path)
    allocation.pop("coverage")
    with pytest.raises(ValueError, match="bound project manifest"):
        make_store(tmp_path, [allocation], monkeypatch)
    assert not (tmp_path / "campaign").exists()
    assert counting_policy({}) == PROGRAM_ENTRIES
    with pytest.raises(ValueError, match="Unknown"):
        counting_policy({"counting_policy": "anything"})


@pytest.mark.parametrize("kind", ["duplicate", "cross_partition"])
def test_global_catalog_exclusions_cannot_be_bypassed(manifest, tmp_path, kind):
    if kind == "duplicate":
        second = copy.deepcopy(manifest["candidates"][0])
        task = task_of(manifest)
    else:
        changed = variant(manifest, tmp_path, "scenario")
        second = changed["candidates"][0]
        task = task_of(changed)
        rows = {phase: json.loads(Path(ref["path"]).read_text()) for phase, ref in task["data"].items()}
        first_training = rows["training"][:20]
        rows["training"] = rows["validation"] + rows["training"][20:]
        rows["validation"] = first_training
        groups = {}
        for phase, values in rows.items():
            task["data"][phase] = save(tmp_path / f"swapped_{phase}.json", values)
            groups.update({r["id"]: {"group": r["id"], "phase": phase} for r in values})
        second["groups"] = save(tmp_path / "swapped_groups.json", groups)
    second["id"] = "candidate_two"
    task["id"] = "second_task"
    second["task"] = save(tmp_path / "second_candidate_task.json", task)
    manifest["candidates"].append(second)
    catalog = build_catalog(manifest)
    second_result = next(row for row in catalog["entries"] if row["id"] == "candidate_two")
    assert second_result["status"] == ("duplicate" if kind == "duplicate" else "deferred")
    reference = save(tmp_path / "manifest_with_exclusion.json", manifest)
    task["references"].append(reference)
    with pytest.raises(ValueError, match="catalog admission"):
        resolve_scope({"task": task, "coverage": {"manifest": reference, "candidate_id": "candidate_two"}})
