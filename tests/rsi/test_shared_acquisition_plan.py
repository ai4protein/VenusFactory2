"""Real source-binding tests on synthetic data; no scientific models acquired."""

import copy
import json
from pathlib import Path

import pytest
import test_research_project_catalog as fixtures

from agent.rsi import shared_acquisition_plan as planner
from agent.rsi.datasets.research_project_catalog import build_catalog

manifest = fixtures.manifest


def make_manifest(manifest, tmp_path, counts=(3,)):
    result = copy.deepcopy(manifest)
    source = result["candidates"][0]
    template = fixtures.task_of(result)
    project = result["projects"][0]
    result["candidates"], result["projects"] = [], []
    index = 0
    for p, count in enumerate(counts):
        pr = copy.deepcopy(project)
        pr["id"] = f"project_{p}"
        pr["study_doi"] = f"test:fixture_{p}"
        result["projects"].append(pr)
        for _ in range(count):
            c, task = copy.deepcopy(source), copy.deepcopy(template)
            c["id"] = f"candidate_{index:03d}"
            c["project_id"] = pr["id"]
            task["id"] = f"task_{index:03d}"
            task["paper"]["doi"] = pr["study_doi"]
            c["protein_scope"]["identifier"] = task["scope"]["protein"] = f"Synthetic protein {index}"
            for phase, reference in template["data"].items():
                rows = json.loads(Path(reference["path"]).read_text())
                for row in rows:
                    row["target"] += index * 0.125
                task["data"][phase] = fixtures.save(tmp_path / f"{index}_{phase}.json", rows)
            c["task"] = fixtures.save(tmp_path / f"task_{index}.json", task)
            result["candidates"].append(c)
            index += 1
    return result


def prepared(m, tmp_path):
    return (
        fixtures.save(tmp_path / "manifest_bound.json", m),
        fixtures.save(tmp_path / "catalog_bound.json", build_catalog(m)),
    )


def plan(m, tmp_path, **kwargs):
    args = {"projects": [p["id"] for p in m["projects"]], "target_count": len(m["candidates"])}
    args.update(kwargs)
    return planner.plan_acquisition(*prepared(m, tmp_path), **args)


def tasks_of(result):
    return [t for batch in result["batches"] for t in batch["tasks"]]


def test_real_admission_and_homogeneous_round_robin(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path, (34, 3))
    result = plan(m, tmp_path, target_count=35)
    assert [len(b["tasks"]) for b in result["batches"]] == [32, 3]
    assert result["selection_counts"] == {"project_0": 32, "project_1": 3}
    assert result["available_scopes"] == 37 and result["remaining_unallocated_scopes"] == 2
    assert all(len({t["paper"]["doi"] for t in b["tasks"]}) == 1 for b in result["batches"])
    assert result["new_autonomous_models"] == result["external_model_calls"] == 0
    assert result["final_labels_scored"] is False and result["completion_claim"] is False


def test_failed_queued_and_registered_scopes_are_excluded(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path, (4,))
    tasks = [json.loads(Path(c["task"]["path"]).read_text()) for c in m["candidates"]]
    historical = fixtures.save(
        tmp_path / "prior.json",
        {"batches": [{"status": "failed", "tasks": [tasks[0]]}, {"status": "queued", "tasks": [tasks[1]]}]},
    )
    # Changed interface/source/training metadata cannot create another scope.
    key = {
        "plugin_id": tasks[2]["plugin_id"],
        "scope": tasks[2]["scope"],
        "program_sha256": "another_program",
        "training_sha256": "another_training_set",
    }
    registered = fixtures.save(
        tmp_path / "registered.json", {"indexed_entries": 1, "entries": [{"entry_id": "historical", "key": key}]}
    )
    result = plan(m, tmp_path, target_count=1, previous_panels=[historical], registered_catalogs=[registered])
    assert [t["id"] for t in tasks_of(result)] == ["task_003"]
    assert result["excluded_scope_count"] == 3
    assert result["skipped"] == {"previously_registered_or_allocated": 3}


def test_selection_is_independent_of_future_targets(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path)
    first = plan(m, tmp_path, target_count=2)
    for c in m["candidates"]:
        task = json.loads(Path(c["task"]["path"]).read_text())
        path = Path(task["data"]["future"]["path"])
        rows = json.loads(path.read_text())
        for row in rows:
            row["target"] += 5000
        task["data"]["future"] = fixtures.save(path, rows)
        c["task"] = fixtures.save(Path(c["task"]["path"]), task)
    second = plan(m, tmp_path, target_count=2)
    assert [t["id"] for t in tasks_of(first)] == [t["id"] for t in tasks_of(second)]
    assert [r["scope_key"] for r in first["source_bindings"]] == [r["scope_key"] for r in second["source_bindings"]]


def test_forged_catalog_rejected(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path)
    a, b = prepared(m, tmp_path)
    report = json.loads(Path(b["path"]).read_text())
    report["entries"][0]["status"] = "deferred"
    b = fixtures.save(Path(b["path"]), report)
    with pytest.raises(ValueError, match="Catalog or source"):
        planner.plan_acquisition(a, b, projects=["project_0"], target_count=1)


def test_changed_data_cannot_reuse_admission(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path)
    a, b = prepared(m, tmp_path)
    task = json.loads(Path(m["candidates"][0]["task"]["path"]).read_text())
    Path(task["data"]["training"]["path"]).write_text("[]")
    with pytest.raises(ValueError, match="Catalog or source"):
        planner.plan_acquisition(a, b, projects=["project_0"], target_count=1)


def test_mutated_binding_during_plan_is_rejected(manifest, tmp_path, monkeypatch):
    m = make_manifest(manifest, tmp_path)
    real = planner.admit
    path = Path(m["candidates"][0]["task"]["path"])

    def changed(tasks):
        result = real(tasks)
        path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(planner, "admit", changed)
    with pytest.raises(ValueError):
        plan(m, tmp_path)


def test_mutated_training_file_after_batch_admission_is_rejected(manifest, tmp_path, monkeypatch):
    m = make_manifest(manifest, tmp_path)
    task = json.loads(Path(m["candidates"][0]["task"]["path"]).read_text())
    path = Path(task["data"]["training"]["path"])
    real = planner.admit

    def changed(tasks):
        result = real(tasks)
        path.write_text("[]")
        return result

    monkeypatch.setattr(planner, "admit", changed)
    with pytest.raises(ValueError):
        plan(m, tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_count": 0},
        {"target_count": True},
        {"target_count": -1},
        {"batch_size": 0},
        {"batch_size": 33},
        {"batch_size": True},
        {"projects": []},
        {"projects": ["project_0", "project_0"]},
        {"projects": ["unknown"]},
    ],
)
def test_invalid_requests_fail(manifest, tmp_path, kwargs):
    m = make_manifest(manifest, tmp_path)
    with pytest.raises(ValueError):
        plan(m, tmp_path, **kwargs)


def test_insufficient_capacity_fails_instead_of_repeating_scopes(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path)
    with pytest.raises(ValueError, match="Only 3 eligible"):
        plan(m, tmp_path, target_count=4)


def test_empty_history_is_not_silently_ignored(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path)
    previous = fixtures.save(tmp_path / "empty.json", {"tasks": []})
    with pytest.raises(ValueError, match="no task declarations"):
        plan(m, tmp_path, previous_panels=[previous])


def test_registered_snapshot_counts_must_match(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path)
    registered = fixtures.save(tmp_path / "bad_registry.json", {"indexed_entries": 2, "entries": []})
    with pytest.raises(ValueError, match="entry count mismatch"):
        plan(m, tmp_path, registered_catalogs=[registered])


def test_execution_scope_collision_rejected_even_with_distinct_catalog_scenarios(manifest, tmp_path):
    m = make_manifest(manifest, tmp_path)
    c = m["candidates"][1]
    task = json.loads(Path(c["task"]["path"]).read_text())
    c["protein_scope"]["identifier"] = task["scope"]["protein"] = m["candidates"][0]["protein_scope"]["identifier"]
    c["scenario"] = {"medium": "another catalog-only scenario"}
    c["task"] = fixtures.save(Path(c["task"]["path"]), task)
    assert build_catalog(m)["summary"]["statuses"] == {"admitted": 3}
    with pytest.raises(ValueError, match="duplicate execution scope"):
        plan(m, tmp_path)
