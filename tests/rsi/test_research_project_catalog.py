"""Synthetic admission checks; these fixtures are not acquired protein models."""

import copy
import json
from pathlib import Path

import pytest

from agent.rsi.datasets.research_project_catalog import SCHEMA, build_catalog, campaign_plan
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.rsi.engine.traditional_baselines import run_controls


def bound(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def save(path, value):
    atomic_json(path, value)
    return bound(path)


@pytest.fixture
def manifest(tmp_path):
    reference = save(tmp_path / "source.json", {"kind": "synthetic_fixture"})
    assets, groups = {}, {}
    aa = "ACDEFGHIKLMNPQRSTVWY"
    for phase, start, count in [("training", 0, 60), ("validation", 60, 20), ("future", 80, 20)]:
        rows = []
        for i in range(start, start + count):
            key = str(i)
            rows.append({"id": key, "sequence": aa[i // 20] + aa[i % 20] + "A" * 18, "target": float(i % 17)})
            groups[key] = {"group": key, "phase": phase}
        assets[phase] = save(tmp_path / f"{phase}.json", rows)
    task = {
        "schema": 1,
        "id": "synthetic_scope",
        "stage": "solubility",
        "plugin_id": "generated_sequence.v1",
        "scope": {
            "protein": "Synthetic test protein",
            "assay": "synthetic_assay",
            "target": "solubility",
            "unit": "%",
            "minimum_sequence_length": 20,
            "maximum_sequence_length": 2048,
        },
        "paper": {"title": "Synthetic fixture", "doi": "test:fixture"},
        "references": [reference],
        "data": assets,
        "minimum_skill": 0.0,
    }
    candidate = {
        "id": "candidate_one",
        "project_id": "project_one",
        "task": save(tmp_path / "task.json", task),
        "protein_scope": {"kind": "protein", "identifier": "Synthetic test protein"},
        "property": "solubility",
        "scenario": {"medium": "synthetic test context"},
        "question": "Does an independently admitted synthetic task reach the campaign interface?",
        "groups": save(tmp_path / "groups.json", groups),
        "group_basis": "replicate_group",
        "scenario_evidence": reference,
    }
    return {
        "schema": SCHEMA,
        "target_models": 10000,
        "projects": [
            {
                "id": "project_one",
                "title": "Synthetic project",
                "url": "https://example.org/fixture",
                "study_doi": "test:fixture",
                "source_evidence": reference,
                "license_evidence": reference,
                "license": "Synthetic test data",
            }
        ],
        "candidates": [candidate],
    }


def task_of(manifest):
    return json.loads(Path(manifest["candidates"][0]["task"]["path"]).read_text())


def update_task(manifest, task):
    candidate = manifest["candidates"][0]
    candidate["task"] = save(Path(candidate["task"]["path"]), task)


def test_admission_counts_scopes_not_models(manifest):
    report = build_catalog(manifest)
    assert report["summary"]["statuses"] == {"admitted": 1}
    assert report["target_models"] == 10000
    assert report["summary"]["new_published_models"] == 0
    assert report["summary"]["property_families"] == 1
    assert not report["entries"][0]["baseline_plan"]["domain_comparison_complete"]


@pytest.mark.parametrize("dimension", ["protein", "property", "scenario"])
def test_same_algorithm_can_count_different_research_scopes(manifest, tmp_path, dimension):
    candidate = copy.deepcopy(manifest["candidates"][0])
    task = task_of(manifest)
    candidate["id"] = "candidate_two"
    task["id"] = "synthetic_second"
    if dimension == "protein":
        task["scope"]["protein"] = candidate["protein_scope"]["identifier"] = "Another synthetic protein"
    elif dimension == "property":
        task["scope"]["target"] = candidate["property"] = "expression"
        task["stage"] = "expression"
    else:
        candidate["scenario"]["medium"] = "second observed synthetic context"
    for phase in task["data"]:
        rows = json.loads(Path(task["data"][phase]["path"]).read_text())
        for row in rows:
            if dimension == "protein":
                row["sequence"] = row["sequence"][:-1] + "C"
            row["target"] += 0.5
        task["data"][phase] = save(tmp_path / f"second_{phase}.json", rows)
    candidate["task"] = save(tmp_path / "second_task.json", task)
    manifest["candidates"].append(candidate)
    assert build_catalog(manifest)["summary"]["statuses"] == {"admitted": 2}


@pytest.mark.parametrize("attack", ["rename", "row_order", "scenario_alias"])
def test_duplicate_repackaging_does_not_inflate_count(manifest, tmp_path, attack):
    candidate = copy.deepcopy(manifest["candidates"][0])
    candidate["id"] = "candidate_two"
    task = task_of(manifest)
    task["id"] = "renamed_task"
    if attack == "row_order":
        for phase in task["data"]:
            rows = json.loads(Path(task["data"][phase]["path"]).read_text())[::-1]
            task["data"][phase] = save(tmp_path / f"reordered_{phase}.json", rows)
    if attack == "scenario_alias":
        candidate["scenario"]["alias"] = "new metadata without new observations"
    candidate["task"] = save(tmp_path / "renamed_task.json", task)
    manifest["candidates"].append(candidate)
    assert build_catalog(manifest)["summary"]["statuses"] == {"admitted": 1, "duplicate": 1}


@pytest.mark.parametrize(
    "attack",
    [
        "changed_source",
        "changed_task",
        "wrong_protein",
        "wrong_property",
        "group_overlap",
        "wrong_phase",
        "missing_group",
        "wrong_study",
        "unknown_project",
    ],
)
def test_rejects_broken_or_mismatched_evidence(manifest, attack):
    candidate = manifest["candidates"][0]
    if attack == "changed_source":
        Path(manifest["projects"][0]["source_evidence"]["path"]).write_text("changed")
        with pytest.raises(ValueError):
            build_catalog(manifest)
        return
    if attack == "changed_task":
        Path(candidate["task"]["path"]).write_text("{}")
    elif attack == "wrong_protein":
        candidate["protein_scope"]["identifier"] = "unrelated"
    elif attack == "wrong_property":
        candidate["property"] = "unrelated"
    elif attack == "wrong_study":
        manifest["projects"][0]["study_doi"] = "test:other"
    elif attack == "unknown_project":
        candidate["project_id"] = "unknown"
    else:
        path = Path(candidate["groups"]["path"])
        groups = json.loads(path.read_text())
        if attack == "group_overlap":
            groups["60"]["group"] = groups["0"]["group"]
        elif attack == "wrong_phase":
            groups["60"]["phase"] = "training"
        else:
            del groups["60"]
        candidate["groups"] = save(path, groups)
    assert build_catalog(manifest)["summary"]["statuses"] == {"invalid": 1}


def test_low_data_is_deferred_not_counted_as_model(manifest):
    task = task_of(manifest)
    path = Path(task["data"]["training"]["path"])
    task["data"]["training"] = save(path, json.loads(path.read_text())[:10])
    update_task(manifest, task)
    report = build_catalog(manifest)
    assert report["entries"][0]["reasons"] == ["insufficient_training_rows"]
    assert report["summary"]["statuses"] == {"deferred": 1}


def test_controls_ignore_future_labels_and_do_not_train_on_validation(manifest):
    task = task_of(manifest)
    first = run_controls(task)
    for phase in ["validation", "future"]:
        path = Path(task["data"][phase]["path"])
        rows = json.loads(path.read_text())
        for row in rows:
            row["target"] += 1000
        task["data"][phase] = save(path, rows)
    second = run_controls(task)
    assert not first["future_scored"]
    for a, b in zip(first["outcomes"], second["outcomes"], strict=True):
        assert a["predictions"] == b["predictions"]
        assert a["metrics"]["mae"] != b["metrics"]["mae"]


def test_export_binds_manifest_and_uses_existing_campaign_admission(manifest, tmp_path):
    from agent.harness import Budget, ModelBinding, Profile

    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("dmxapi", "glm-4-flash", "https://www.dmxapi.cn/v1"),
        Budget(wall_seconds=100, model_calls=10, tool_calls=10, output_tokens=4096),
    ).public()
    budget = {"max_cycles": 4, "model_calls": 40, "tool_calls": 40, "wall_seconds": 400}
    reference = save(tmp_path / "manifest.json", manifest)
    report = build_catalog(manifest)
    plan = campaign_plan(manifest, report, profile, budget, reference)
    assert len(plan["allocations"]) == 1
    assert reference in plan["allocations"][0]["task"]["references"]
    assert plan["resources"]["model_calls"] == 40
    report["entries"][0]["status"] = "duplicate"
    with pytest.raises(ValueError, match="Catalog or source"):
        campaign_plan(manifest, report, profile, budget, reference)
