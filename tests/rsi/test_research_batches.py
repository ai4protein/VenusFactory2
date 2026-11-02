"""Batching cannot relabel deferred/duplicate tasks or bypass current asset checks."""

import copy
import json
from pathlib import Path

import pytest
from test_research_project_catalog import manifest as manifest
from test_research_project_catalog import save, task_of

from agent.rsi.model_coverage import resolve_scope, summarize_scopes
from agent.rsi.research_batches import export_batch, project_batch
from agent.rsi.research_project_catalog import build_catalog


def parent(manifest, tmp_path):
    return save(tmp_path / "parent_manifest.json", manifest), save(
        tmp_path / "parent_catalog.json", build_catalog(manifest)
    )


def test_projection_preserves_identity_and_does_not_duplicate_models(manifest, tmp_path):
    mr, cr = parent(manifest, tmp_path)
    before = copy.deepcopy(manifest)
    provenance = export_batch(mr, cr, ["candidate_one"], tmp_path / "batch")
    child = json.loads((tmp_path / "batch/manifest.json").read_text())
    assert child == before == manifest
    assert provenance["new_published_models"] == 0
    task = task_of(manifest)
    records = []
    for reference in [mr, provenance["outputs"]["manifest.json"]]:
        allocation = {
            "task": {**task, "references": task["references"] + [reference]},
            "coverage": {"manifest": reference, "candidate_id": "candidate_one"},
        }
        result = resolve_scope(allocation)
        records.append({"entry_id": str(len(records)), **result})
    assert summarize_scopes(records)["primary_count"] == 1


@pytest.mark.parametrize("ids", [[], ["missing"], ["candidate_one", "candidate_one"], [False]])
def test_invalid_selection_has_no_output(manifest, tmp_path, ids):
    mr, cr = parent(manifest, tmp_path)
    with pytest.raises(ValueError):
        export_batch(mr, cr, ids, tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


@pytest.mark.parametrize("attack", ["parent", "catalog", "training", "status", "descriptor"])
def test_changed_parent_or_scientific_evidence_fails(manifest, tmp_path, attack):
    mr, cr = parent(manifest, tmp_path)
    if attack in {"parent", "catalog"}:
        Path((mr if attack == "parent" else cr)["path"]).write_text("{}")
    elif attack == "training":
        Path(task_of(manifest)["data"]["training"]["path"]).write_text("[]")
    else:
        catalog = json.loads(Path(cr["path"]).read_text())
        if attack == "status":
            catalog["entries"][0]["status"] = "deferred"
        else:
            catalog["entries"][0]["scope_key"] = "0" * 64
        cr = save(Path(cr["path"]), catalog)
    with pytest.raises(ValueError):
        project_batch(mr, cr, ["candidate_one"])


def test_duplicate_in_parent_cannot_be_readmitted_by_omitting_its_peer(manifest, tmp_path):
    duplicate = copy.deepcopy(manifest["candidates"][0])
    duplicate["id"] = "candidate_two"
    manifest["candidates"].append(duplicate)
    mr, cr = parent(manifest, tmp_path)
    assert build_catalog(manifest)["entries"][1]["status"] == "duplicate"
    with pytest.raises(ValueError, match="not admitted"):
        project_batch(mr, cr, ["candidate_two"])


def test_export_never_overwrites_existing_directory(manifest, tmp_path):
    mr, cr = parent(manifest, tmp_path)
    output = tmp_path / "batch"
    export_batch(mr, cr, ["candidate_one"], output)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="exists"):
        export_batch(mr, cr, ["candidate_one"], output)
    assert (output / "manifest.json").read_bytes() == before
