"""Small execution manifests projected from a hash-bound admitted source catalog."""

import copy
import json
from pathlib import Path

from .model_tasks import bound_file
from agent.rsi.engine.protein_model import digest
from .research_project_catalog import SCHEMA, build_catalog
from agent.rsi.engine.storage import atomic_json, sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def project_batch(manifest_reference, catalog_reference, candidate_ids):
    """Keep source task identities and global admission decisions; recheck selected assets."""
    require(isinstance(candidate_ids, list) and 0 < len(candidate_ids) <= 128, "Select 1..128 candidate IDs")
    require(all(isinstance(x, str) for x in candidate_ids), "Candidate IDs must be strings")
    require(len(candidate_ids) == len(set(candidate_ids)), "Repeated batch candidate")
    manifest = json.loads(bound_file(manifest_reference).read_text())
    catalog = json.loads(bound_file(catalog_reference).read_text())
    require(manifest.get("schema") == catalog.get("schema") == SCHEMA, "Unsupported parent catalog")
    require(catalog.get("manifest_sha256") == digest(manifest), "Parent catalog does not bind its manifest")
    candidates = {x["id"]: x for x in manifest["candidates"]}
    entries = {x["id"]: x for x in catalog["entries"]}
    require(
        len(candidates) == len(manifest["candidates"]) and len(entries) == len(catalog["entries"]),
        "Repeated parent IDs",
    )
    require(set(candidates) == set(entries), "Incomplete parent catalog membership")
    for identity in candidate_ids:
        require(
            identity in entries and entries[identity]["status"] == "admitted", "Candidate is not admitted in parent"
        )
        require(candidates[identity]["task"] == entries[identity]["task"], "Parent task binding differs")
    projects = {candidates[x]["project_id"] for x in candidate_ids}
    child = {
        "schema": SCHEMA,
        "target_models": manifest["target_models"],
        "projects": [copy.deepcopy(x) for x in manifest["projects"] if x["id"] in projects],
        "candidates": [copy.deepcopy(candidates[x]) for x in candidate_ids],
    }
    checked = build_catalog(child)
    require(checked["entries"] == [entries[x] for x in sorted(candidate_ids)], "Selected admission changed on replay")
    # Re-read bindings after data admission; callers use an immutable parent catalog receipt.
    bound_file(manifest_reference)
    bound_file(catalog_reference)
    provenance = {
        "schema": "research_batch.v1",
        "parent_manifest": copy.deepcopy(manifest_reference),
        "parent_catalog": copy.deepcopy(catalog_reference),
        "selected_candidates": list(candidate_ids),
        "parent_candidates": len(candidates),
        "batch_candidates": len(candidate_ids),
        "admission": "Hash-bound prior global catalog, with fresh selected-task admission and unchanged scope identities",
        "new_trained_models": 0,
        "new_published_models": 0,
        "external_model_calls": 0,
    }
    return child, checked, provenance


def export_batch(manifest_reference, catalog_reference, candidate_ids, output):
    output = Path(output).absolute()
    require(not any(x.is_symlink() for x in [output, *output.parents]), "Symlinked batch output")
    require(not output.exists(), "Batch output already exists")
    manifest, catalog, provenance = project_batch(manifest_reference, catalog_reference, candidate_ids)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "catalog.json", catalog)
    provenance["outputs"] = {
        name: {"path": str(output / name), "sha256": sha256_file(output / name)}
        for name in ["manifest.json", "catalog.json"]
    }
    atomic_json(output / "provenance.json", provenance)
    return provenance
