"""Evidence-bound coverage counts, independent of model programs and versions."""

from __future__ import annotations

import json

from agent.rsi.datasets.model_tasks import bound_file, read_task
from .protein_model import digest
from agent.rsi.datasets.research_project_catalog import SCHEMA, build_catalog, identifier, text

SCIENTIFIC_SCOPES = "scientific_scope.v1"
PROGRAM_ENTRIES = "program_entry.v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def counting_policy(definition):
    # Missing policy is the historical program-entry protocol, never a migration.
    policy = definition.get("counting_policy", PROGRAM_ENTRIES)
    require(policy in {SCIENTIFIC_SCOPES, PROGRAM_ENTRIES}, "Unknown model counting policy")
    return policy


def resolve_scope(allocation, *, cache=None):
    """Recompute a scientific scope from source-bound project admission evidence.

    The cache is local to one admission/audit call. It only avoids repeatedly
    rebuilding the same catalog; source/data hashes and cross-task partitions
    are rechecked once per manifest in each call.
    """
    binding = allocation.get("coverage")
    require(
        isinstance(binding, dict) and set(binding) == {"manifest", "candidate_id"},
        "Scientific-scope counting requires a bound project manifest and candidate_id",
    )
    candidate_id = identifier(binding["candidate_id"])
    cache = {} if cache is None else cache
    manifest_key = digest(binding["manifest"])
    if manifest_key not in cache:
        manifest = json.loads(bound_file(binding["manifest"]).read_text())
        require(
            isinstance(manifest, dict)
            and set(manifest) == {"schema", "target_models", "projects", "candidates"}
            and manifest["schema"] == SCHEMA,
            "Unsupported project manifest",
        )
        catalog = build_catalog(manifest)
        cache[manifest_key] = (
            {row["id"]: row for row in manifest["candidates"]},
            {row["id"]: row for row in catalog["entries"]},
        )
    candidates, descriptors = cache[manifest_key]
    require(candidate_id in candidates, "Coverage candidate is absent from manifest")
    candidate = candidates[candidate_id]
    descriptor = descriptors[candidate_id]
    require(descriptor["status"] == "admitted", "Coverage candidate does not meet catalog admission")
    source_task, _ = read_task(bound_file(candidate["task"]))
    task = allocation["task"]
    # An exported task only appends provenance; scope, data and thresholds must
    # still be exactly those whose candidate admission is being claimed.
    require(
        {k: v for k, v in task.items() if k != "references"}
        == {k: v for k, v in source_task.items() if k != "references"},
        "Coverage task differs from source-bound candidate",
    )
    require(
        task["references"] == source_task["references"] + [binding["manifest"]],
        "Coverage manifest must be bound into the executed task",
    )
    return {
        "scope_key": digest(
            {
                "policy": SCIENTIFIC_SCOPES,
                "protein": text(descriptor["protein_scope"]["identifier"]),
                "property": text(descriptor["property"]),
                "scenario": {text(k): text(v) for k, v in descriptor["scenario"].items()},
            }
        ),
        "catalog_scope_key": descriptor["scope_key"],
        "measurement_key": descriptor["measurement_key"],
        "protein_scope": descriptor["protein_scope"],
        "property": descriptor["property"],
        "scenario": descriptor["scenario"],
        "stage": descriptor["stage"],
        "manifest": binding["manifest"],
        "candidate_id": candidate_id,
    }


def summarize_scopes(records):
    """Collapse programs sharing a scope OR the same measurements transitively.

    This is evidence-linked identity, not a protein ontology or proof that
    differently named biological scopes are independent.
    """
    entries, parents, anchors = {}, {}, {}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for record in sorted(records, key=lambda r: r["entry_id"]):
        entry = record["entry_id"]
        require(entry not in entries, "Repeated verified entry in coverage inventory")
        entries[entry] = record
        parents[entry] = entry
        for kind in ("scope_key", "measurement_key"):
            value = record[kind]
            require(
                isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                "Invalid coverage identity",
            )
            key = (kind, value)
            if key in anchors:
                parents[find(entry)] = find(anchors[key])
            else:
                anchors[key] = entry
    groups = {}
    for entry, record in entries.items():
        group = groups.setdefault(find(entry), {"entry_ids": [], "scope_keys": set(), "measurement_keys": set()})
        group["entry_ids"].append(entry)
        group["scope_keys"].add(record["scope_key"])
        group["measurement_keys"].add(record["measurement_key"])
    rows = [{k: sorted(v) for k, v in group.items()} for group in groups.values()]
    rows.sort(key=lambda row: row["entry_ids"])
    return {
        "counting_policy": SCIENTIFIC_SCOPES,
        "primary_count": len(rows),
        "verified_program_entries": len(entries),
        "collapsed_additional_entries": len(entries) - len(rows),
        "scope_groups": rows,
        "scope_evidence": [entries[entry] for entry in sorted(entries)],
    }
