"""Source-bound, outcome-blind allocation of catalog tasks into shared batches."""

from __future__ import annotations

import copy
import json
from collections import Counter, deque

from agent.harness.contracts import digest
from agent.rsi.datasets.model_tasks import bound_file, read_task
from agent.rsi.datasets.research_project_catalog import build_catalog, require
from agent.rsi.engine.model_scope_inventory import canonical_scope
from agent.rsi.engine.shared_model_bundle import admit

SCHEMA = "shared_acquisition_plan.v1"
SUPPORTED = {
    "generated_context.v1",
    "generated_context.v2",
    "generated_context.v3",
    "generated_sequence.v1",
    "generated_assay.v1",
    "generated_antibody.v1",
}


def task_scopes(value):
    """Read task declarations only, including unsuccessful or queued allocations."""
    found = []
    if isinstance(value, dict):
        if {"id", "scope", "plugin_id", "data", "paper"} <= set(value):
            require(isinstance(value["scope"], dict), "Invalid prior task scope")
            found.append(value)
        else:
            for child in value.values():
                found.extend(task_scopes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(task_scopes(child))
    return found


def plan_acquisition(
    manifest_reference,
    catalog_reference,
    *,
    projects,
    target_count,
    batch_size=32,
    previous_panels=(),
    registered_catalogs=(),
    coverage_policy=None,
):
    """Revalidate admission, exclude named history, and allocate without fitting.

    Registry metadata exclusions deliberately do not certify registered models.
    The result is an allocation plan, not a runnable or successful experiment.
    """
    require(type(target_count) is int and target_count > 0, "Positive target count required")
    require(type(batch_size) is int and 1 <= batch_size <= 32, "Batch size must be 1..32")
    require(
        isinstance(projects, list) and projects and len(projects) == len(set(projects)),
        "Declare distinct project identities",
    )
    manifest = json.loads(bound_file(manifest_reference).read_text())
    catalog = json.loads(bound_file(catalog_reference).read_text())
    require(catalog == build_catalog(manifest), "Catalog or source admission changed")
    project_ids = {p["id"] for p in manifest["projects"]}
    require(set(projects) <= project_ids, "Unknown project identity")
    excluded, panel_bindings, registry_bindings = set(), [], []
    for reference in previous_panels:
        value = json.loads(bound_file(reference).read_text())
        tasks = task_scopes(value)
        require(bool(tasks), "Prior panel has no task declarations")
        scopes = {canonical_scope(t) for t in tasks}
        excluded.update(scopes)
        panel_bindings.append({"reference": reference, "declarations": len(tasks), "scopes": len(scopes)})
    for reference in registered_catalogs:
        value = json.loads(bound_file(reference).read_text())
        require(isinstance(value.get("entries"), list), "Invalid registered catalog snapshot")
        require(value.get("indexed_entries") == len(value["entries"]), "Registered entry count mismatch")
        scopes = {canonical_scope(row["key"]) for row in value["entries"]}
        excluded.update(scopes)
        registry_bindings.append({"reference": reference, "entries": len(value["entries"]), "scopes": len(scopes)})
    policy = None
    if coverage_policy is not None:
        from .property_coverage import select_by_property, validate_policy

        policy = json.loads(bound_file(coverage_policy).read_text())
        require(validate_policy(policy) == target_count, "Property quotas must equal target count")
    groups, bindings, coverage_records = {}, {}, []
    skipped = Counter()
    unique_scopes, unique_ids = set(), set()
    for entry in sorted(catalog["entries"], key=lambda e: e["id"]):
        if entry["status"] != "admitted" or entry["project_id"] not in projects:
            continue
        task, _ = read_task(bound_file(entry["task"]))
        scope = canonical_scope(task)
        if scope in excluded:
            skipped["previously_registered_or_allocated"] += 1
            continue
        if task["plugin_id"] not in SUPPORTED:
            skipped["unsupported_shared_interface"] += 1
            continue
        require(scope not in unique_scopes, "Catalog contains duplicate execution scope")
        require(task["id"] not in unique_ids, "Catalog contains repeated task identity")
        unique_scopes.add(scope)
        unique_ids.add(task["id"])
        task = copy.deepcopy(task)
        if task["plugin_id"] == "generated_context.v1":
            task["plugin_id"] = "generated_context.v2"
            task["scope"]["interface_version"] = "generated_context.v2"
        require(canonical_scope(task) == scope, "Interface conversion changed scientific scope")
        group = (entry["project_id"], task["paper"]["doi"], task["plugin_id"], task["stage"])
        groups.setdefault(group, deque()).append(task)
        coverage_records.append(
            {
                "id": task["id"],
                "stage": task["stage"],
                "target": task["scope"]["target"],
                "protein": task["scope"]["protein"],
                "study_doi": task["paper"]["doi"],
                "scenario": entry["scenario"],
                "interface": task["plugin_id"],
            }
        )
        bindings[task["id"]] = {
            "candidate_id": entry["id"],
            "task": entry["task"],
            "scope_key": scope,
            "catalog_scope_key": entry["scope_key"],
            "project_id": entry["project_id"],
            "rows": entry["rows"],
            "measurement_key": entry["measurement_key"],
        }
    available = sum(map(len, groups.values()))
    group_capacity = {key: len(queue) for key, queue in groups.items()}
    coverage = select_by_property(coverage_records, policy) if policy is not None else None
    selection_target = target_count
    if coverage is None:
        require(available >= target_count, f"Only {available} eligible unallocated scopes; requested {target_count}")
    else:
        selected_ids = set(coverage["selected_ids"])
        selection_target = len(selected_ids)
        for key, queue in groups.items():
            groups[key] = deque(t for t in queue if t["id"] in selected_ids)
        for name, family in coverage["assignments"].items():
            bindings[name]["property_family"] = family
    capacity = [
        {
            "project_id": key[0],
            "study_doi": key[1],
            "plugin_id": key[2],
            "stage": key[3],
            "available_scopes": group_capacity[key],
        }
        for key, values in sorted(groups.items())
    ]
    batches, chosen, counts = [], [], Counter()
    # Rotate whole homogeneous batches, retaining small projects before filling
    # the remaining allocation from larger cohorts. No model outcomes are read.
    while len(chosen) < selection_target:
        for group, queue in sorted(groups.items()):
            size = min(batch_size, len(queue), selection_target - len(chosen))
            if not size:
                continue
            tasks = [queue.popleft() for _ in range(size)]
            admit(tasks)
            batches.append({"id": f"batch_{len(batches):04d}", "tasks": tasks})
            chosen.extend(tasks)
            counts[group[0]] += size
    selected = [bindings[t["id"]] for t in chosen]
    require(len({v["scope_key"] for v in selected}) == selection_target, "Repeated selected scientific scope")
    require(not {v["scope_key"] for v in selected} & excluded, "Prior scope reallocated")
    # Recheck explicitly bound inputs after the potentially lengthy admission.
    for reference in [
        manifest_reference,
        catalog_reference,
        *previous_panels,
        *registered_catalogs,
        *([coverage_policy] if coverage_policy is not None else []),
        *(row["task"] for row in selected),
    ]:
        bound_file(reference)
    for row in selected:
        read_task(bound_file(row["task"]))
    result = {
        "schema": SCHEMA,
        "status": "prepared_allocation_only",
        "manifest": manifest_reference,
        "catalog": catalog_reference,
        "projects": sorted(projects),
        "target_count": target_count,
        "batch_size": batch_size,
        "batch_count": len(batches),
        "available_scopes": available,
        "remaining_unallocated_scopes": available - len(chosen),
        "capacity": capacity,
        "selection_counts": dict(sorted(counts.items())),
        "excluded_scope_count": len(excluded),
        "exclusions": {"panels": panel_bindings, "registered_catalogs": registry_bindings},
        "skipped": dict(skipped),
        "source_bindings": selected,
        "batches": batches,
        "selection_rule": "Project/study/interface/stage batch round-robin, ascending candidate identity within group; no prediction outcomes used.",
        "exclusion_scope": "Only the explicitly bound registered snapshots and allocation panels; not an exhaustive repository-history claim.",
        "catalog_digest": digest(catalog),
        "new_autonomous_models": 0,
        "external_model_calls": 0,
        "final_labels_scored": False,
        "completion_claim": False,
    }

    if coverage is not None:
        result.update(
            schema="shared_acquisition_plan.v2",
            coverage_policy=coverage_policy,
            coverage=coverage,
            selected_count=len(chosen),
            status="prepared_allocation_only" if coverage["coverage_satisfied"] else "prepared_partial_allocation",
            selection_rule=coverage["selection_basis"],
        )
    return result
