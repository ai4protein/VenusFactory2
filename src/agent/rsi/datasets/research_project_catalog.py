"""Admit researcher-defined protein/property/context tasks before model acquisition.

This catalog counts candidate scientific scopes, never trained or published models.
It wraps the existing source-bound model-task interface without changing old goals.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from .model_tasks import PHASES, bound_file, read_task
from agent.rsi.engine.protein_model import digest

SCHEMA = "research_project_catalog.v1"
MIN_ROWS = {"training": 40, "validation": 15, "future": 15}
MIN_GROUPS = {"training": 5, "validation": 3, "future": 3}
CONTEXT_MIN_ROWS = {"training": 40, "validation": 10, "future": 10}
CONTEXT_MIN_GROUPS = {"training": 4, "validation": 1, "future": 1}
GROUP_BASES = {
    "sequence_cluster",
    "parent_design",
    "replicate_group",
    "mutation_block",
    "study",
    "genetic_population",
    "deleted_gene",
    "publication",
    "protein_identity",
    "protein_publication_component",
    "antibody_patent_component",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    require(isinstance(value, str) and 0 < len(value.strip()) <= 4096, "Missing or oversized text")
    return " ".join(value.split()).casefold()


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", value), "Invalid identity")
    return value


def scope_key(candidate, task):
    """Architecture, seed, project alias, row order and task name do not add scopes."""
    require(candidate["protein_scope"]["kind"] in {"protein", "family", "project"}, "Unknown protein scope")
    protein = text(candidate["protein_scope"]["identifier"])
    require(protein == text(task["scope"]["protein"]), "Protein scope disagrees with the admitted task")
    require(text(candidate["property"]) == text(task["scope"]["target"]), "Property disagrees with task")
    scenario = candidate["scenario"]
    require(isinstance(scenario, dict) and 1 <= len(scenario) <= 20, "An observed scenario is required")
    normalized = {text(k): text(v) for k, v in scenario.items()}
    if task["plugin_id"] in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        require(
            normalized == {text(k): text(v) for k, v in task["scope"]["context"].items()},
            "Scenario differs from model context",
        )
    require(len(normalized) == len(scenario), "Repeated normalized scenario field")
    return digest(
        {
            "protein": protein,
            "property": text(candidate["property"]),
            "unit": text(task["scope"]["unit"]),
            "scenario": normalized,
        }
    )


def baseline_plan(task):
    if task["plugin_id"] in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        if task["plugin_id"] == "generated_antibody.v1":
            return {
                "implemented": [
                    "training_mean",
                    "antibody_ridge",
                    "antibody_random_forest",
                    "antibody_svr",
                    "paired_edit_nearest_neighbor",
                ],
                "domain_methods_pending": ["published_affinity_predictors_if_compatible"],
                "selection_data": "training_only",
                "comparison_data": "validation; final partition reserved",
                "domain_comparison_complete": False,
            }
        if task["plugin_id"] == "generated_assay.v1":
            plant_assays = task["scope"]["representation"]["name"] == "plant_assay_descriptors.v1"
            domain = ["enzyme_family_median", "substrate_condition_reference"]
            return {
                "implemented": ["training_mean", "assay_ridge", "assay_random_forest", "assay_svr"]
                + (domain if plant_assays else []),
                "domain_methods_pending": ["published_kinetics_predictors_if_compatible"] if plant_assays else domain,
                "selection_data": "training_only",
                "comparison_data": "validation; final partition reserved",
                "domain_comparison_complete": False,
            }
        from agent.rsi.engine.context_baselines import METHODS, thermal_methods

        return {
            "implemented": list(METHODS) + list(thermal_methods(task["scope"])),
            "domain_methods_pending": ["author_pipeline_reproduction_if_compatible"],
            "selection_data": "training_only",
            "comparison_data": "independent validation group; future reserved for a frozen final comparison",
            "domain_comparison_complete": False,
        }
    regression = "classifier" not in task["plugin_id"]
    domain = {
        "solubility": ["swi", "protein_sol"],
        "expression": ["length_composition_regression"],
        "stability": ["structure_energy_score_if_compatible", "additive_mutation_model_if_identifiable"],
        "activity": ["additive_mutation_model_if_identifiable", "nearest_homologue_if_available"],
        "kinetics": ["enzyme_family_median", "substrate_condition_regression"],
        "substrate_specificity": ["nearest_homologue_if_available", "enzyme_family_frequency"],
    }.get(task["stage"], ["domain_specific_reference_required"])
    return {
        "implemented": ["training_mean", "aac_ridge", "aac_random_forest", "aac_svr"] if regression else [],
        "domain_methods_pending": domain,
        "selection_data": "training_only",
        "comparison_data": "validation; future reserved for a frozen final comparison",
        "domain_comparison_complete": False,
    }


def inspect_candidate(candidate, project):
    required = {
        "id",
        "project_id",
        "task",
        "protein_scope",
        "property",
        "scenario",
        "question",
        "groups",
        "group_basis",
        "scenario_evidence",
    }
    require(isinstance(candidate, dict) and set(candidate) == required, "Invalid candidate fields")
    identifier(candidate["id"])
    text(candidate["question"])
    require(set(candidate["protein_scope"]) == {"kind", "identifier"}, "Invalid protein scope fields")
    task, data = read_task(bound_file(candidate["task"]))
    require(text(task["paper"]["doi"]) == text(project["study_doi"]), "Task belongs to another study")
    key = scope_key(candidate, task)
    bound_file(candidate["scenario_evidence"])
    require(candidate["group_basis"] in GROUP_BASES, "Unspecified independence grouping")
    groups = json.loads(bound_file(candidate["groups"]).read_text())
    require(isinstance(groups, dict), "Group assignments must be keyed by observation identity")
    contextual = task["plugin_id"] in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }
    if contextual:
        level = task["scope"]["grouping"]["level"]
        expected_basis = (
            level
            if task["plugin_id"] in {"generated_assay.v1", "generated_antibody.v1"}
            or level in {"genetic_population", "deleted_gene"}
            else "replicate_group"
        )
        require(candidate["group_basis"] == expected_basis, "Context grouping basis differs from source evidence")
    phase_groups, observations = {}, []
    for phase, rows in data.items():
        assigned = set()
        for row in rows:
            assignment = groups.get(row["id"])
            require(isinstance(assignment, dict) and set(assignment) == {"group", "phase"}, "Missing row grouping")
            require(assignment["phase"] == phase, "Group assignment disagrees with task partition")
            assigned.add(text(assignment["group"]))
            if contextual:
                require(assignment["group"] == row["group"], "Catalog grouping differs from source-bound context group")
                if task["plugin_id"] == "generated_antibody.v1":
                    observations.append(
                        [
                            row["heavy"],
                            row["light"],
                            row["protein"],
                            row["target_sequence_sha256"],
                            task["scope"]["context"],
                            row["values"],
                            row["target"],
                        ]
                    )
                elif task["plugin_id"] == "generated_assay.v1":
                    observations.append([row["sequence"], row["substrate"], row["study"], row["values"], row["target"]])
                else:
                    observations.append(
                        [task["paper"]["doi"], row["protein"], row["group"], row["values"], row["target"]]
                    )
            else:
                observations.append([row["sequence"], row["target"]])
        phase_groups[phase] = assigned
    for i, phase in enumerate(PHASES):
        for other in PHASES[i + 1 :]:
            require(not phase_groups[phase] & phase_groups[other], "Dependent group crosses partitions")
    minimum_rows = (
        CONTEXT_MIN_ROWS
        if contextual and task["plugin_id"] not in {"generated_assay.v1", "generated_antibody.v1"}
        else MIN_ROWS
    )
    minimum_groups = (
        CONTEXT_MIN_GROUPS
        if contextual and task["plugin_id"] not in {"generated_assay.v1", "generated_antibody.v1"}
        else MIN_GROUPS
    )
    reasons = [f"insufficient_{p}_rows" for p in PHASES if len(data[p]) < minimum_rows[p]]
    reasons += [f"insufficient_{p}_groups" for p in PHASES if len(phase_groups[p]) < minimum_groups[p]]
    if "classifier" in task["plugin_id"]:
        reasons.append("classification_baseline_suite_pending")
    elif len({row["target"] for row in data["training"]}) < 2:
        reasons.append("constant_training_endpoint")
    # Deduplicate identical measurements despite renamed projects or re-splitting.
    measurement_key = digest(
        {
            "property": text(task["scope"]["target"]),
            "unit": text(task["scope"]["unit"]),
            "measurements": sorted(observations),
        }
    )
    result = {
        "id": candidate["id"],
        "project_id": candidate["project_id"],
        "task": candidate["task"],
        "scope_key": key,
        "measurement_key": measurement_key,
        "protein_scope": candidate["protein_scope"],
        "property": candidate["property"],
        "scenario": candidate["scenario"],
        "stage": task["stage"],
        "status": "deferred" if reasons else "admitted",
        "reasons": reasons,
        "rows": {p: len(data[p]) for p in PHASES},
        "groups": {p: len(phase_groups[p]) for p in PHASES},
        "group_basis": candidate["group_basis"],
        "baseline_plan": baseline_plan(task),
        "homology_independence_established": False,
    }
    if contextual:
        result["independence_profile"] = (
            "antibody_patent_groups.v1"
            if task["plugin_id"] == "generated_antibody.v1"
            else "assay_observation_groups.v1"
            if task["plugin_id"] == "generated_assay.v1"
            else "context_response_groups.v1"
        )
    return result, data


def build_catalog(manifest):
    require(
        isinstance(manifest, dict) and set(manifest) == {"schema", "target_models", "projects", "candidates"},
        "Invalid research catalog manifest",
    )
    require(manifest["schema"] == SCHEMA, "Unsupported catalog schema")
    require(type(manifest["target_models"]) is int and manifest["target_models"] > 0, "Positive model target required")
    require(isinstance(manifest["projects"], list) and isinstance(manifest["candidates"], list), "Lists required")
    projects = {}
    for project in manifest["projects"]:
        require(
            isinstance(project, dict)
            and set(project) == {"id", "title", "url", "study_doi", "source_evidence", "license_evidence", "license"},
            "Invalid project fields",
        )
        key = identifier(project["id"])
        require(key not in projects, "Repeated project identity")
        for field in ("title", "url", "study_doi", "license"):
            text(project[field])
        require(project["url"].startswith("https://"), "Persistent source URL required")
        bound_file(project["source_evidence"])
        bound_file(project["license_evidence"])
        projects[key] = project
    identifiers = [identifier(c.get("id")) for c in manifest["candidates"]]
    require(len(identifiers) == len(set(identifiers)), "Repeated candidate identity")
    entries, scopes, measurements, sequence_phases = [], {}, {}, {}
    antibody_pairs = {phase: set() for phase in PHASES}
    for candidate in sorted(manifest["candidates"], key=lambda c: c["id"]):
        try:
            require(candidate.get("project_id") in projects, "Unknown research project")
            result, data = inspect_candidate(candidate, projects[candidate["project_id"]])
            previous = scopes.get(result["scope_key"]) or measurements.get(result["measurement_key"])
            if previous:
                result.update(status="duplicate", duplicate_of=previous, reasons=["already_represented_task"])
            elif result["status"] == "admitted":
                if result.get("independence_profile") == "context_response_groups.v1":
                    task, _ = read_task(bound_file(candidate["task"]))
                    keys = {
                        p: [("context", text(task["paper"]["doi"]), text(r["group"])) for r in data[p]] for p in PHASES
                    }
                elif result.get("independence_profile") == "antibody_patent_groups.v1":
                    from agent.rsi.engine.generated_antibody import pairs_overlap, partition_identities

                    task, _ = read_task(bound_file(candidate["task"]))
                    keys = {
                        phase: [
                            ("antibody", *identity)
                            for row in data[phase]
                            for identity in partition_identities(row, task["scope"])
                        ]
                        for phase in PHASES
                    }
                else:
                    keys = {p: [r["sequence"] for r in data[p]] for p in PHASES}
                conflicts = any(sequence_phases.get(key, p) != p for p in PHASES for key in keys[p])
                paired = result.get("independence_profile") == "antibody_patent_groups.v1"
                if paired:
                    current_pairs = {phase: {(row["heavy"], row["light"]) for row in data[phase]} for phase in PHASES}
                    conflicts = conflicts or any(
                        pairs_overlap(current_pairs[phase], antibody_pairs[other])
                        for phase in PHASES
                        for other in PHASES
                        if phase != other
                    )
                if conflicts:
                    result.update(
                        status="deferred",
                        reasons=[
                            "cross_task_group_partition_conflict"
                            if result.get("independence_profile")
                            else "cross_task_sequence_partition_conflict"
                        ],
                    )
                else:
                    scopes[result["scope_key"]] = result["id"]
                    measurements[result["measurement_key"]] = result["id"]
                    sequence_phases.update({key: p for p in PHASES for key in keys[p]})
                    if paired:
                        for phase in PHASES:
                            antibody_pairs[phase].update(current_pairs[phase])
        except (ValueError, TypeError, KeyError, OSError) as exc:
            result = {"id": candidate["id"], "status": "invalid", "reasons": [str(exc)]}
        entries.append(result)
    admitted = [e for e in entries if e["status"] == "admitted"]
    return {
        "schema": SCHEMA,
        "manifest_sha256": digest(manifest),
        "target_models": manifest["target_models"],
        "admission_policy": {
            "minimum_rows": MIN_ROWS,
            "minimum_groups": MIN_GROUPS,
            "context_response_groups.v1": {"minimum_rows": CONTEXT_MIN_ROWS, "minimum_groups": CONTEXT_MIN_GROUPS},
            "meaning": "feasibility floor, not a statistical power guarantee",
        },
        "entries": entries,
        "summary": {
            "candidate_tasks": len(entries),
            "statuses": dict(Counter(e["status"] for e in entries)),
            "projects": len({e["project_id"] for e in admitted}),
            "protein_scopes": len({text(e["protein_scope"]["identifier"]) for e in admitted}),
            "raw_endpoint_labels": len({text(e["property"]) for e in admitted}),
            "property_families": len({e["stage"] for e in admitted}),
            "stages": dict(Counter(e["stage"] for e in admitted)),
            "new_trained_models": 0,
            "new_published_models": 0,
        },
    }


def campaign_plan(manifest, catalog, profile, goal_budget, manifest_reference, generations=1):
    """Export existing campaign allocations; this does not execute or publish them."""
    from agent.harness import Profile

    from agent.rsi.engine.model_access import authorize_profile
    from agent.rsi.engine.model_coverage import SCIENTIFIC_SCOPES
    from agent.rsi.campaigns.scientist_model_campaign import RESOURCES, admit_allocations

    require(catalog == build_catalog(manifest), "Catalog or source evidence changed")
    require(json.loads(bound_file(manifest_reference).read_text()) == manifest, "Manifest binding differs")
    authorize_profile(Profile.from_dict(profile))
    allocations = []
    for entry in catalog["entries"]:
        if entry["status"] == "admitted":
            task, _ = read_task(bound_file(entry["task"]))
            task["references"].append(manifest_reference)
            allocations.append(
                {
                    "id": entry["id"],
                    "task": task,
                    "profile": profile,
                    "generations": generations,
                    "goal_budget": goal_budget,
                    "coverage": {"manifest": manifest_reference, "candidate_id": entry["id"]},
                }
            )
    admit_allocations(allocations, policy=SCIENTIFIC_SCOPES)
    return {
        "counting_policy": SCIENTIFIC_SCOPES,
        "allocations": allocations,
        "resources": {k: goal_budget[k] * len(allocations) for k in RESOURCES},
    }
