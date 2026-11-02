"""Protein-specific tasks from unused observations of previously curated designs.

Only source-declared G0 designed scaffolds in the prior training-parent partition
are considered. This adapter reads no raw archive and generates no new sequences.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from .model_tasks import PHASES, bound_file, load_task, read_task
from .published_sequence_tasks import DESIGN_CLUSTERS, PAPERS, bound, partition_rows
from .research_project_catalog import SCHEMA, build_catalog

VERSION = "designed_protein_tasks.v1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def mutation_site(sequence, parent):
    """Accept one actual residue substitution, independently of row-name syntax."""
    if not isinstance(sequence, str) or not isinstance(parent, str):
        raise ValueError("Sequences must be strings")
    if not 20 <= len(sequence) <= 2048 or set(sequence) - AA or set(parent) - AA:
        raise ValueError("Noncanonical or unsupported sequence")
    if len(sequence) != len(parent):
        return None
    changed = [i + 1 for i, (a, b) in enumerate(zip(sequence, parent, strict=True)) if a != b]
    return changed[0] if len(changed) == 1 else None


def protected_observations(catalog):
    """Protect every phase of previous tasks, including their training samples."""
    ids, sequences, bindings = set(), set(), []
    for entry in catalog["entries"]:
        task, data = read_task(bound_file(entry["task"]))
        bindings.append(entry["task"])
        for rows in data.values():
            ids.update(r["id"] for r in rows)
            sequences.update(r["sequence"] for r in rows)
    return ids, sequences, bindings


def select_records(records, curation, assignments, protected_ids, protected_sequences):
    parents = curation["parent_scaffolds"]
    groups, exclusions, seen_ids, seen_sequences = collections.defaultdict(list), collections.Counter(), set(), set()
    parent_phases = {}
    for row in records:
        if row["id"] in seen_ids or row["sequence"] in seen_sequences:
            raise ValueError("Source must already be deduplicated by observation and sequence")
        seen_ids.add(row["id"])
        seen_sequences.add(row["sequence"])
        assignment = assignments.get(row["id"])
        if not isinstance(assignment, dict) or set(assignment) != {"group", "phase"}:
            raise ValueError("Missing original parent assignment")
        if assignment["group"] != row["group"] or assignment["phase"] not in PHASES:
            raise ValueError("Original parent assignment disagrees with source")
        previous = parent_phases.setdefault(row["group"], assignment["phase"])
        if previous != assignment["phase"]:
            raise ValueError("Original parent crosses partitions")
        if assignment["phase"] != "training":
            exclusions["original_validation_or_future_parent"] += 1
            continue
        parent = parents.get(row["group"])
        if not parent or parent.get("DMS_group") != "G0" or parent.get("WT_cluster") not in DESIGN_CLUSTERS:
            exclusions["outside_curated_designed_scope"] += 1
            continue
        if row["topology"] != parent["WT_cluster"]:
            raise ValueError("Topology disagrees with original design metadata")
        if row["id"] in protected_ids or row["sequence"] in protected_sequences:
            exclusions["previously_used_observation"] += 1
            continue
        site = mutation_site(row["sequence"], parent["aa_seq"])
        if site is None:
            exclusions["not_one_substitution"] += 1
            continue
        groups[row["group"]].append({**row, "mutation_site": site})
    return dict(groups), dict(exclusions)


def site_assignments(records):
    """Place all substitutions at one site together, without consulting labels."""
    result = {}
    for row in records:
        group = f"{row['group']}|site={row['mutation_site']}"
        bucket = int(hashlib.sha256(f"{VERSION}:{group}".encode()).hexdigest(), 16) % 100
        phase = "training" if bucket < 60 else "validation" if bucket < 80 else "future"
        result[row["id"]] = {"group": group, "phase": phase}
    return result


def prepare(source_manifest, output):
    expected = {"schema", "observations", "curation", "assignments", "prior_catalog", "licenses"}
    if (
        not isinstance(source_manifest, dict)
        or set(source_manifest) != expected
        or source_manifest["schema"] != VERSION
    ):
        raise ValueError("Invalid source manifest")
    sources = {k: json.loads(bound_file(source_manifest[k]).read_text()) for k in expected - {"schema"}}
    protected_ids, protected_sequences, task_bindings = protected_observations(sources["prior_catalog"])
    selected, excluded = select_records(
        sources["observations"], sources["curation"], sources["assignments"], protected_ids, protected_sequences
    )
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "source_manifest.json", source_manifest)
    protocol = {
        "schema": VERSION,
        "source_manifest": bound(output / "source_manifest.json"),
        "prior_task_bindings": task_bindings,
        "parent_pool": "Only original training-parent groups; all original validation/future parents excluded",
        "observation_pool": "Exclude IDs and exact sequences from every phase of every previous task",
        "variant_scope": "Previously measured single amino-acid substitutions in author-declared G0 designs",
        "split": "SHA256(version:parent|site=position) modulo 100: train<60, validation<80, future otherwise",
        "generalization": "Unseen mutation sites within a previously studied designed protein",
        "cross_protein_generalization_test": False,
        "limits": {"training": 600, "validation": 250, "future": 250},
        "minimum_skill": 0.0,
        "new_autonomous_models": 0,
        "implementation": bound(Path(__file__).absolute()),
    }
    write_new(output / "protocol.json", protocol)
    paper = PAPERS["tsuboyama2023"]
    project = {
        "id": "designed_protein_stability",
        "title": paper["title"],
        "url": "https://doi.org/" + paper["doi"],
        "study_doi": paper["doi"],
        "source_evidence": bound(output / "protocol.json"),
        "license_evidence": source_manifest["licenses"],
        "license": "CC BY 4.0",
    }
    manifest = {"schema": SCHEMA, "target_models": 10000, "projects": [project], "candidates": []}
    prepared, rejected = [], []
    for parent, records in sorted(selected.items()):
        identity = "designed_stability_" + hashlib.sha256(parent.encode()).hexdigest()[:20]
        folder = output / identity
        folder.mkdir()
        assignment = site_assignments(records)
        partitions, eligible = partition_rows(records, "dG_ML", assignment)
        used_ids = {row["id"] for rows in partitions.values() for row in rows}
        assignment = {key: value for key, value in assignment.items() if key in used_ids}
        write_new(folder / "assignments.json", assignment)
        write_new(
            folder / "source_records.json",
            {
                "parent": sources["curation"]["parent_scaffolds"][parent],
                "records": [r for r in records if r["id"] in used_ids],
            },
        )
        assets = {}
        for phase, rows in partitions.items():
            write_new(folder / (phase + ".json"), rows)
            assets[phase] = bound(folder / (phase + ".json"))
        task = {
            "schema": 1,
            "id": identity,
            "stage": "stability",
            "plugin_id": "generated_sequence.v1",
            "scope": {
                "protein": "Author-designed scaffold " + parent,
                "assay": "tsuboyama2023_cdna_display_proteolysis",
                "target": "dG_ML",
                "unit": "kcal/mol",
                "minimum_sequence_length": 20,
                "maximum_sequence_length": 2048,
            },
            "paper": paper,
            "references": [
                bound(output / "protocol.json"),
                bound(output / "source_manifest.json"),
                source_manifest["licenses"],
                bound(folder / "assignments.json"),
                bound(folder / "source_records.json"),
            ],
            "data": assets,
            "minimum_skill": 0.0,
        }
        write_new(folder / "task.json", task)
        candidate = {
            "id": identity,
            "project_id": project["id"],
            "task": bound(folder / "task.json"),
            "protein_scope": {"kind": "protein", "identifier": task["scope"]["protein"]},
            "property": "dG_ML",
            "scenario": {"assay": "cDNA display proteolysis", "context": "author assay conditions"},
            "question": "Predict measured folding stability for substitutions at unseen sites of this designed protein",
            "groups": bound(folder / "assignments.json"),
            "group_basis": "mutation_block",
            "scenario_evidence": bound(output / "protocol.json"),
        }
        manifest["candidates"].append(candidate)
        try:
            load_task(task)
            prepared.append(
                {
                    "task": candidate["task"],
                    "id": identity,
                    "stage": "stability",
                    "paper": paper["doi"],
                    "parent": parent,
                    "eligible_rows": eligible,
                }
            )
        except ValueError as exc:
            rejected.append({"id": identity, "reason": str(exc)})
    write_new(output / "project_manifest.json", manifest)
    catalog = build_catalog(manifest)
    write_new(output / "project_catalog.json", catalog)
    write_new(output / "model_task_catalog.json", {"schema": 1, "entries": prepared})
    report = {
        "schema": VERSION,
        "eligible_training_parents": len(selected),
        "excluded_observations": excluded,
        "task_interface_rejections": rejected,
        "catalog": catalog["summary"],
        "protected_prior_observation_ids": len(protected_ids),
        "protected_prior_sequences": len(protected_sequences),
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "sequence_generation_performed": False,
    }
    write_new(output / "report.json", report)
    return report
