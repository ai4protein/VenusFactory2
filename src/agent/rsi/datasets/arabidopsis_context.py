"""Source-specific export of measured Arabidopsis thermal-response tasks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from .model_tasks import ordinary
from .research_project_catalog import SCHEMA, build_catalog
from agent.rsi.engine.storage import atomic_json, sha256_file

VERSION = "arabidopsis_relative_thermal_response.v1"
DOI = "10.1074/mcp.RA118.001124"
COMMIT = "da6e2257451d7cf91e33a6d01f8850c5126bac56"
REPLICATES = ("C1", "C2", "C3", "C4", "C5", "C6")
PARTITIONS = {"training": REPLICATES[:4], "validation": ("C5",), "future": ("C6",)}
BSA = "sp_ALBU_BOVIN_"
PROJECT = "arabidopsis_thermal_2019"


def bound(path):
    path = ordinary(Path(path).absolute())
    return {"path": str(path), "sha256": sha256_file(path)}


def save(path, value):
    atomic_json(path, value)
    return bound(path)


def read_mapping(path):
    with ordinary(Path(path).absolute()).open() as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    channels = [r["channel"].replace("TMT10.", "TMT10-") for r in records]
    temperatures = [float(r["temp"]) for r in records]
    if (
        len(channels) != 10
        or len(set(channels)) != 10
        or len(set(temperatures)) != 10
        or any(not math.isfinite(t) or not 15 <= t <= 75 for t in temperatures)
    ):
        raise ValueError("Ten unique temperature/channel assignments required")
    return channels, temperatures


def aggregate_psms(path, channels):
    """Single plant loci and BSA only; PSMs are technical measurements, not samples."""
    proteins, seen, counts = {}, set(), Counter()
    with ordinary(Path(path).absolute()).open() as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        required = {"id", "protein", "peptide", "coelute_inf", *channels}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("PSM table lacks required fields")
        for row in reader:
            counts["source_psms"] += 1
            if row["id"] in seen:
                raise ValueError("Repeated PSM identity")
            seen.add(row["id"])
            protein = row["protein"]
            if protein != BSA and not re.fullmatch(r"AT[1-5CM]G[0-9]{5}", protein):
                counts["excluded_ambiguous_or_nonplant"] += 1
                continue
            interference = float(row["coelute_inf"])
            if not math.isfinite(interference) or not 0 <= interference <= 0.3:
                counts["excluded_interference"] += 1
                continue
            values = [float(row[c]) for c in channels]
            if not all(math.isfinite(v) and v >= 0 for v in values):
                counts["excluded_invalid_intensity"] += 1
                continue
            item = proteins.setdefault(protein, {"psms": 0, "peptides": set(), "intensities": [0.0] * 10})
            item["psms"] += 1
            item["peptides"].add(row["peptide"])
            item["intensities"] = [a + b for a, b in zip(item["intensities"], values, strict=True)]
    control = proteins.get(BSA)
    if control is None or not all(math.isfinite(v) and v > 0 for v in control["intensities"]):
        raise ValueError("Positive BSA control required in every channel")
    return proteins, dict(counts)


def quantifiable(proteins, temperatures):
    anchor = min(range(10), key=lambda i: temperatures[i])
    return {
        key
        for key, item in proteins.items()
        if key != BSA and item["psms"] >= 10 and len(item["peptides"]) >= 2 and item["intensities"][anchor] > 0
    }


def normalized_observations(protein, replicate, proteins, channels, temperatures):
    """Fixed assay target construction, independently within each replicate."""
    if replicate not in REPLICATES or protein not in quantifiable(proteins, temperatures):
        raise ValueError("Unqualified protein/replicate")
    values = proteins[protein]["intensities"]
    control = proteins[BSA]["intensities"]
    corrected = [v / c for v, c in zip(values, control, strict=True)]
    anchor = min(range(10), key=lambda i: temperatures[i])
    reference = corrected[anchor]
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Positive within-replicate reference signal required")
    rows = []
    for i in sorted(range(10), key=lambda i: temperatures[i]):
        target = corrected[i] / reference
        if not math.isfinite(target):
            raise ValueError("Nonfinite normalized response")
        rows.append(
            {
                "id": f"{protein}_{replicate}_{channels[i]}",
                "protein": protein,
                "group": f"{DOI}:{replicate}",
                "values": [temperatures[i], temperatures[anchor], float(replicate in {"C3", "C4"})],
                "target": target,
            }
        )
    return rows


def protocol():
    return {
        "schema": VERSION,
        "source_commit": COMMIT,
        "study_doi": DOI,
        "partition": {p: list(reps) for p, reps in PARTITIONS.items()},
        "maximum_coisolation_interference": 0.3,
        "minimum_psms_per_replicate": 10,
        "minimum_distinct_peptides_per_replicate": 2,
        "aggregation": "sum reporter-channel intensities across retained PSMs assigned to one unambiguous plant locus",
        "target": "(protein_channel / BSA_channel) / (protein_reference_channel / BSA_reference_channel)",
        "reference": "lowest measured temperature of the same biological replicate; selected by metadata, not intensity",
        "covariates": ["temperature", "reference_temperature", "preparation_fresh"],
        "preparation_encoding": "0=frozen tissue preparation; 1=fresh tissue preparation (C3/C4)",
        "normalization_fit": "no learned global normalization, curve-fit filtering or held-out outcome tuning",
        "target_calibration": "within-sample reference measurement defines the response target; absolute abundance is not predicted",
        "selection": "all training-qualified proteins; report missing holdout quantification separately",
        "acceptance_minimum_skill": 0.0,
        "primary_assessment": "all ten points in the independent biological replicate, with non-reference-point sensitivity metrics",
        "independence": "groups are biological replicates; temperatures and proteins sharing a replicate are not independent experiments",
        "reproduction_scope": "partial measured-response reconstruction, not the author's global normalization/Tm pipeline",
    }


def export_tasks(raw_root, methods_root, output):
    raw_root, methods_root, output = (Path(p).absolute() for p in (raw_root, methods_root, output))
    tree = json.loads(ordinary(raw_root / "source_tree_selection.json").read_text())
    retrieved = {r["path"]: r for r in json.loads(ordinary(raw_root / "retrieval_api.json").read_text())}
    reviewed_mapping = json.loads(ordinary(methods_root / "temperature_mapping.json").read_text())
    source_protocol = json.loads(ordinary(raw_root / "protocol.json").read_text())
    if source_protocol["source_commit"] != COMMIT or source_protocol["paper_doi"] != DOI:
        raise ValueError("Source study or commit changed")
    if source_protocol["proposed_partition_before_model_fitting"] != protocol()["partition"]:
        raise ValueError("Previously declared biological-replicate partition changed")
    output.mkdir(parents=True, exist_ok=False)
    protocol_ref = save(output / "protocol.json", protocol())
    proteins, mappings, available, summaries, sources = {}, {}, {}, {}, []
    for replicate in REPLICATES:
        relative = f"R/data.rmbaseline/{replicate}.pp.f0.01.tsv"
        path = ordinary(raw_root / relative)
        content = path.read_bytes()
        git_sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if (
            len(content) != tree[relative]["size"]
            or git_sha != tree[relative]["sha"]
            or hashlib.sha256(content).hexdigest() != retrieved[relative]["sha256"]
        ):
            raise ValueError("Raw source object changed")
        metadata = ordinary(methods_root / f"R/meta/{replicate}.meta")
        if sha256_file(metadata) != reviewed_mapping[replicate]["sha256"]:
            raise ValueError("Reviewed channel metadata changed")
        channels, temperatures = read_mapping(metadata)
        if (
            channels != [c.replace("TMT10.", "TMT10-") for c in reviewed_mapping[replicate]["channel_names"]]
            or temperatures != reviewed_mapping[replicate]["temperatures"]
        ):
            raise ValueError("Reviewed temperature mapping changed")
        proteins[replicate], counts = aggregate_psms(path, channels)
        mappings[replicate] = channels, temperatures
        available[replicate] = quantifiable(proteins[replicate], temperatures)
        summaries[replicate] = {"counts": counts, "qualified_proteins": len(available[replicate])}
        sources.append(
            {"replicate": replicate, "raw": bound(path), "git_blob_sha": git_sha, "metadata": bound(metadata)}
        )
    source_ref = save(
        output / "source_evidence.json",
        {
            "schema": 1,
            "study_doi": DOI,
            "source_commit": COMMIT,
            "repository": "https://github.com/Sussman-Lab/at_thermal_proteome",
            "raw_sources": sources,
            "author_methods": bound(methods_root / "methods.Rnw"),
            "aggregation_implementation": bound(Path(__file__).resolve()),
            "replicate_conditions": {r: "fresh" if r in {"C3", "C4"} else "frozen" for r in REPLICATES},
        },
    )
    license_ref = save(
        output / "source_use.json",
        {
            "access": "Public author-deposited numerical measurements; attributed research analysis",
            "repository_license": "No repository license identified in the source review",
            "redistribution_permission": "not established by this adapter",
            "author_code_execution": False,
        },
    )
    training = set.intersection(*(available[r] for r in PARTITIONS["training"]))
    complete = training & available["C5"] & available["C6"]
    exclusions = [
        {"protein": p, "missing_qualified_replicates": [r for r in REPLICATES if p not in available[r]]}
        for p in sorted(training - complete)
    ]
    candidates = []
    for protein in sorted(complete):
        folder = output / "tasks" / protein.lower()
        task_id = "at_thermal_" + protein.lower()
        assets, assignments = {}, {}
        for phase, replicates in PARTITIONS.items():
            rows = []
            for replicate in replicates:
                rows.extend(normalized_observations(protein, replicate, proteins[replicate], *mappings[replicate]))
            assets[phase] = save(folder / (phase + ".json"), rows)
            assignments.update({row["id"]: {"group": row["group"], "phase": phase} for row in rows})
        groups_ref = save(folder / "groups.json", assignments)
        evidence_ref = save(
            folder / "group_evidence.json",
            {
                "schema": "context_groups.v1",
                "level": "biological_replicate",
                "study_doi": DOI,
                "sources": [source_ref, protocol_ref],
                "observations": assignments,
            },
        )
        context = {
            "organism": "Arabidopsis thaliana",
            "material": "plant protein extract",
            "preparation_encoding": "0=frozen; 1=fresh",
            "response_reference": "BSA-corrected lowest-temperature signal within the same replicate",
        }
        scope = {
            "protein": protein,
            "assay": "thermal_proteome_soluble_fraction",
            "target": "relative_soluble_abundance",
            "unit": "ratio_to_within_replicate_reference",
            "context": context,
            "covariates": [
                {
                    "name": "temperature",
                    "unit": "degC",
                    "minimum": min(min(t) for _, t in mappings.values()),
                    "maximum": max(max(t) for _, t in mappings.values()),
                },
                {
                    "name": "reference_temperature",
                    "unit": "degC",
                    "minimum": min(min(t) for _, t in mappings.values()),
                    "maximum": max(min(t) for _, t in mappings.values()),
                },
                {"name": "preparation_fresh", "unit": "binary_indicator", "minimum": 0.0, "maximum": 1.0},
            ],
            "grouping": {"level": "biological_replicate", "evidence_sha256": evidence_ref["sha256"]},
        }
        task_ref = save(
            folder / "task.json",
            {
                "schema": 1,
                "id": task_id,
                "stage": "stability",
                "plugin_id": "generated_context.v1",
                "scope": scope,
                "paper": {
                    "title": "Proteome-wide Analysis of Protein Thermal Stability in the Model Higher Plant Arabidopsis thaliana",
                    "doi": DOI,
                },
                "references": [source_ref, protocol_ref, evidence_ref],
                "data": assets,
                "minimum_skill": 0.0,
            },
        )
        candidates.append(
            {
                "id": task_id,
                "project_id": PROJECT,
                "task": task_ref,
                "protein_scope": {"kind": "protein", "identifier": protein},
                "property": scope["target"],
                "scenario": context,
                "question": "Predict the calibrated soluble response under recorded temperature and preparation conditions in an independent biological replicate",
                "groups": groups_ref,
                "group_basis": "replicate_group",
                "scenario_evidence": source_ref,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "target_models": 10000,
        "projects": [
            {
                "id": PROJECT,
                "title": "Arabidopsis protein thermal responses",
                "study_doi": DOI,
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6356070/",
                "source_evidence": source_ref,
                "license_evidence": license_ref,
                "license": "Public numerical data; repository license unspecified; redistribution permission unassessed",
            }
        ],
        "candidates": candidates,
    }
    save(output / "manifest.json", manifest)
    catalog = build_catalog(manifest)
    save(output / "catalog.json", catalog)
    report = {
        "schema": VERSION,
        "training_qualified_proteins": len(training),
        "exported_tasks": len(candidates),
        "catalog_statuses": catalog["summary"]["statuses"],
        "excluded_training_candidates": exclusions,
        "per_replicate": summaries,
        "new_trained_models": 0,
        "external_model_calls": 0,
        "future_scored": False,
    }
    save(output / "report.json", report)
    return report
