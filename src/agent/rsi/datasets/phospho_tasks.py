"""Native Scientist tasks for condition-held-out yeast phosphosite responses."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .model_tasks import load_task
from .phospho_source import PAPER, SCHEMA, SITE, TARGET, UNIT, require, validate_source
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json, sha256_file

PHASES = ("training", "validation", "future")
MINIMUM_GROUPS = {"training": 30, "validation": 8, "future": 8}
MINIMUM_REPLICATES = 3
ASSETS = {
    "source.json",
    "conditions.json",
    "samples.json",
    "sites.json",
    "source_witness.json",
    "exclusions.json",
    "run_centering.json",
    "features.npy",
    "targets.npy",
    "source_rows.npy",
}


def task_identifier(site):
    require(SITE.fullmatch(site) is not None, "Unresolved site identity")
    return "yeast_phospho_" + site.lower().replace("-", "_")


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def partition_conditions(samples, seed):
    require(type(seed) is int and 0 <= seed < 2**32, "Integer partition seed required")
    codes = sorted(
        {item["condition"] for item in samples.values()},
        key=lambda code: digest({"study": PAPER["doi"], "seed": seed, "condition": code}),
    )
    require(len(codes) >= 60, "At least sixty independent condition codes required")
    a, b = int(0.7 * len(codes)), int(0.85 * len(codes))
    return {code: "training" if i < a else "validation" if i < b else "future" for i, code in enumerate(codes)}


def aggregate_site(site, samples, order, features, values, source_rows, partitions):
    match = SITE.fullmatch(site)
    require(match is not None, "Unresolved site identity")
    grouped = defaultdict(list)
    for i, sample in enumerate(order):
        if np.isfinite(values[i]):
            grouped[samples[sample]["condition"]].append(i)
    rows, evidence = {phase: [] for phase in PHASES}, {}
    for condition, indexes in sorted(grouped.items()):
        if len(indexes) < MINIMUM_REPLICATES:
            continue
        require(len({samples[order[i]]["replicate"] for i in indexes}) == len(indexes), "Repeated biological replicate")
        require(condition in partitions, "Missing global condition partition")
        expected = features[indexes[0]]
        require(all(np.array_equal(features[i], expected) for i in indexes), "Conflicting condition features")
        require(all(source_rows[i] >= 2 for i in indexes), "Observed value lacks source row")
        identifier = site.lower() + "__" + condition.lower()
        rows[partitions[condition]].append(
            {
                "id": identifier,
                "protein": match[1],
                "group": condition,
                "values": expected.tolist(),
                "target": float(np.mean(values[indexes])),
            }
        )
        evidence[identifier] = {
            "condition": condition,
            "replicates_used": len(indexes),
            "observations": [
                {
                    "sample": order[i],
                    "run": samples[order[i]]["run"],
                    "source_row": int(source_rows[i]),
                    "batch": samples[order[i]]["batch"],
                }
                for i in indexes
            ],
        }
    return rows, evidence


def prepare(source_root, output, *, seed=20261027):
    source_root, output = Path(source_root), Path(output)
    require(
        source_root.is_absolute() and not any(p.is_symlink() for p in (source_root, *source_root.parents)),
        "Absolute nonsymlink source required",
    )
    require(
        output.is_absolute() and ".." not in output.parts and not output.exists(),
        "New absolute task directory required",
    )
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Symlink output rejected")
    manifest_path = source_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest["schema"] == SCHEMA and manifest["status"] == "source_reconstructed",
        "Expected reconstructed phosphosite source",
    )
    require(manifest["target"] == {"name": TARGET, "unit": UNIT}, "Phosphosite endpoint changed")
    require(set(manifest["files_sha256"]) == ASSETS, "Missing prepared source asset")
    for name, expected in manifest["files_sha256"].items():
        path = source_root / name
        require(not path.is_symlink() and sha256_file(path) == expected, "Prepared source asset changed")
    validate_source(manifest["source"])
    samples = json.loads((source_root / "samples.json").read_text())
    sites = json.loads((source_root / "sites.json").read_text())
    features = np.load(source_root / "features.npy", allow_pickle=False)
    targets = np.load(source_root / "targets.npy", allow_pickle=False)
    source_rows = np.load(source_root / "source_rows.npy", allow_pickle=False)
    order = manifest["sample_order"]
    require(len(order) == len(set(order)) and set(order) == set(samples), "Sample identities differ")
    require(
        len(sites) == len(set(sites)) and all(SITE.fullmatch(site) for site in sites),
        "Invalid or duplicated phosphosites",
    )
    require(
        features.shape == (len(order), len(manifest["covariates"]))
        and targets.shape == source_rows.shape == (len(order), len(sites)),
        "Source matrix dimensions differ",
    )
    require(
        np.isfinite(features).all() and np.isin(features, [0, 1]).all() and (features.sum(axis=1) == 1).all(),
        "Expected one-hot condition metadata",
    )
    require(not np.isinf(targets).any() and source_rows.dtype.kind in "iu", "Invalid targets or source row witnesses")
    partitions = partition_conditions(samples, seed)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "yeast_phospho_partition.v1",
        "study_doi": PAPER["doi"],
        "seed": seed,
        "source_manifest": binding(manifest_path),
        "conditions": partitions,
        "fractions": [0.7, 0.15, 0.15],
        "minimum_groups": MINIMUM_GROUPS,
        "minimum_replicates_per_condition": MINIMUM_REPLICATES,
        "aggregation": "Mean of observed run-centered log intensities across at least three distinct biological replicates of each condition code; no imputation",
        "evaluation": "Unseen condition codes within the same laboratory yeast study and preparation batches; no new-batch, species, protein or chemical-family generalization claim",
        "covariates": "Author-defined condition-type indicators only; no run, batch, condition-code identifier or molecular measurements supplied as features",
        "normalization": "Each label is centered against observed sites from its own run; no training-to-holdout normalization fit",
        "models_trained": 0,
        "external_model_calls": 0,
    }
    atomic_json(output / "partition_protocol.json", protocol)
    common = [binding(manifest_path), binding(source_root / "source.json"), binding(output / "partition_protocol.json")]
    admitted, excluded = [], []
    for column, site in enumerate(sites):
        rows, evidence = aggregate_site(
            site, samples, order, features, targets[:, column], source_rows[:, column], partitions
        )
        counts = {phase: len(rows[phase]) for phase in PHASES}
        if any(counts[phase] < MINIMUM_GROUPS[phase] for phase in PHASES):
            excluded.append(
                {
                    "site": site,
                    "reason": "insufficient_condition_groups_with_three_observed_replicates",
                    "groups": counts,
                }
            )
            continue
        match = SITE.fullmatch(site)
        folder = output / task_identifier(site)
        folder.mkdir()
        atomic_json(
            folder / "source_evidence.json",
            {
                "schema": "yeast_phosphosite_witness.v1",
                "site": site,
                "protein": match[1],
                "residue": match[2],
                "position": int(match[3]),
                "observations": evidence,
                "original_sources": manifest["source"],
                "run_centering": binding(source_root / "run_centering.json"),
            },
        )
        refs = [*common, binding(folder / "source_evidence.json")]
        atomic_json(
            folder / "groups.json",
            {
                "schema": "context_groups.v1",
                "level": "treatment_condition",
                "study_doi": PAPER["doi"],
                "sources": refs,
                "observations": {
                    row["id"]: {"group": row["group"], "phase": phase} for phase in PHASES for row in rows[phase]
                },
            },
        )
        group_ref = binding(folder / "groups.json")
        for phase in PHASES:
            atomic_json(folder / (phase + ".json"), rows[phase])
        task = {
            "schema": 1,
            "id": folder.name,
            "stage": "other",
            "plugin_id": "generated_context.v3",
            "scope": {
                "protein": match[1],
                "assay": "DIA phosphosite mass spectrometry after short-duration treatment",
                "target": TARGET,
                "unit": UNIT,
                "context": {
                    "organism": "laboratory Saccharomyces cerevisiae BY4741",
                    "study": PAPER["doi"],
                    "site": site,
                    "residue": match[2],
                    "position": match[3],
                    "duration_minutes": "5",
                    "interpretation": "relative observed phosphosite signal, not occupancy or kinase activity",
                },
                "covariates": manifest["covariates"],
                "interface_version": "generated_context.v3",
                "grouping": {"level": "treatment_condition", "evidence_sha256": group_ref["sha256"]},
            },
            "paper": PAPER,
            "references": [*refs, group_ref],
            "data": {phase: binding(folder / (phase + ".json")) for phase in PHASES},
            "minimum_skill": 0,
        }
        load_task(task)
        atomic_json(folder / "task.json", task)
        admitted.append({"protein": match[1], "site": site, "task": binding(folder / "task.json"), "groups": counts})
    require(admitted, "No phosphosite tasks satisfy the declared data requirements")
    report = {
        "schema": "yeast_phospho_tasks.v1",
        "status": "offline_tasks_admitted",
        "source": binding(manifest_path),
        "partition_protocol": binding(output / "partition_protocol.json"),
        "scientific_scopes": len(admitted),
        "distinct_proteins": len({item["protein"] for item in admitted}),
        "tasks": admitted,
        "excluded": excluded,
        "models_trained": 0,
        "models_published": 0,
        "external_model_calls": 0,
        "limitations": [
            "Candidate site-specific tasks, not acquired models",
            "Same-study condition holdout only",
            "Coarse condition descriptors may have limited predictive power",
            "No final-label scoring performed",
        ],
    }
    atomic_json(output / "tasks.json", report)
    return report
