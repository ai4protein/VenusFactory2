"""Prepare protein-specific T-MAP tasks with a shared held-out-gene partition."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .model_tasks import load_task
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json, sha256_file
from .tmap_source import PAPER, SCHEMA, TARGET, UNIT, require, validate_source

PHASES = ("training", "validation", "future")
MINIMUM_GROUPS = {"training": 30, "validation": 8, "future": 8}


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def partition_genes(samples, seed):
    require(type(seed) is int and 0 <= seed < 2**32, "Integer partition seed required")
    genes = sorted(
        {item["gene"] for item in samples.values()},
        key=lambda gene: digest({"study": PAPER["doi"], "seed": seed, "gene": gene}),
    )
    require(len(genes) >= 60, "At least sixty distinct source-defined genes required")
    first, second = int(0.7 * len(genes)), int(0.85 * len(genes))
    return {
        gene: ("training" if index < first else "validation" if index < second else "future")
        for index, gene in enumerate(genes)
    }


def aggregate_protein(protein, samples, order, features, values, partitions):
    """Average log ratios within each gene; repeated measurements stay together."""
    grouped = defaultdict(list)
    for i, sample in enumerate(order):
        if np.isfinite(values[i]):
            grouped[samples[sample]["gene"]].append(i)
    rows = {phase: [] for phase in PHASES}
    evidence = {}
    for gene, indexes in sorted(grouped.items()):
        require(gene in partitions, "Missing global gene partition")
        expected = features[indexes[0]]
        require(
            all(np.array_equal(features[index], expected) for index in indexes), "Conflicting annotations within a gene"
        )
        identifier = protein.lower() + "__" + gene.lower()
        rows[partitions[gene]].append(
            {
                "id": identifier,
                "protein": protein,
                "group": gene,
                "values": expected.tolist(),
                "target": float(np.mean(values[indexes])),
            }
        )
        evidence[identifier] = {
            "gene": gene,
            "samples": [order[index] for index in indexes],
            "replicates_used": len(indexes),
        }
    return rows, evidence


def prepare(source_root, output, *, seed=20261027):
    source_root, output = Path(source_root), Path(output)
    require(
        source_root.is_absolute() and not any(p.is_symlink() for p in (source_root, *source_root.parents)),
        "Absolute nonsymlink source directory required",
    )
    require(
        output.is_absolute() and ".." not in output.parts and not output.exists(),
        "New absolute task directory required",
    )
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Symlink task output rejected")
    manifest_path = source_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest["schema"] == SCHEMA and manifest["status"] == "source_reconstructed",
        "Expected reconstructed T-MAP source",
    )
    require(manifest["target"] == {"name": TARGET, "unit": UNIT}, "Turnover endpoint changed")
    for name, expected in manifest["files_sha256"].items():
        require(
            Path(name).name == name and sha256_file(source_root / name) == expected, "Prepared source asset changed"
        )
    validate_source(manifest["source"])
    samples = json.loads((source_root / "samples.json").read_text())
    proteins = json.loads((source_root / "proteins.json").read_text())
    witness = json.loads((source_root / "source_witness.json").read_text())
    features = np.load(source_root / "features.npy", allow_pickle=False)
    targets = np.load(source_root / "targets.npy", allow_pickle=False)
    order = manifest["sample_order"]
    require(len(order) == len(set(order)) and set(order) == set(samples), "Sample identities differ")
    require(
        features.shape == (len(order), len(manifest["covariates"])) and targets.shape == (len(order), len(proteins)),
        "Source matrix dimensions differ",
    )
    require(
        np.isfinite(features).all() and np.isin(features, [0, 1]).all() and (features.sum(axis=1) == 1).all(),
        "Expected complete one-hot source functions",
    )
    require(len(proteins) == len(set(proteins)), "Repeated protein identities")
    partitions = partition_genes(samples, seed)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "tmap_task_partition.v1",
        "study_doi": PAPER["doi"],
        "seed": seed,
        "source_manifest": binding(manifest_path),
        "genes": partitions,
        "fractions": [0.7, 0.15, 0.15],
        "minimum_groups": MINIMUM_GROUPS,
        "aggregation": "Arithmetic mean of observed log2(H/L) values within each exact source gene; no missing-value imputation",
        "evaluation": "Held-out perturbed genes within this source study; future is the interface name for the final holdout, not a later observation date",
        "covariates": "Source-defined function category only; sample IDs, gene IDs, raw ratios and cohort-normalized quantities are not model inputs",
        "minimum_skill": 0,
        "models_trained": 0,
        "external_model_calls": 0,
    }
    atomic_json(output / "partition_protocol.json", protocol)
    common_refs = [
        binding(manifest_path),
        binding(source_root / "source.json"),
        binding(output / "partition_protocol.json"),
    ]
    tasks, excluded = [], []
    for column, protein in enumerate(proteins):
        rows, row_sources = aggregate_protein(protein, samples, order, features, targets[:, column], partitions)
        counts = {phase: len(rows[phase]) for phase in PHASES}
        if any(counts[phase] < MINIMUM_GROUPS[phase] for phase in PHASES):
            excluded.append({"protein": protein, "reason": "insufficient_observed_gene_groups", "groups": counts})
            continue
        folder = output / ("tmap_turnover_" + protein.lower())
        folder.mkdir()
        source_evidence = {
            "schema": "tmap_protein_witness.v1",
            "protein": protein,
            "original_sources": manifest["source"],
            "source_protein_row": witness["proteins"][protein],
            "observations": row_sources,
            "source_columns": {
                sample: witness["columns"][sample] for record in row_sources.values() for sample in record["samples"]
            },
            "target_transform": protocol["aggregation"],
        }
        atomic_json(folder / "source_evidence.json", source_evidence)
        refs = [*common_refs, binding(folder / "source_evidence.json")]
        groups = {
            "schema": "context_groups.v1",
            "level": "deleted_gene",
            "study_doi": PAPER["doi"],
            "sources": refs,
            "observations": {
                row["id"]: {"group": row["group"], "phase": phase} for phase in PHASES for row in rows[phase]
            },
        }
        atomic_json(folder / "groups.json", groups)
        group_ref = binding(folder / "groups.json")
        for phase in PHASES:
            atomic_json(folder / (phase + ".json"), rows[phase])
        scope = {
            "protein": protein,
            "assay": "LC-MS isotope ratio after a single one-hour chase",
            "target": TARGET,
            "unit": UNIT,
            "context": {
                "organism": "laboratory Saccharomyces cerevisiae",
                "study": PAPER["doi"],
                "genetic_context": "exact source-annotated deletion genes",
                "interpretation": "observed isotope turnover proxy; not intrinsic half-life",
            },
            "covariates": manifest["covariates"],
            "interface_version": "generated_context.v2",
            "grouping": {"level": "deleted_gene", "evidence_sha256": group_ref["sha256"]},
        }
        task = {
            "schema": 1,
            "id": folder.name,
            "stage": "stability",
            "plugin_id": "generated_context.v2",
            "scope": scope,
            "paper": PAPER,
            "references": [*refs, group_ref],
            "data": {phase: binding(folder / (phase + ".json")) for phase in PHASES},
            "minimum_skill": 0,
        }
        load_task(task)
        atomic_json(folder / "task.json", task)
        tasks.append({"protein": protein, "task": binding(folder / "task.json"), "groups": counts})
    require(tasks, "No protein tasks satisfy the predeclared data requirements")
    report = {
        "schema": "tmap_tasks.v1",
        "status": "offline_tasks_admitted",
        "source": binding(manifest_path),
        "partition_protocol": binding(output / "partition_protocol.json"),
        "scientific_scopes": len(tasks),
        "tasks": tasks,
        "excluded": excluded,
        "models_trained": 0,
        "models_published": 0,
        "external_model_calls": 0,
        "limitations": [
            "Candidate tasks, not acquired models",
            "Coarse function descriptors may have limited predictive power",
            "No final-label scoring performed",
            "No half-life or distant-species generalization claim",
        ],
    }
    atomic_json(output / "tasks.json", report)
    return report
