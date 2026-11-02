"""Export source-bound amino-acid/protein tasks under one frozen genotype split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .model_tasks import PHASES, bound_file
from .protpredict_source import FEATURES, PAPER_DOI, validate_source
from .protpredict_source import SCHEMA as SOURCE_SCHEMA
from .research_project_catalog import SCHEMA as CATALOG_SCHEMA
from .research_project_catalog import build_catalog
from agent.rsi.engine.storage import atomic_json, sha256_file

SCHEMA = "protpredict_amino_acid_tasks.v1"
SEED = "protpredict-aa-genotype-holdout-20261025-v1"
PROJECT = "protpredict_amino_acid_abundance"
FILES = {
    "features.npy",
    "targets.npy",
    "genotypes.json",
    "proteins.json",
    "covariates.json",
    "row_provenance.json",
    "source.json",
}
CAPS = {"training": 600, "validation": 200, "future": 200}
MINIMUM = {"training": 40, "validation": 10, "future": 10}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    atomic_json(path, value)
    return {"path": str(path), "sha256": sha256_file(path)}


def load(source):
    require(
        isinstance(source, dict)
        and set(source) == {"schema", "matrix", "partition_plan", "license"}
        and source["schema"] == SCHEMA,
        "Bound matrix, prospective partition and repository license required",
    )
    matrix_path = bound_file(source["matrix"])
    plan_path = bound_file(source["partition_plan"])
    require(matrix_path.stat().st_size <= 1048576 and plan_path.stat().st_size <= 2097152, "Oversized source metadata")
    bound_file(source["license"])
    matrix, plan = read(matrix_path), read(plan_path)
    require(
        matrix.get("schema") == SOURCE_SCHEMA
        and matrix.get("status") == "source_reconstructed"
        and set(matrix.get("files_sha256", {})) == FILES
        and matrix.get("acquisition_admitted") is False,
        "Unsupported reconstructed matrix",
    )
    assets = {
        name: bound_file({"path": str(matrix_path.parent / name), "sha256": sha})
        for name, sha in matrix["files_sha256"].items()
    }
    require(read(assets["source.json"]) == matrix["source"], "Matrix source declaration differs")
    validate_source(matrix["source"])
    require(source["license"] in matrix["source"]["references"], "Repository license must be source-bound")
    genes, proteins = read(assets["genotypes.json"]), read(assets["proteins.json"])
    for values in (genes, proteins):
        require(
            isinstance(values, list)
            and values
            and len(values) == len(set(values))
            and all(isinstance(v, str) and v.strip() == v and 0 < len(v) <= 100 for v in values),
            "Unique exact genotype and protein identities required",
        )
    require(len(genes) <= 10000 and len(proteins) <= 5000, "Source matrix exceeds admitted dimensions")
    require(
        read(assets["covariates.json"]) == [{"name": n, "unit": "mM"} for n in FEATURES],
        "All 19 measured covariates required",
    )
    x = np.load(assets["features.npy"], mmap_mode="r", allow_pickle=False)
    y = np.load(assets["targets.npy"], mmap_mode="r", allow_pickle=False)
    require(
        x.dtype == y.dtype == np.dtype("float64")
        and x.shape == (len(genes), len(FEATURES))
        and y.shape == (len(genes), len(proteins))
        and np.isfinite(x).all()
        and (x >= 0).all()
        and (x <= 1000).all()
        and not np.isinf(y).any()
        and not (y < 0).any(),
        "Invalid source matrix shape, values or predeclared concentration domains",
    )
    provenance = read(assets["row_provenance.json"])
    require(isinstance(provenance, dict) and set(provenance) == set(genes), "Complete genotype row provenance required")
    for row in provenance.values():
        require(
            isinstance(row, dict)
            and set(row) == {"protein_csv_rows", "metabolite_xls_row"}
            and isinstance(row["protein_csv_rows"], list)
            and row["protein_csv_rows"]
            and all(type(i) is int and i >= 2 for i in row["protein_csv_rows"])
            and type(row["metabolite_xls_row"]) is int
            and row["metabolite_xls_row"] >= 2,
            "Invalid source row provenance",
        )
    ordered = sorted(genes, key=lambda g: hashlib.sha256((SEED + "\0" + g).encode()).hexdigest())
    a, b = int(0.6 * len(genes)), int(0.8 * len(genes))
    expected = {"training": ordered[:a], "validation": ordered[a:b], "future": ordered[b:]}
    require(
        plan.get("schema") == "protpredict_partition_plan.v1"
        and plan.get("seed") == SEED
        and plan.get("matrix_manifest_sha256") == source["matrix"]["sha256"]
        and plan.get("partitions") == expected
        and plan.get("per_task_caps") == CAPS
        and plan.get("minimum_rows") == MINIMUM
        and plan.get("declared_proteins") == len(proteins)
        and plan.get("acquisition_admitted") is False
        and plan.get("models_trained") == 0
        and plan.get("external_model_calls") == 0,
        "Partition plan differs from the frozen common-genotype protocol",
    )
    lookup = {gene: i for i, gene in enumerate(genes)}
    indices = {phase: np.asarray([lookup[g] for g in expected[phase]], dtype=int) for phase in PHASES}
    counts = {phase: np.isfinite(y[index]).sum(axis=0) for phase, index in indices.items()}
    eligible = sum(all(counts[phase][i] >= MINIMUM[phase] for phase in PHASES) for i in range(len(proteins)))
    require(
        plan.get("proteins_with_sufficient_counts") == eligible, "Plan feasibility count differs from measured targets"
    )
    return matrix, genes, proteins, x, y, provenance, indices, counts


def prepare(source, output):
    output = Path(output)
    require(
        output.is_absolute()
        and ".." not in output.parts
        and not output.exists()
        and not any(p.is_symlink() for p in [output, *output.parents]),
        "New absolute nonsymlink task output required",
    )
    matrix, genes, proteins, x, y, provenance, indices, counts = load(source)
    output.mkdir(parents=True, exist_ok=False)
    source_ref = save(
        output / "source_evidence.json",
        {
            "schema": SCHEMA,
            "input_bindings": source,
            "source": matrix["source"],
            "target": matrix["target"],
            "matrix_files_sha256": matrix["files_sha256"],
            "population": "Laboratory S288c-derived Saccharomyces cerevisiae deletion strains",
            "pairing": "Exact deleted-gene match across studies; not paired measurements of the same biological sample",
            "feature_domain": "19 measured amino-acid concentrations, mM, fixed admitted range 0..1000",
            "partition": "One common whole-genotype split for every protein; first finite targets up to frozen caps",
            "source_hashes_verified_at_export": True,
        },
    )
    context = {
        "organism": "Saccharomyces cerevisiae, laboratory S288c-derived strains",
        "application": "protein abundance from measured intracellular amino-acid concentrations",
        "perturbation": "gene deletion",
        "study_design": "cross-study exact genotype matching",
        "quantity_scale": "unchanged author-reported abundance",
    }
    candidates, excluded, identifiers = [], [], set()
    for column, protein in enumerate(proteins):
        if any(counts[phase][column] < MINIMUM[phase] for phase in PHASES):
            excluded.append({"protein": protein, "reason": "insufficient_partition_observations"})
            continue
        task_id = "protpredict_aa_" + hashlib.sha256(protein.encode()).hexdigest()[:20]
        require(task_id not in identifiers, "Protein task identifier collision")
        identifiers.add(task_id)
        folder = output / "tasks" / task_id
        folder.mkdir(parents=True)
        assets, assignments, witnesses = {}, {}, {}
        for phase in PHASES:
            selected = [int(i) for i in indices[phase] if np.isfinite(y[i, column])][: CAPS[phase]]
            rows = []
            for index in selected:
                genotype = genes[index]
                identity, group = task_id + "_" + genotype, PAPER_DOI + ":" + genotype
                rows.append(
                    {
                        "id": identity,
                        "protein": protein,
                        "group": group,
                        "values": x[index].tolist(),
                        "target": float(y[index, column]),
                    }
                )
                assignments[identity] = {"group": group, "phase": phase}
                witnesses[identity] = {
                    "genotype": genotype,
                    "matrix_row": index,
                    "protein_column": column,
                    **provenance[genotype],
                }
            assets[phase] = save(folder / (phase + ".json"), rows)
        witness_ref = save(folder / "row_sources.json", witnesses)
        groups_ref = save(folder / "group_assignments.json", assignments)
        evidence_ref = save(
            folder / "group_evidence.json",
            {
                "schema": "context_groups.v1",
                "level": "deleted_gene",
                "study_doi": PAPER_DOI,
                "sources": [source_ref, source["partition_plan"], witness_ref],
                "observations": assignments,
            },
        )
        scope = {
            "interface_version": "generated_context.v2",
            "protein": protein,
            "assay": "deposited_proteome_abundance",
            "target": "protein_abundance",
            "unit": "author_reported_protein_abundance",
            "context": context,
            "covariates": [{"name": n, "unit": "mM", "minimum": 0.0, "maximum": 1000.0} for n in FEATURES],
            "grouping": {"level": "deleted_gene", "evidence_sha256": evidence_ref["sha256"]},
        }
        task_ref = save(
            folder / "task.json",
            {
                "schema": 1,
                "id": task_id,
                "stage": "expression",
                "plugin_id": "generated_context.v2",
                "scope": scope,
                "paper": matrix["source"]["paper"],
                "references": [source_ref, source["partition_plan"], witness_ref, evidence_ref],
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
                "question": "Predict the measured abundance of this protein in a held-out laboratory yeast deletion genotype from measured amino-acid concentrations",
                "groups": groups_ref,
                "group_basis": "deleted_gene",
                "scenario_evidence": source_ref,
            }
        )
    # Recheck immutable inputs after materialization. Task rows bind the exported
    # values; the compact source witness retains the original matrix/raw hashes.
    load(source)
    manifest = {
        "schema": CATALOG_SCHEMA,
        "target_models": 10000,
        "projects": [
            {
                "id": PROJECT,
                "title": "Laboratory yeast protein abundance from amino-acid measurements",
                "study_doi": PAPER_DOI,
                "url": "https://doi.org/" + PAPER_DOI,
                "source_evidence": source_ref,
                "license_evidence": source["license"],
                "license": "MIT author repository; underlying measurements retain the cited source terms",
            }
        ],
        "candidates": candidates,
    }
    save(output / "manifest.json", manifest)
    catalog = build_catalog(manifest)
    save(output / "catalog.json", catalog)
    report = {
        "schema": SCHEMA,
        "source_proteins": len(proteins),
        "source_genotypes": len(genes),
        "exported_tasks": len(candidates),
        "catalog_statuses": catalog["summary"]["statuses"],
        "excluded": excluded,
        "per_task_caps": CAPS,
        "covariates": len(FEATURES),
        "future_scored": False,
        "external_model_calls": 0,
        "new_trained_models": 0,
        "new_published_models": 0,
    }
    save(output / "report.json", report)
    return report
