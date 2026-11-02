"""Export audited Y3K protein/condition tasks under a common genotype holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .model_tasks import PHASES, bound_file
from .research_project_catalog import SCHEMA as CATALOG_SCHEMA
from .research_project_catalog import build_catalog
from agent.rsi.engine.storage import atomic_json, sha256_file
from .y3k_source import FEATURES, PAPER, SHEETS, UNIT, validate_source
from .y3k_source import SCHEMA as SOURCE_SCHEMA

SCHEMA = "y3k_amino_acid_tasks.v1"
SEED = "y3k-common-deletion-genotype-holdout-20261025-v1"
PROJECT = "y3k_amino_acid_fold_change"
MINIMUM = {"training": 40, "validation": 10, "future": 10}
DOMAIN = (-64.0, 64.0)
GLOBAL_FILES = {"source.json", "strain_metadata.json", "protein_metadata.json", "excluded_proteins.json"}
CONDITION_FILES = {"features.npy", "targets.npy", "genotypes.json", "proteins.json", "row_sources.json"}
FILES = GLOBAL_FILES | {f"{condition}/{name}" for condition in SHEETS for name in CONDITION_FILES}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    atomic_json(path, value)
    return {"path": str(path), "sha256": sha256_file(path)}


def identities(values, kind):
    require(
        isinstance(values, list)
        and values
        and all(isinstance(v, str) and v.strip() == v and 0 < len(v) <= 100 for v in values),
        f"Invalid {kind} identifiers",
    )
    require(len(values) == len(set(values)), f"Repeated {kind} identifiers")


def load(source):
    require(
        isinstance(source, dict)
        and set(source) == {"schema", "matrix", "partition_plan", "audit", "source_terms"}
        and source["schema"] == SCHEMA,
        "Bound matrix, audit, partition and source terms required",
    )
    paths = {name: bound_file(source[name]) for name in ("matrix", "partition_plan", "audit", "source_terms")}
    require(
        all(paths[name].stat().st_size <= 8 * 1024**2 for name in ("matrix", "partition_plan", "audit")),
        "Oversized source metadata",
    )
    matrix, plan, audit = (read(paths[name]) for name in ("matrix", "partition_plan", "audit"))
    require(
        matrix.get("schema") == SOURCE_SCHEMA
        and matrix.get("status") == "source_reconstructed"
        and matrix.get("acquisition_admitted") is False
        and set(matrix.get("files_sha256", {})) == FILES,
        "Unsupported reconstructed Y3K matrix",
    )
    assets = {
        name: bound_file({"path": str(paths["matrix"].parent / name), "sha256": sha})
        for name, sha in matrix["files_sha256"].items()
    }
    require(read(assets["source.json"]) == matrix["source"], "Matrix source declaration differs")
    validate_source(matrix["source"])
    require(
        source["source_terms"] in matrix["source"]["references"], "Source terms must be bound with the original source"
    )
    require(
        matrix["covariates"] == [{"name": n, "unit": UNIT} for n in FEATURES]
        and matrix["target"] == {"name": "protein_abundance_log2_fold_change", "unit": UNIT},
        "Published covariates or target units differ",
    )
    require(
        audit.get("passed") is True
        and audit.get("raw_hashes_rechecked") is True
        and audit.get("matrix_manifest_sha256") == source["matrix"]["sha256"],
        "Independent source audit is missing or stale",
    )
    require(
        plan.get("schema") == "y3k_partition_plan.v1"
        and plan.get("seed") == SEED
        and plan.get("matrix_manifest") == source["matrix"]
        and plan.get("acquisition_admitted") is False
        and plan.get("models_trained") == 0
        and plan.get("external_model_calls") == 0
        and set(plan.get("conditions", {})) == set(SHEETS),
        "Frozen common-genotype protocol differs",
    )
    metadata = read(assets["strain_metadata.json"])
    result, common = {}, None
    for condition in SHEETS:
        condition_assets = {name: assets[f"{condition}/{name}"] for name in CONDITION_FILES}

        genes, proteins, witness = (
            read(condition_assets["genotypes.json"]),
            read(condition_assets["proteins.json"]),
            read(condition_assets["row_sources.json"]),
        )
        identities(genes, "genotype")
        identities(proteins, "protein")
        require(
            len(genes) <= 1000 and len(proteins) <= 5000 and set(genes) == set(metadata),
            "Source dimensions or declared genotypes differ",
        )
        if common is None:
            common = set(genes)
        require(set(genes) == common, "Genotypes differ across conditions")
        ordered = sorted(genes, key=lambda g: hashlib.sha256((SEED + "\0" + g).encode()).hexdigest())
        a, b = int(0.6 * len(genes)), int(0.8 * len(genes))
        expected = {"training": ordered[:a], "validation": ordered[a:b], "future": ordered[b:]}
        require(plan.get("partitions") == expected, "Partition assignments differ from the frozen genotype split")
        x = np.load(condition_assets["features.npy"], mmap_mode="r", allow_pickle=False)
        y = np.load(condition_assets["targets.npy"], mmap_mode="r", allow_pickle=False)
        require(
            x.dtype == y.dtype == np.dtype("float64")
            and x.shape == (len(genes), len(FEATURES))
            and y.shape == (len(genes), len(proteins))
            and not np.isinf(x).any()
            and not np.isinf(y).any(),
            "Invalid source matrix dimensions or values",
        )
        finite = x[np.isfinite(x)]
        require(
            (finite >= DOMAIN[0]).all() and (finite <= DOMAIN[1]).all(),
            "Measured log2 covariates exceed the declared computational domain",
        )
        require(
            set(witness) == {"columns", "features", "proteins", "excluded_protein_rows"}
            and set(witness["features"]) == set(FEATURES)
            and set(witness["proteins"]) == set(proteins)
            and len(witness["columns"]) == len(genes),
            "Incomplete source-cell witnesses",
        )
        for index, record in enumerate(witness["columns"]):
            require(
                set(record) == {"column", "source_label", "genotype"}
                and record["column"] == index + 3
                and record["genotype"] == genes[index]
                and isinstance(record["source_label"], str)
                and record["source_label"].startswith("∆"),
                "Invalid source column witness",
            )
        for record in witness["features"].values():
            require(
                set(record) == {"row", "unit"}
                and type(record["row"]) is int
                and record["row"] >= 2
                and record["unit"] == UNIT,
                "Invalid feature-cell witness",
            )
        for record in witness["proteins"].values():
            require(
                set(record) == {"row", "name", "locus"}
                and type(record["row"]) is int
                and record["row"] >= 2
                and all(isinstance(record[k], str) and record[k] for k in ("name", "locus")),
                "Invalid protein-cell witness",
            )
        positions = {gene: i for i, gene in enumerate(genes)}
        complete = np.isfinite(x).all(axis=1)
        indices = {
            phase: np.asarray([positions[g] for g in expected[phase] if complete[positions[g]]], dtype=int)
            for phase in PHASES
        }
        counts = {phase: np.isfinite(y[index]).sum(axis=0) for phase, index in indices.items()}
        expected_counts = {
            protein: {phase: int(counts[phase][j]) for phase in PHASES} for j, protein in enumerate(proteins)
        }
        eligible = sum(all(counts[phase][j] >= MINIMUM[phase] for phase in PHASES) for j in range(len(proteins)))
        require(
            plan["conditions"][condition]
            == {
                "resolved_proteins": len(proteins),
                "proteins_with_minimum_partition_pairs": eligible,
                "minimum_rows": MINIMUM,
                "per_protein_counts": expected_counts,
            },
            "Partition feasibility differs from source measurements",
        )
        result[condition] = (genes, proteins, x, y, witness, indices, counts)
    return matrix, result


def prepare(source, output):
    output = Path(output)
    require(
        output.is_absolute()
        and ".." not in output.parts
        and not output.exists()
        and not any(p.is_symlink() for p in (output, *output.parents)),
        "New absolute nonsymlink task output required",
    )
    matrix, conditions = load(source)
    output.mkdir(parents=True, exist_ok=False)
    source_ref = save(
        output / "source_evidence.json",
        {
            "schema": SCHEMA,
            "input_bindings": source,
            "source": matrix["source"],
            "target": matrix["target"],
            "matrix_files_sha256": matrix["files_sha256"],
            "observation_unit": matrix["observation_unit"],
            "preprocessing": matrix["preprocessing"],
            "limitations": matrix["limitations"],
            "pairing": "Metabolite and protein strain summaries from the same Y3K condition; published triplicates are not separate observations",
            "partition": "One common deletion-genotype split across proteins and both growth conditions",
            "missingness": "Use finite targets and complete measured feature vectors only; no new imputation",
            "domain": "Computational log2 input range -64..64; not evidence of predictive validity throughout that range",
        },
    )
    candidates, excluded, task_ids = [], [], set()
    for condition, (genes, proteins, x, y, witness, indices, counts) in conditions.items():
        context = {
            "organism": "Saccharomyces cerevisiae, laboratory S288c-derived strains",
            "application": "protein abundance change from measured amino-acid changes",
            "perturbation": "single gene deletion",
            "growth_condition": condition,
            "study_design": "paired multi-omic strain summaries",
            "quantity_scale": "published mean log2 KO relative to matched WT",
        }
        for column, protein in enumerate(proteins):
            if any(counts[phase][column] < MINIMUM[phase] for phase in PHASES):
                excluded.append(
                    {"condition": condition, "protein": protein, "reason": "insufficient_partition_observations"}
                )
                continue
            task_id = "y3k_aa_" + hashlib.sha256((condition + "\0" + protein).encode()).hexdigest()[:20]
            require(task_id not in task_ids, "Scientific task identity collision")
            task_ids.add(task_id)
            folder = output / "tasks" / task_id
            folder.mkdir(parents=True)
            assets, assignments, observations = {}, {}, {}
            for phase in PHASES:
                rows = []
                for index in indices[phase]:
                    if not np.isfinite(y[index, column]):
                        continue
                    index = int(index)
                    gene = genes[index]
                    identity = task_id + "_" + gene
                    group = PAPER["doi"] + ":" + gene
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
                    observations[identity] = {
                        "genotype": gene,
                        "matrix_row": index,
                        "protein_column": column,
                        "source_column": witness["columns"][index]["column"],
                    }
                assets[phase] = save(folder / (phase + ".json"), rows)
            witness_ref = save(
                folder / "row_sources.json",
                {
                    "sheet": SHEETS[condition],
                    "protein": witness["proteins"][protein],
                    "features": witness["features"],
                    "observations": observations,
                },
            )
            groups_ref = save(folder / "group_assignments.json", assignments)
            evidence_ref = save(
                folder / "group_evidence.json",
                {
                    "schema": "context_groups.v1",
                    "level": "deleted_gene",
                    "study_doi": PAPER["doi"],
                    "sources": [source_ref, source["partition_plan"], witness_ref],
                    "observations": assignments,
                },
            )
            scope = {
                "interface_version": "generated_context.v2",
                "protein": protein,
                "assay": "published_y3k_protein_fold_change",
                "target": matrix["target"]["name"],
                "unit": UNIT,
                "context": context,
                "covariates": [{"name": n, "unit": UNIT, "minimum": DOMAIN[0], "maximum": DOMAIN[1]} for n in FEATURES],
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
                    "paper": PAPER,
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
                    "question": "Predict this protein's measured abundance change in a held-out laboratory yeast deletion genotype under the specified metabolic condition from measured amino-acid changes",
                    "groups": groups_ref,
                    "group_basis": "deleted_gene",
                    "scenario_evidence": source_ref,
                }
            )
    load(source)
    manifest = {
        "schema": CATALOG_SCHEMA,
        "target_models": 10000,
        "projects": [
            {
                "id": PROJECT,
                "title": "Y3K protein abundance response across measured metabolic conditions",
                "study_doi": PAPER["doi"],
                "url": "https://doi.org/" + PAPER["doi"],
                "source_evidence": source_ref,
                "license_evidence": source["source_terms"],
                "license": "Public supplementary data used for academic research; original author and publisher rights retained",
            }
        ],
        "candidates": candidates,
    }
    save(output / "manifest.json", manifest)
    catalog = build_catalog(manifest)
    save(output / "catalog.json", catalog)
    report = {
        "schema": SCHEMA,
        "exported_tasks": len(candidates),
        "catalog_statuses": catalog["summary"]["statuses"],
        "excluded": excluded,
        "minimum_rows": MINIMUM,
        "covariates": len(FEATURES),
        "future_scored": False,
        "external_model_calls": 0,
        "new_trained_models": 0,
        "new_published_models": 0,
    }
    save(output / "report.json", report)
    return report
