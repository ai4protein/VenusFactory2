"""Export measured food-yeast expression tasks with fixed genetic populations."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .model_tasks import bound_file, ordinary
from .research_project_catalog import SCHEMA, build_catalog
from agent.rsi.engine.storage import atomic_json, sha256_file
from .yeast_population import DOI, target_protocol

PROJECT = "yeast_food_population_expression"


def bound(path):
    path = ordinary(Path(path).absolute())
    return {"path": str(path), "sha256": sha256_file(path)}


def save(path, value):
    atomic_json(path, value)
    return bound(path)


def measured_rows(gene, measurements, samples):
    rows = {"training": [], "validation": [], "future": []}
    for sample in samples:
        identifier = sample["id"]
        observation = measurements["observations"].get(identifier)
        rna = measurements["rna"].get(identifier)
        if observation is None or rna is None:
            continue
        rows[sample["phase"]].append(
            {
                "id": gene + "_" + identifier,
                "protein": gene,
                "group": DOI + ":" + sample["population"],
                "values": [rna, sample["od_rna"], sample["od_protein"], sample["ploidy"]],
                "target": observation["target"],
            }
        )
    return rows


def export_tasks(quantification, output):
    quantification, output = Path(quantification).absolute(), Path(output).absolute()
    state = json.loads(ordinary(quantification / "status.json").read_text())
    if state["status"] != "completed":
        raise ValueError("Completed quantification required before task export")
    source_ref = bound(quantification / "source_evidence.json")
    source = json.loads(bound_file(source_ref).read_text())
    if source["schema"] != "yeast_food_population_quantification.v1" or source["study_doi"] != DOI:
        raise ValueError("Unexpected quantified study")
    plan = json.loads(bound_file(source["plan"]).read_text())
    protocol = json.loads(bound_file(source["protocol"]).read_text())
    if protocol != target_protocol() or plan["study_doi"] != DOI:
        raise ValueError("Frozen metadata or target protocol differs")
    genes = json.loads(ordinary(quantification / "gene_index.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    license_ref = save(
        output / "source_use.json",
        {
            "author_data": {"license": "CC-BY-4.0", "url": "https://zenodo.org/records/10567083"},
            "pride_report": {"license": "CC0", "url": "https://www.ebi.ac.uk/pride/archive/projects/PXD044523"},
            "reference_identifiers": {"license": "CC-BY-4.0", "release": "UniProt 2026_03"},
            "author_code_executed": False,
        },
    )
    candidates, excluded = [], []
    summary_rows = Counter()
    for gene, reference in sorted(genes.items()):
        measurements = json.loads(bound_file(reference).read_text())
        if measurements["gene"] != gene:
            raise ValueError("Gene identity differs from quantified source")
        rows = measured_rows(gene, measurements, plan["samples"])
        counts = {phase: len(values) for phase, values in rows.items()}
        groups = {phase: len({row["group"] for row in values}) for phase, values in rows.items()}
        summary_rows.update(counts)
        if counts["training"] < 40 or groups["training"] < 4 or not counts["validation"] or not counts["future"]:
            excluded.append(
                {"gene": gene, "rows": counts, "groups": groups, "reason": "insufficient_partition_coverage"}
            )
            continue
        task_id = "yeast_expression_" + re.sub(r"[^a-z0-9_]", "_", gene.casefold())
        folder = output / "tasks" / task_id
        assets = {phase: save(folder / (phase + ".json"), values) for phase, values in rows.items()}
        assignments = {
            row["id"]: {"phase": phase, "group": row["group"]} for phase, values in rows.items() for row in values
        }
        groups_ref = save(folder / "groups.json", assignments)
        # These references expose identities and hashes, never holdout labels.
        gene_ref = save(
            folder / "quantification_evidence.json",
            {
                "gene": gene,
                "quantified_source": reference,
                "source_evidence": source_ref,
                "rows": counts,
                "population_groups": groups,
                "endpoint": protocol["target"],
                "covariates": protocol["covariates"],
            },
        )
        evidence_ref = save(
            folder / "group_evidence.json",
            {
                "schema": "context_groups.v1",
                "level": "genetic_population",
                "study_doi": DOI,
                "sources": [source_ref, source["plan"], source["protocol"], gene_ref],
                "observations": assignments,
            },
        )
        context = {
            "organism": "Saccharomyces cerevisiae",
            "sample_origin": "food and fermentation",
            "growth_medium": "synthetic complete medium",
            "reference": "same-acquisition-batch pooled QC",
            "transcript_assay": "author-provided log2 TPM",
        }
        scope = {
            "protein": gene,
            "assay": "population_protein_expression",
            "target": "qc_relative_protein_abundance",
            "unit": "log2_ratio_to_pooled_qc",
            "context": context,
            "covariates": [
                {"name": name, "unit": unit, "minimum": domain[0], "maximum": domain[1]}
                for name, unit, domain in zip(
                    protocol["covariates"],
                    ["author_log2_tpm", "OD600", "OD600", "chromosome_sets"],
                    protocol["feature_domains"],
                    strict=True,
                )
            ],
            "grouping": {"level": "genetic_population", "evidence_sha256": evidence_ref["sha256"]},
        }
        task_ref = save(
            folder / "task.json",
            {
                "schema": 1,
                "id": task_id,
                "stage": "expression",
                "plugin_id": "generated_context.v1",
                "scope": scope,
                "paper": {
                    "title": "Species-wide quantitative transcriptomes and proteomes reveal distinct genetic control of gene expression variation in yeast",
                    "doi": DOI,
                },
                "references": [source_ref, source["plan"], source["protocol"], gene_ref, evidence_ref],
                "data": assets,
                "minimum_skill": 0.0,
            },
        )
        candidates.append(
            {
                "id": task_id,
                "project_id": PROJECT,
                "task": task_ref,
                "protein_scope": {"kind": "protein", "identifier": gene},
                "property": scope["target"],
                "scenario": context,
                "question": "Predict QC-relative protein abundance from transcript abundance and culture metadata in a held-out food-yeast population",
                "groups": groups_ref,
                "group_basis": "genetic_population",
                "scenario_evidence": source_ref,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "target_models": 10000,
        "projects": [
            {
                "id": PROJECT,
                "title": "Food-yeast population-specific protein expression",
                "study_doi": DOI,
                "url": "https://doi.org/" + DOI,
                "source_evidence": source_ref,
                "license_evidence": license_ref,
                "license": "Author data CC-BY-4.0; PRIDE report CC0; UniProt identifiers CC-BY-4.0",
            }
        ],
        "candidates": candidates,
    }
    save(output / "manifest.json", manifest)
    catalog = build_catalog(manifest)
    save(output / "catalog.json", catalog)
    report = {
        "schema": "yeast_food_population_tasks.v1",
        "source_genes": len(genes),
        "exported_tasks": len(candidates),
        "catalog_statuses": catalog["summary"]["statuses"],
        "available_observations": dict(summary_rows),
        "excluded": excluded,
        "new_trained_models": 0,
        "new_published_models": 0,
        "future_scored": False,
        "external_model_calls": 0,
    }
    save(output / "report.json", report)
    return report
