"""Import the public maize population tables without granting task admission.

Both tables contain negative values despite their ratio/FPKM descriptions.
Preserve the author-reported values without transformation; block acquisition.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

SOURCE_FILES = {
    "139955_1_supp_230491_phrhk9.xlsx": "da9c09a6d4c11669e5ebb0cf5e22b2282bf054f83722f366eb115f74f1a0ac75",
    "139955_1_supp_230492_phrhk9.xlsx": "7897a831fe651a58ca1595ff0fc3a97e7550a2c2e477600c59336b285885b293",
}
DOI = "10.1074/mcp.RA118.001021"
POLICY = {
    "protein_feature": "author_reported_table_S2_relative_abundance",
    "protein_transformation": "unresolved_nonpositive_reported_ratios",
    "minimum_unique_peptides": 2,
    "rna_feature": "author_reported_table_S3_value",
    "rna_transformation": "unresolved",
    "missing_value_imputation": False,
    "ready_for_acquisition": False,
    "blocking_conditions": [
        "Nonpositive protein values conflict with the reported ratio description",
        "RNA transformation and its cohort dependence require source evidence",
        "Genotype and assay-batch split protocol has not been frozen",
    ],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sample_label(value):
    match = re.fullmatch(r"([A-Za-z0-9_-]+)\s*\((Mixed|NSS|SS|TST)\)", str(value))
    require(match is not None, "Unexpected sample/population header")
    return match[1], match[2].upper()


def number(value, location):
    if value is None or (isinstance(value, str) and value.strip() in {"", "NA", "NaN"}):
        return None
    require(type(value) in (int, float), f"Non-numeric observation at {location}")
    require(math.isfinite(value), f"Non-finite observation at {location}")
    return float(value)


def indexed_rows(rows, gene_column, width, first_row):
    result = {}
    for excel_row, row in enumerate(rows, first_row):
        require(len(row) == width, "Unexpected table width")
        gene = row[gene_column]
        require(isinstance(gene, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", gene), "Invalid gene ID")
        require(gene not in result, "Duplicate gene ID")
        result[gene] = (excel_row, row)
    return result


def pair_tables(protein_header, protein_rows, rna_header, rna_rows):
    """Join exact gene/sample IDs, retaining source cells and measured targets."""
    require(list(protein_header[:2]) == ["Uniprot_Accession", "Ensembl_GID"], "Protein header mismatch")
    require((len(protein_header) - 2) % 4 == 0, "Protein block width mismatch")
    require(rna_header[0] == "Ensembl_GID", "RNA header mismatch")
    protein_samples = {}
    for col in range(2, len(protein_header), 4):
        require(
            list(protein_header[col + 1 : col + 4]) == ["Coverage", "#Unique Peptides", "#Peptides"],
            "Protein quality headers mismatch",
        )
        sample, population = sample_label(protein_header[col])
        require(sample not in protein_samples, "Duplicate protein sample")
        protein_samples[sample] = (population, col)
    rna_samples = {}
    for col in range(1, len(rna_header)):
        sample, population = sample_label(rna_header[col])
        require(sample not in rna_samples, "Duplicate RNA sample")
        require(sample in protein_samples, "RNA sample absent from protein table")
        require(population == protein_samples[sample][0], "Population label disagreement")
        rna_samples[sample] = (population, col)
    proteins = indexed_rows(protein_rows, 1, len(protein_header), 9)
    transcripts = indexed_rows(rna_rows, 0, len(rna_header), 3)
    require(set(transcripts) <= set(proteins), "RNA gene absent from protein table")
    excluded = Counter()
    genes = {}
    negative_rna_count = 0
    nonpositive_protein_observed = 0
    nonpositive_protein_retained = 0
    for gene in sorted(transcripts):
        protein_row, p = proteins[gene]
        rna_row, t = transcripts[gene]
        observations = []
        for sample in sorted(rna_samples):
            population, rna_col = rna_samples[sample]
            protein_col = protein_samples[sample][1]
            ratio = number(p[protein_col], f"S2:{protein_row}:{protein_col + 1}")
            unique = number(p[protein_col + 2], f"S2:{protein_row}:{protein_col + 3}")
            value = number(t[rna_col], f"S3:{rna_row}:{rna_col + 1}")
            if ratio is not None:
                nonpositive_protein_observed += ratio <= 0
            if unique is not None:
                require(unique >= 0 and unique.is_integer(), "Unique-peptide count must be a nonnegative integer")
            if ratio is None:
                excluded["missing_protein_ratio"] += 1
            elif unique is None:
                excluded["missing_unique_peptides"] += 1
            elif unique < POLICY["minimum_unique_peptides"]:
                excluded["fewer_than_two_unique_peptides"] += 1
            elif value is None:
                excluded["missing_rna"] += 1
            else:
                negative_rna_count += value < 0
                nonpositive_protein_retained += ratio <= 0
                observations.append(
                    {
                        "sample": sample,
                        "population": population,
                        "author_reported_protein": ratio,
                        "author_reported_rna": value,
                        "unique_peptides": int(unique),
                        "source_cells": {
                            "protein_ratio": [protein_row, protein_col + 1],
                            "unique_peptides": [protein_row, protein_col + 3],
                            "rna": [rna_row, rna_col + 1],
                        },
                    }
                )
        genes[gene] = {"gene_id": gene, "protein_accession": p[0], "observations": observations}
    counts = [len(g["observations"]) for g in genes.values()]
    return genes, {
        "protein_genes": len(proteins),
        "rna_genes": len(transcripts),
        "exact_gene_matches": len(genes),
        "protein_samples": len(protein_samples),
        "paired_samples": len(rna_samples),
        "measured_pairs_retained": sum(counts),
        "excluded_pairs": dict(sorted(excluded.items())),
        "genes_with_at_least_60_pairs_before_split": sum(n >= 60 for n in counts),
        "negative_rna_values_retained": negative_rna_count,
        "nonpositive_protein_values_observed_in_paired_samples": nonpositive_protein_observed,
        "nonpositive_protein_values_retained": nonpositive_protein_retained,
        "admitted_tasks": 0,
        "autonomous_models": 0,
    }


def export_source(source, output):
    """Write an immutable, source-bound numerical inventory, never runnable tasks."""
    import openpyxl

    source, output = Path(source), Path(output)
    require(not output.exists(), "Output already exists")
    for name, sha in SOURCE_FILES.items():
        require(digest(source / name) == sha, "Source file hash mismatch: " + name)
    tables = []
    for name, header_row in zip(SOURCE_FILES, (8, 2)):
        workbook = openpyxl.load_workbook(source / name, read_only=True, data_only=True)
        try:
            rows = iter(workbook["Sheet1"].iter_rows(min_row=header_row, values_only=True))
            tables.append((next(rows), list(rows)))
        finally:
            workbook.close()
    genes, summary = pair_tables(*tables[0], *tables[1])
    output.mkdir(parents=True, exist_ok=False)
    (output / "genes").mkdir()
    index = {}
    for gene, data in genes.items():
        path = output / "genes" / (gene + ".json")
        path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
        index[gene] = {
            "path": str(path.relative_to(output)),
            "sha256": digest(path),
            "observations": len(data["observations"]),
        }
    manifest = {
        "schema": "maize_population_source.v1",
        "study_doi": DOI,
        "source": {name: {"path": str((source / name).resolve()), "sha256": sha} for name, sha in SOURCE_FILES.items()},
        "implementation_sha256": digest(__file__),
        "policy": POLICY,
        "summary": summary,
        "genes": index,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
