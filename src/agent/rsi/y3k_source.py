"""Reconstruct Y3K strain-level amino-acid/protein fold changes from source tables.

Published triplicate summaries are one observation per deletion genotype and
condition. Missing measurements and ambiguous protein groups are not filled in.
This module prepares source evidence, not admitted tasks or trained models.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .storage import atomic_json, sha256_file

SCHEMA = "y3k_source.v1"
PAPER = {
    "title": "Mitochondrial protein functions elucidated by multi-omic mass spectrometry profiling",
    "doi": "10.1038/nbt.3683",
}
SHEETS = {"respiration": "KO vs WT_Resp (∆LFQ)", "fermentation": "KO vs WT_Ferm (∆LFQ)"}
FEATURES = (
    "Alanine",
    "Aspartic Acid",
    "Glutamic Acid",
    "Phenylalanine",
    "Glycine",
    "Histidine",
    "Isoleucine",
    "Lysine",
    "Leucine",
    "Methionine",
    "Asparagine",
    "Proline",
    "Glutamine",
    "Serine",
    "Threonine",
    "Valine",
    "Tryptophan",
    "Tyrosine (3TMS)",
    "Cysteine",
)
UNIT = "log2_fold_change_vs_matched_WT"
# Exact display aliases present in Supplementary Table 1. The expected locus
# and complete source label are checked; no fuzzy identity matching is used.
STRAIN_ALIASES = {
    "ARG5": ("YER069W", "ARG5,6"),
    "RDL2": ("YOR286W", "RDL2 (AIM42)"),
    "SHE9": ("YDR393W", "SHE9 (MDM33)"),
}
LOCUS = re.compile(r"Y[A-P][LR][0-9]{3}[CW](?:-[A-Z])?")
PROTEIN_LOCUS = re.compile(r"(?:Y[A-P][LR][0-9]{3}[CW](?:-[A-Z])?|Q[0-9]{4}|R[0-9]{4}[CW])")
ACCESSION = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[1-9][0-9]*)?")


def require(value, message):
    if not value:
        raise ValueError(message)


def bound(binding):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "Source path and digest required")
    require(
        isinstance(binding["path"], str) and re.fullmatch(r"[0-9a-f]{64}", str(binding["sha256"])),
        "Invalid source binding",
    )
    p = Path(binding["path"])
    require(
        p.is_absolute() and p.is_file() and not any(x.is_symlink() for x in (p, *p.parents)),
        "Absolute nonsymlink source required",
    )
    require(p.stat().st_size <= 128 * 1024**2 and sha256_file(p) == binding["sha256"], "Source size or digest differs")
    return p


def validate_source(source):
    require(
        isinstance(source, dict)
        and set(source) == {"schema", "paper", "strains", "biomolecules", "quantitative", "references"},
        "Invalid Y3K source declaration",
    )
    require(source["schema"] == SCHEMA and source["paper"] == PAPER, "Unexpected source study")
    require(
        isinstance(source["references"], list) and 1 <= len(source["references"]) <= 8,
        "Bound source documentation required",
    )
    paths = {name: bound(source[name]) for name in ("strains", "biomolecules", "quantitative")}
    require(len(set(paths.values())) == 3, "Distinct source workbooks required")
    for reference in source["references"]:
        bound(reference)
    return paths


def strain_index(rows):
    """Resolve exact source labels, systematic loci and three documented aliases."""
    index, records = {}, {}
    for excel_row, row in enumerate(rows, start=2):
        if not any(x is not None for x in row):
            continue
        require(len(row) == 5, "Unexpected strain table width")
        locus, standard = row[:2]
        if locus == standard == "None (WT)":
            continue
        require(isinstance(locus, str) and LOCUS.fullmatch(locus), "Invalid deleted-gene locus")
        require(isinstance(standard, str) and standard.strip() == standard and standard, "Invalid strain display name")
        require(locus not in records, "Repeated source deletion genotype")
        records[locus] = {"source_row": excel_row, "systematic": locus, "standard": standard, "accession": row[3]}
        labels = {locus, standard}
        labels.update(alias for alias, expected in STRAIN_ALIASES.items() if (locus, standard) == expected)
        for label in labels:
            require(label not in index, "Ambiguous strain alias")
            index[label] = locus
    return index, records


def protein_index(rows, strain_records=None):
    """Resolve single-accession loci, retaining ambiguities and cross-table repairs."""
    strain_records = strain_records or {}
    by_locus = defaultdict(list)
    for excel_row, row in enumerate(rows, start=2):
        if not any(x is not None for x in row) or row[0] != "PROTEIN":
            continue
        require(len(row) == 5, "Unexpected biomolecule table width")
        _, _, gene, locus, accession = row
        require(isinstance(locus, str) and locus.strip() == locus and locus, "Invalid protein locus")
        require(isinstance(accession, str), "Missing protein accession")
        record = {
            "source_row": excel_row,
            "gene": gene if isinstance(gene, str) else None,
            "systematic": locus.upper(),
            "accession": accession,
        }
        if not isinstance(gene, str):
            record["original_gene_value"] = gene.isoformat() if isinstance(gene, (date, datetime)) else repr(gene)
            record["original_gene_type"] = type(gene).__name__
            witness = strain_records.get(locus.upper())
            if (
                isinstance(gene, (date, datetime))
                and witness
                and witness.get("accession") == accession
                and re.fullmatch(r"[A-Z][A-Z0-9]+", witness["standard"])
            ):
                record["gene"] = witness["standard"]
                record["gene_label_repair"] = {
                    "method": "Exact locus and accession match to the independent strain table",
                    "strain_table_row": witness["source_row"],
                    "source_gene_label": witness["standard"],
                }
        by_locus[locus.upper()].append(record)
    accepted, excluded, accessions = {}, {}, set()
    for locus, records in by_locus.items():
        reason = None
        if not PROTEIN_LOCUS.fullmatch(locus):
            reason = "not_a_single_resolved_locus"
        elif len(records) != 1:
            reason = "multiple_source_protein_records_for_locus"
        elif not ACCESSION.fullmatch(records[0]["accession"]):
            reason = "not_a_single_resolved_accession"
        elif not records[0]["gene"]:
            reason = "unresolved_gene_label"
        if reason:
            excluded[locus] = {"reason": reason, "records": records}
            continue
        record = records[0]
        require(record["accession"] not in accessions, "Accession assigned to distinct loci")
        accessions.add(record["accession"])
        accepted[locus] = record
    return accepted, excluded


def numeric(values, location):
    result = []
    for value in values:
        if value is None:
            result.append(float("nan"))
        else:
            require(type(value) in (int, float) and math.isfinite(value), f"Invalid measurement at {location}")
            result.append(float(value))
    return result


def parse_condition(header, rows, strains, proteins, excluded):
    require(tuple(header[:2]) == ("Molecule Name", "Molecule Type"), "Unexpected quantitative header")
    genotypes, column_sources = [], []
    for column, label in enumerate(header[2:], start=3):
        require(isinstance(label, str) and label.startswith("∆") and label[1:] in strains, "Unresolved deletion column")
        genotypes.append(strains[label[1:]])
        column_sources.append({"column": column, "source_label": label, "genotype": strains[label[1:]]})
    require(genotypes and len(genotypes) == len(set(genotypes)), "Repeated canonical deletion columns")
    features, targets, omitted = {}, {}, []
    for excel_row, row in enumerate(rows, start=2):
        if not any(x is not None for x in row):
            continue
        require(len(row) == len(header), "Quantitative row width differs")
        name, kind = row[:2]
        if kind == "Metabolite" and name in FEATURES:
            require(name not in features, "Repeated requested metabolite")
            features[name] = {"row": excel_row, "values": numeric(row[2:], f"metabolite row {excel_row}")}
        elif kind == "Protein":
            match = re.fullmatch(r"(.+) \(([^()]*)\)", str(name))
            require(match is not None, "Unparseable protein display identity")
            gene, locus = match[1], match[2].upper()
            if locus in excluded:
                omitted.append({"row": excel_row, "name": name, "locus": locus, "reason": excluded[locus]["reason"]})
                continue
            require(
                locus in proteins and gene == proteins[locus]["gene"], "Protein label does not match source metadata"
            )
            accession = proteins[locus]["accession"]
            require(accession not in targets, "Repeated resolved protein target")
            targets[accession] = {
                "row": excel_row,
                "name": name,
                "locus": locus,
                "values": numeric(row[2:], f"protein row {excel_row}"),
            }
        else:
            require(kind in {"Metabolite", "Lipid"}, "Unexpected molecule type")
    require(set(features) == set(FEATURES), "Required amino-acid measurement is missing")
    require(targets, "No resolved protein targets")
    protein_ids = sorted(targets)
    x = np.asarray([features[name]["values"] for name in FEATURES], dtype=np.float64).T
    y = np.asarray([targets[name]["values"] for name in protein_ids], dtype=np.float64).T
    witness = {
        "columns": column_sources,
        "features": {name: {"row": features[name]["row"], "unit": UNIT} for name in FEATURES},
        "proteins": {
            name: {key: value for key, value in targets[name].items() if key != "values"} for name in protein_ids
        },
        "excluded_protein_rows": omitted,
    }
    return genotypes, protein_ids, x, y, witness


def prepare(source, output):
    import openpyxl

    output = Path(output)
    require(
        output.is_absolute()
        and ".." not in output.parts
        and not output.exists()
        and not any(p.is_symlink() for p in (output, *output.parents)),
        "New absolute nonsymlink output required",
    )
    paths = validate_source(source)
    workbooks = {
        name: openpyxl.load_workbook(path, read_only=True, data_only=False, keep_links=False)
        for name, path in paths.items()
    }
    try:
        strains, strain_records = strain_index(
            workbooks["strains"]["KO Strains"].iter_rows(min_row=2, values_only=True)
        )
        proteins, excluded = protein_index(
            workbooks["biomolecules"]["Molecule_Table"].iter_rows(min_row=2, values_only=True), strain_records
        )
        matrices = {}
        for condition, sheet in SHEETS.items():
            iterator = workbooks["quantitative"][sheet].iter_rows(values_only=True)
            matrices[condition] = parse_condition(next(iterator), iterator, strains, proteins, excluded)
        require(
            set(matrices["respiration"][0]) == set(matrices["fermentation"][0]) == set(strain_records),
            "Conditions do not cover the same declared deletion genotypes",
        )
        validate_source(source)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "source.json", source)
        atomic_json(output / "strain_metadata.json", strain_records)
        atomic_json(output / "excluded_proteins.json", excluded)
        atomic_json(output / "protein_metadata.json", proteins)
        summaries = {}
        for condition, (genes, ids, x, y, witness) in matrices.items():
            folder = output / condition
            folder.mkdir()
            np.save(folder / "features.npy", x, allow_pickle=False)
            np.save(folder / "targets.npy", y, allow_pickle=False)
            atomic_json(folder / "genotypes.json", genes)
            atomic_json(folder / "proteins.json", ids)
            atomic_json(folder / "row_sources.json", witness)
            complete = np.isfinite(x).all(axis=1)
            summaries[condition] = {
                "strains": len(genes),
                "resolved_proteins": len(ids),
                "feature_count": len(FEATURES),
                "complete_feature_strains": int(complete.sum()),
                "finite_targets": int(np.isfinite(y).sum()),
                "proteins_with_60_complete_pairs": int((np.isfinite(y[complete]).sum(axis=0) >= 60).sum()),
                "excluded_protein_rows": len(witness["excluded_protein_rows"]),
            }
        report = {
            "schema": SCHEMA,
            "status": "source_reconstructed",
            "source": source,
            "conditions": summaries,
            "covariates": [{"name": name, "unit": UNIT} for name in FEATURES],
            "target": {"name": "protein_abundance_log2_fold_change", "unit": UNIT},
            "observation_unit": "One published triplicate summary per deletion genotype and growth condition",
            "preprocessing": "Preserve published log2 changes relative to matched WT; the authors already imputed and filtered measurements. No additional imputation, averaging or normalization.",
            "limitations": [
                "Retrospective processed summaries, not individual cultures or raw intensities",
                "Batch and matched-WT identities are not provided by these quantitative sheets",
                "Arginine is absent; cysteine is included and tyrosine uses the explicitly named 3TMS measurement",
                "No model-task partitions, task admission, training or publication performed",
            ],
            "acquisition_admitted": False,
            "models_trained": 0,
            "external_model_calls": 0,
        }
        report["files_sha256"] = {
            p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()
        }
        atomic_json(output / "manifest.json", report)
        return report
    finally:
        for workbook in workbooks.values():
            workbook.close()
