"""Source-bound reconstruction of the published yeast metabolite/proteome join.

This adapter preserves measurements and missing labels. It does not allocate
training partitions, fit predictors or admit models to the Scientist catalog.
"""

from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import atomic_json, sha256_file

SCHEMA = "protpredict_source.v1"
PAPER_DOI = "10.1093/bioinformatics/btae050"
FEATURES = (
    "alanine",
    "aspartate",
    "glutamate",
    "phenylalanine",
    "glycine",
    "histidine",
    "isoleucine",
    "lysine",
    "leucine",
    "methionine",
    "asparagine",
    "proline",
    "glutamine",
    "arginine",
    "serine",
    "threonine",
    "valine",
    "tryptophan",
    "tyrosine",
)


def bound_source(binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Source path and SHA256 required")
    if not isinstance(binding["path"], str) or re.fullmatch(r"[0-9a-f]{64}", str(binding["sha256"])) is None:
        raise ValueError("Invalid source path or SHA256")
    path = Path(binding["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Source must be an absolute regular file")
    if path.stat().st_size > 128 * 1024**2 or sha256_file(path) != binding["sha256"]:
        raise ValueError("Source size or SHA256 differs")
    return path


def validate_source(source):
    if not isinstance(source, dict) or set(source) != {
        "schema",
        "repository_commit",
        "paper",
        "proteomics",
        "metabolites",
        "references",
    }:
        raise ValueError("Invalid ProtPredict source declaration")
    if source["schema"] != SCHEMA or re.fullmatch(r"[0-9a-f]{40}", str(source["repository_commit"])) is None:
        raise ValueError("Versioned source and immutable repository commit required")
    if source["paper"] != {
        "title": "Interpreting protein abundance in Saccharomyces cerevisiae through relational learning",
        "doi": PAPER_DOI,
    }:
        raise ValueError("Unexpected source paper")
    if not isinstance(source["references"], list) or not source["references"]:
        raise ValueError("Source documentation must be bound")
    for item in source["references"]:
        bound_source(item)
    return bound_source(source["proteomics"]), bound_source(source["metabolites"])


def _identifiers(values, name):
    if any(not isinstance(x, str) or not x.strip() or x != x.strip() or len(x) > 100 for x in values):
        raise ValueError(f"Invalid exact {name} identifier")


def combine_frames(proteomics, metabolites):
    """Reproduce the author join; retain whole deletion genotypes as units."""
    if (
        not isinstance(proteomics, pd.DataFrame)
        or not isinstance(metabolites, pd.DataFrame)
        or proteomics.empty
        or metabolites.empty
        or proteomics.columns.duplicated().any()
        or list(proteomics.columns[:1]) != ["Unnamed: 0"]
        or not 2 <= len(proteomics.columns) <= 5001
        or len(proteomics) > 10000
        or list(metabolites.columns) != ["ORF", "gene", *FEATURES]
        or len(metabolites) > 10000
    ):
        raise ValueError("Unexpected source columns or matrix dimensions")
    _identifiers(proteomics["Unnamed: 0"].tolist(), "proteome genotype")
    _identifiers(metabolites["ORF"].tolist(), "metabolite genotype")
    _identifiers(list(proteomics.columns[1:]), "protein")
    if metabolites["ORF"].duplicated().any():
        raise ValueError("Duplicate metabolite genotype; ambiguous join")
    if set(proteomics.columns[1:]) & set(FEATURES):
        raise ValueError("Protein and predictor names overlap")
    protein_values = proteomics.iloc[:, 1:].apply(pd.to_numeric, errors="raise").astype(float)
    metabolite_values = metabolites.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="raise").astype(float)
    for values in (protein_values, metabolite_values):
        array = values.to_numpy()
        if np.isinf(array).any() or (array < 0).any():
            raise ValueError("Nonfinite or negative measured concentration/abundance")
    protein_values.index = proteomics["Unnamed: 0"].tolist()
    # Average duplicate deletion IDs exactly as the published preprocessing.
    means = protein_values.groupby(level=0, sort=True).mean()
    metabolite_values.index = metabolites["ORF"].tolist()
    joined = means.join(metabolite_values, how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No exact deletion-genotype matches")
    valid = np.isfinite(joined.loc[:, list(FEATURES)].to_numpy()).all(axis=1)
    selected = joined.loc[valid]
    if selected.empty:
        raise ValueError("No complete measured predictor rows")
    proteins = list(means.columns)
    x = selected.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    y = selected.loc[:, proteins].to_numpy(dtype=np.float64)
    counts = np.isfinite(y).sum(axis=0)
    if not counts.any():
        raise ValueError("No observed protein targets")
    source_rows = {}
    for row, gene in enumerate(proteomics["Unnamed: 0"], 2):
        source_rows.setdefault(gene, []).append(row)
    metabolite_rows = {gene: row for row, gene in enumerate(metabolites["ORF"], 2)}
    provenance = {
        gene: {"protein_csv_rows": source_rows[gene], "metabolite_xls_row": metabolite_rows[gene]}
        for gene in selected.index
    }
    summary = {
        "raw_protein_rows": len(proteomics),
        "distinct_proteome_genotypes": len(means),
        "raw_metabolite_rows": len(metabolites),
        "joined_genotypes": len(joined),
        "complete_covariate_genotypes": len(selected),
        "protein_columns": len(proteins),
        "covariate_columns": len(FEATURES),
        "observed_targets": int(counts.sum()),
        "protein_minimum_observations": int(counts.min()),
        "protein_median_observations": float(np.median(counts)),
        "protein_maximum_observations": int(counts.max()),
        "proteins_with_at_least_60_observations": int((counts >= 60).sum()),
        "duplicate_rows_averaged": len(proteomics) - len(means),
        "unmatched_proteome_genotypes": len(means) - len(joined),
        "incomplete_covariate_genotypes_excluded": int((~valid).sum()),
    }
    return list(selected.index), proteins, x, y, provenance, summary


def load_frames(source):
    protein_path, metabolite_path = validate_source(source)
    with zipfile.ZipFile(protein_path) as archive:
        names = archive.namelist()
        if (
            names.count("proteomics_messner.csv") != 1
            or any(name != "proteomics_messner.csv" and not name.startswith("__MACOSX/") for name in names)
            or archive.getinfo("proteomics_messner.csv").file_size > 300 * 1024**2
        ):
            raise ValueError("Unexpected or oversized proteomics archive member")
        with archive.open("proteomics_messner.csv") as stream:
            import io

            header = next(csv.reader(io.TextIOWrapper(stream, encoding="utf-8")))
            if len(header) != len(set(header)):
                raise ValueError("Duplicate CSV protein headers")
        with archive.open("proteomics_messner.csv") as stream:
            proteomics = pd.read_csv(stream)
    metabolites = pd.read_excel(metabolite_path, sheet_name="intracellular_concentration_mM", engine="xlrd")
    return proteomics, metabolites


def prepare(source, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Source output already exists")
    genotypes, proteins, x, y, provenance, summary = combine_frames(*load_frames(source))
    validate_source(source)
    output.mkdir(parents=True, exist_ok=False)
    for name, array in (("features.npy", x), ("targets.npy", y)):
        with (output / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
    atomic_json(output / "genotypes.json", genotypes)
    atomic_json(output / "proteins.json", proteins)
    atomic_json(output / "covariates.json", [{"name": name, "unit": "mM"} for name in FEATURES])
    atomic_json(output / "row_provenance.json", provenance)
    atomic_json(output / "source.json", source)
    files = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    report = {
        "schema": SCHEMA,
        "status": "source_reconstructed",
        "source": source,
        "summary": summary,
        "files_sha256": files,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target": "unchanged author-reported protein abundance; no log, imputation or rescaling",
        "grouping_unit": "deleted_gene_identity",
        "preprocessing": "within-genotype arithmetic mean of observed protein replicates, exact genotype join, complete measured covariates",
        "scope": "Reconstruction of the deposited author tables, not reprocessing of raw mass spectra",
        "acquisition_admitted": False,
        "partitions_created": False,
        "external_model_calls": 0,
        "new_models": 0,
    }
    atomic_json(output / "manifest.json", report)
    return report
