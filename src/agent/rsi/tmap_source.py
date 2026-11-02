"""Source-bound reconstruction of yeast single-time-point turnover observations.

Use unnormalized per-experiment H/L ratios, never the across-strain median or
published robust scores. Ambiguous protein groups and unresolved strain labels
remain explicit exclusions. This module does not fit or publish models.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .storage import atomic_json, sha256_file

SCHEMA = "tmap_source.v1"
PAPER = {
    "title": "A Systematic Protein Turnover Map for Decoding Protein Degradation",
    "doi": "10.1016/j.celrep.2020.108378",
}
ACCESSION = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})")
SAMPLE = re.compile(r"([A-Z][A-Z0-9]+)\.([1-9][0-9]*)")
FLAGS = ("Reverse", "Potential contaminant", "Only identified by site")
TARGET = "single_time_point_turnover_log2_heavy_light_ratio"
UNIT = "log2_H_L_ratio"


def require(value, message):
    if not value:
        raise ValueError(message)


def bound(binding, maximum=512 * 1024**2):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "Bound source path and digest required")
    require(isinstance(binding["path"], str), "Source path must be text")
    path = Path(binding["path"])
    require(path.is_absolute() and ".." not in path.parts and path.is_file(), "Absolute source file required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink source rejected")
    require(
        isinstance(binding["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]), "Invalid source digest"
    )
    require(path.stat().st_size <= maximum and sha256_file(path) == binding["sha256"], "Source size or digest differs")
    return path


def validate_source(source):
    require(
        isinstance(source, dict)
        and set(source)
        == {"schema", "paper", "protein_groups", "table_s1", "ms_summary", "references", "unavailable_samples"},
        "Invalid T-MAP source declaration",
    )
    require(source["schema"] == SCHEMA and source["paper"] == PAPER, "Unexpected T-MAP study")
    unavailable = source["unavailable_samples"]
    require(
        isinstance(unavailable, list)
        and len(unavailable) <= 32
        and all(isinstance(x, str) and x for x in unavailable)
        and len(unavailable) == len(set(unavailable)),
        "Explicit unique unavailable-sample list required",
    )
    paths = {name: bound(source[name]) for name in ("protein_groups", "table_s1", "ms_summary")}
    require(len(set(paths.values())) == 3, "Distinct source files required")
    require(
        isinstance(source["references"], list) and 1 <= len(source["references"]) <= 12,
        "Acquisition references required",
    )
    for item in source["references"]:
        bound(item, 4 * 1024**2)
    return paths


def screen_index(rows):
    index = {}
    for number, row in enumerate(rows, 2):
        if not any(value is not None for value in row):
            continue
        require(len(row) == 3, "Unexpected screen annotation width")
        _, gene, function = row
        require(isinstance(gene, str) and gene.strip() == gene and gene, "Invalid screen gene")
        require(isinstance(function, str) and function.strip() == function and function, "Invalid screen function")
        if gene in index:
            require(index[gene]["function"] == function, "Conflicting duplicate screen annotations")
            index[gene]["source_rows"].append(number)
        else:
            index[gene] = {"function": function, "source_rows": [number]}
    require(index, "Empty screen annotation table")
    return index


def protein_index(rows):
    candidates = defaultdict(list)
    excluded = []
    for number, row in enumerate(rows, 2):
        if not any(value is not None for value in row):
            continue
        require(len(row) == 5, "Unexpected protein annotation width")
        _, accession, gene, locus, name = row

        def metadata(value):
            if isinstance(value, (date, datetime)):
                return {"unresolved_original_type": type(value).__name__, "original_value": value.isoformat()}
            require(value is None or type(value) in (str, int, float, bool), "Unsupported annotation cell type")
            if type(value) is float:
                require(math.isfinite(value), "Nonfinite annotation cell")
            return value

        record = {
            "source_row": number,
            "accession": accession,
            "gene": metadata(gene),
            "locus": metadata(locus),
            "name": metadata(name),
        }
        if not isinstance(accession, str) or not ACCESSION.fullmatch(accession):
            excluded.append({**record, "reason": "unresolved_or_multiple_accessions"})
            continue
        candidates[accession].append(record)
    accepted = {}
    for accession, records in candidates.items():
        if len(records) != 1:
            excluded.extend({**record, "reason": "repeated_annotation_accession"} for record in records)
        else:
            accepted[accession] = records[0]
    return accepted, excluded


def resolve_samples(published_samples, screen, experiment_runs, unavailable_samples=()):
    accepted, excluded = {}, []
    require(set(unavailable_samples) <= set(published_samples), "Unavailable sample not found in the publication")
    require(not set(unavailable_samples) & set(experiment_runs), "Declared unavailable sample exists in the archive")
    for label in sorted(published_samples):
        require(isinstance(label, str) and label, "Invalid published sample label")
        match = SAMPLE.fullmatch(label)
        if not match or match[1] not in screen:
            excluded.append({"sample": label, "reason": "no_exact_screen_gene_match"})
            continue
        if label in unavailable_samples:
            excluded.append({"sample": label, "reason": "explicitly_unavailable_in_bound_archive"})
            continue
        require(label in experiment_runs and experiment_runs[label], "Published sample absent from acquisition summary")
        accepted[label] = {
            "gene": match[1],
            "replicate_label": match[2],
            "function": screen[match[1]]["function"],
            "screen_rows": screen[match[1]]["source_rows"],
            "raw_files": sorted(experiment_runs[label]),
        }
    require(len({item["gene"] for item in accepted.values()}) >= 4, "Insufficient exactly resolved genetic contexts")
    return accepted, excluded


def acquisition_index(path):
    result = defaultdict(list)
    seen = set()
    summaries = []
    with Path(path).open(newline="", encoding="utf-8-sig") as reader:
        rows = csv.DictReader(reader, delimiter="\t")
        require(rows.fieldnames and len(rows.fieldnames) == len(set(rows.fieldnames)), "Duplicate acquisition headers")
        require({"Raw file", "Experiment"} <= set(rows.fieldnames), "Missing acquisition sample columns")
        for row in rows:
            require(None not in row, "Acquisition row width differs")
            raw, experiment = row["Raw file"], row["Experiment"]
            if not experiment:
                require(raw, "Missing acquisition summary identity")
                summaries.append(raw)
                continue
            require(raw and experiment and raw not in seen, "Missing or duplicate acquisition identity")
            seen.add(raw)
            result[experiment].append(raw)
    require(all(name == "Total" or name in result for name in summaries), "Unresolved acquisition summary row")
    return dict(result)


def measurement(value, *, count=False):
    if value in {"", "NaN"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Malformed numeric source observation") from error
    require(math.isfinite(number) and number >= 0, "Nonfinite or negative source observation")
    if count:
        require(number.is_integer(), "Noninteger peptide/ratio count")
        return int(number)
    return number if number > 0 else None


def parse_quantitative(header, rows, samples, proteins):
    require(len(header) == len(set(header)), "Duplicate quantitative headers")
    required = {"Protein IDs", "Number of proteins", *FLAGS}
    fields = ("Ratio H/L", "Ratio H/L count", "Peptides", "Unique peptides")
    required.update(f"{field} {sample}" for sample in samples for field in fields)
    require(required <= set(header), "Missing unnormalized per-sample ratios or quality columns")
    positions = {name: i for i, name in enumerate(header)}
    sample_ids = sorted(samples)
    columns = [[positions[f"{field} {sample}"] for field in fields] for sample in sample_ids]
    values = {}
    witnesses = {}
    excluded = []
    duplicates = set()
    counts = Counter()
    for number, row in enumerate(rows, 2):
        require(len(row) == len(header), "Quantitative source row width differs")
        counts["source_protein_rows"] += 1
        accession = row[positions["Protein IDs"]]
        flags = [row[positions[name]] for name in FLAGS]
        require(all(flag in {"", "+"} for flag in flags), "Unknown MaxQuant exclusion flag")
        reason = None
        if any(flags):
            reason = "reverse_contaminant_or_site_only"
        elif not ACCESSION.fullmatch(accession) or measurement(row[positions["Number of proteins"]], count=True) != 1:
            reason = "unresolved_or_multiple_proteins"
        elif accession not in proteins:
            reason = "no_unique_published_protein_annotation"
        if reason:
            excluded.append({"row": number, "protein_ids": accession, "reason": reason})
            continue
        if accession in witnesses:
            duplicates.add(accession)
            excluded.append({"row": number, "protein_ids": accession, "reason": "duplicate_quantitative_accession"})
            continue
        witnesses[accession] = {"row": number, "annotation": proteins[accession]}
        vector = []
        for ratio_column, count_column, peptide_column, unique_column in columns:
            ratio = measurement(row[ratio_column])
            ratio_count, peptides, unique = [
                measurement(row[column], count=True) for column in (count_column, peptide_column, unique_column)
            ]
            if ratio is None:
                counts["source_nan_ratio" if row[ratio_column] == "NaN" else "missing_or_zero_ratio"] += 1
                vector.append(float("nan"))
            elif (
                ratio_count is None
                or peptides is None
                or unique is None
                or ratio_count < 2
                or peptides < 2
                or unique < 1
            ):
                counts["insufficient_ratio_or_peptide_evidence"] += 1
                vector.append(float("nan"))
            else:
                vector.append(math.log2(ratio))
        values[accession] = vector
    for accession in duplicates:
        excluded.append(
            {"row": witnesses[accession]["row"], "protein_ids": accession, "reason": "duplicate_quantitative_accession"}
        )
        del values[accession]
        del witnesses[accession]
    require(values, "No uniquely resolved quantified proteins")
    protein_ids = sorted(values)
    targets = np.asarray([values[key] for key in protein_ids], dtype=np.float64).T
    counts["retained_finite_observations"] = int(np.isfinite(targets).sum())
    return (
        sample_ids,
        protein_ids,
        targets,
        {
            "proteins": witnesses,
            "excluded_rows": excluded,
            "counts": dict(counts),
            "columns": {
                sample: {field: positions[f"{field} {sample}"] + 1 for field in fields} for sample in sample_ids
            },
        },
    )


def prepare(source, output):
    import openpyxl

    output = Path(output)
    require(output.is_absolute() and ".." not in output.parts and not output.exists(), "New absolute output required")
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Symlink output rejected")
    paths = validate_source(source)
    workbook = openpyxl.load_workbook(paths["table_s1"], read_only=True, data_only=False, keep_links=False)
    try:
        screen_rows = workbook["Screen description"].iter_rows(values_only=True)
        require(
            next(screen_rows)
            == (
                "# (number of strain)",
                "Strain (name of gene deleted in strain)",
                "Function (encoded by the deleted gene)",
            ),
            "Unexpected screen header",
        )
        screen = screen_index(screen_rows)
        protein_rows = workbook["Protein abundance"].iter_rows(values_only=True)
        require(next(protein_rows)[1:3] == ("Uniprot ID", "Gene Name"), "Unexpected protein annotation header")
        proteins, excluded_proteins = protein_index(protein_rows)
        sample_rows = workbook["Turnover robust Z-Scores"].iter_rows(min_row=2, max_col=1, values_only=True)
        published_samples = {row[0] for row in sample_rows if row[0] is not None}
    finally:
        workbook.close()
    samples, excluded_samples = resolve_samples(
        published_samples, screen, acquisition_index(paths["ms_summary"]), source["unavailable_samples"]
    )
    previous_limit = csv.field_size_limit(8 * 1024**2)
    try:
        with paths["protein_groups"].open(newline="", encoding="utf-8-sig") as reader:
            rows = csv.reader(reader, delimiter="\t")
            sample_ids, protein_ids, targets, witness = parse_quantitative(next(rows), rows, samples, proteins)
    finally:
        csv.field_size_limit(previous_limit)
    classes = sorted({record["function"] for record in screen.values()})
    features = np.asarray(
        [[int(samples[sample]["function"] == category) for category in classes] for sample in sample_ids],
        dtype=np.float64,
    )
    validate_source(source)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "source.json", source)
    atomic_json(output / "samples.json", {name: samples[name] for name in sample_ids})
    atomic_json(output / "proteins.json", protein_ids)
    atomic_json(output / "source_witness.json", witness)
    atomic_json(output / "exclusions.json", {"samples": excluded_samples, "protein_annotations": excluded_proteins})
    np.save(output / "features.npy", features, allow_pickle=False)
    np.save(output / "targets.npy", targets, allow_pickle=False)
    report = {
        "schema": SCHEMA,
        "status": "source_reconstructed",
        "paper": PAPER,
        "source": source,
        "sample_order": sample_ids,
        "covariates": [
            {
                "name": "perturbed_gene_function_" + value.replace(" ", "_"),
                "unit": "binary_indicator",
                "minimum": 0,
                "maximum": 1,
            }
            for value in classes
        ],
        "target": {"name": TARGET, "unit": UNIT},
        "samples": len(samples),
        "independent_genes": len({item["gene"] for item in samples.values()}),
        "resolved_proteins": len(protein_ids),
        "finite_observations": int(np.isfinite(targets).sum()),
        "excluded_samples": len(excluded_samples),
        "preprocessing": "log2 of positive per-experiment unnormalized H/L ratios; at least two ratio counts, two peptides and one unique peptide; source NaN, blank and zero ratios remain unavailable; no imputation or across-strain normalization",
        "limitations": [
            "Fixed-time isotope-ratio endpoint, not an inferred half-life or intrinsic degradation constant",
            "Exact screen-gene matches only; unresolved prefixes and allele suffixes retained as exclusions",
            "Function categories are coarse prior metadata, not measured molecular-state covariates",
            "MaxQuant matches identifications between runs; reconstructed values remain processed measurements",
            "No task admission, model fitting or publication performed",
        ],
        "acquisition_admitted": False,
        "models_trained": 0,
        "external_model_calls": 0,
    }
    report["files_sha256"] = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    atomic_json(output / "manifest.json", report)
    return report
