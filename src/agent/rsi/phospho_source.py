"""Reconstruct observed yeast phosphosite responses without cohort imputation.

Each run is centered by its own observed-site median. This defines a relative
log-intensity endpoint; it is neither phosphorylation occupancy nor activity.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .storage import atomic_json, sha256_file
from .tmap_source import bound, require

SCHEMA = "yeast_phospho_source.v1"
PAPER = {
    "title": "The regulatory landscape of the yeast phosphoproteome",
    "doi": "10.1038/s41594-023-01115-3",
}
RECORD = "10.5281/zenodo.10016524"
SOURCE_HASHES = {
    "data_psites.csv": "e3458c7e29104769861577524f009835a7f175b2a764d2e806ea038dbbf61976",
    "meta_conditions.csv": "c03da0bf268738de815904133ea2e2a4c180d3af9474eca8cca8468ffae9feb2",
    "meta_runs_samples.csv": "0e7ed7ea8363995604d3d150b8c6af9fba7f9f61787ea68b3a605f1715a3a34d",
}
TARGET = "run_centered_observed_phosphosite_log_intensity"
UNIT = "source_log_intensity_relative_to_run_median"
SITE = re.compile(r"(Y[A-P][LR][0-9]{3}[CW](?:-[A-Z])?)_([STY])_([1-9][0-9]*)")
SAMPLE = re.compile(r"([A-Z][A-Z0-9]*)\.([0-9]+)_([1-6])")
BATCHES = {"2", "3", "5", "6", "7", "8", "9"}
MINIMUM_ANCHORS = 100


def validate_source(source):
    require(
        isinstance(source, dict) and set(source) == {"schema", "paper", "record_doi", "files", "references"},
        "Invalid phosphosite source declaration",
    )
    require(source["schema"] == SCHEMA and source["paper"] == PAPER, "Unexpected phosphosite study")
    require(source["record_doi"] == RECORD, "Unexpected author archive")
    require(isinstance(source["files"], dict) and set(source["files"]) == set(SOURCE_HASHES), "Missing source files")
    paths = {}
    for name, expected in SOURCE_HASHES.items():
        item = source["files"][name]
        path = bound(item, 230000000)
        require(path.name == name and item["sha256"] == expected, "Expected pinned nonimputed author file")
        paths[name] = path
    require(
        isinstance(source["references"], list) and 1 <= len(source["references"]) <= 12, "Acquisition receipt required"
    )
    for item in source["references"]:
        bound(item, 4 * 1024**2)
    return paths


def read_rows(path, header, encoding="utf-8"):
    with Path(path).open(newline="", encoding=encoding) as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == header, "Unexpected source header")
        for number, row in enumerate(reader, 2):
            require(None not in row and all(value is not None for value in row.values()), "Source row width differs")
            yield number, row


def metadata(condition_rows, run_rows):
    conditions = {}
    for number, row in condition_rows:
        code = row["cond"]
        require(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", code), "Invalid condition code")
        record = {key: row[key] for key in ("description", "type", "time")}
        require(
            record["description"] and record["type"] and (record["time"] or record["type"] == "other"),
            "Missing condition annotation",
        )
        if code in conditions:
            require(conditions[code]["annotation"] == record, "Conflicting condition annotations")
            conditions[code]["source_rows"].append(number)
        else:
            conditions[code] = {"annotation": record, "source_rows": [number]}
    samples, excluded, runs = {}, [], {}
    for number, row in run_rows:
        run, sample = row["run"], row["sample"]
        require(re.fullmatch(r"x[0-9]+", run) and sample and run not in runs, "Duplicate or malformed run")
        require(re.fullmatch(r"[0-9]+", row["batch"]) and row["block"], "Invalid batch metadata")
        runs[run] = {**row, "source_row": number}
        match = SAMPLE.fullmatch(sample)
        reason = None
        if not match:
            reason = "not_a_resolved_biological_sample"
        elif match[1] not in conditions:
            reason = "no_exact_condition_code"
        elif conditions[match[1]]["annotation"]["time"] != "5min":
            reason = "different_duration_or_background"
        elif row["batch"] not in BATCHES:
            reason = "outside_author_retained_batches"
        if reason:
            excluded.append({**row, "source_row": number, "reason": reason})
            continue
        require(sample not in samples, "Repeated biological sample would duplicate an injection")
        samples[sample] = {
            **row,
            "source_row": number,
            "condition": match[1],
            "replicate": int(match[3]),
            "condition_source_rows": conditions[match[1]]["source_rows"],
            "type": conditions[match[1]]["annotation"]["type"],
        }
    require(samples, "No resolved short-duration samples")
    return conditions, runs, samples, excluded


def parse_quantitative(rows, runs, samples):
    order = sorted(samples)
    positions = {sample: i for i, sample in enumerate(order)}
    arrays, row_numbers, excluded_sites = {}, {}, {}
    counts = Counter()
    for number, row in rows:
        counts["source_rows"] += 1
        run, sample, reference, site = (row[key] for key in ("run", "sample", "reference", "var"))
        require(run in runs and runs[run]["sample"] == sample, "Quantification and sample metadata disagree")
        if sample not in samples:
            counts["excluded_sample_rows"] += 1
            continue
        match = SITE.fullmatch(site)
        if not match or match[1] != reference:
            counts["unresolved_site_rows"] += 1
            excluded_sites.setdefault(site, {"reference": reference, "first_row": number})
            continue
        if site not in arrays:
            arrays[site] = np.full(len(order), np.nan)
            row_numbers[site] = np.zeros(len(order), dtype=np.int64)
        index = positions[sample]
        require(row_numbers[site][index] == 0, "Duplicate sample-site measurement")
        row_numbers[site][index] = number
        if row["value"] in {"", "NA", "NaN"}:
            counts["explicit_missing_measurements"] += 1
            continue
        try:
            value = float(row["value"])
        except ValueError as error:
            raise ValueError("Invalid source log intensity") from error
        require(math.isfinite(value), "Nonfinite source log intensity")
        arrays[site][index] = value
    require(arrays, "No uniquely resolved observed phosphosites")
    sites = sorted(arrays)
    values = np.asarray([arrays[site] for site in sites]).T
    witnesses = np.asarray([row_numbers[site] for site in sites]).T
    return order, sites, values, witnesses, {"counts": dict(counts), "excluded_sites": excluded_sites}


def center_runs(values):
    require(values.ndim == 2 and not np.isinf(values).any(), "Invalid log-intensity matrix")
    coverage = np.isfinite(values).sum(axis=1)
    medians = np.asarray(
        [np.median(row[np.isfinite(row)]) if count else np.nan for row, count in zip(values, coverage)]
    )
    admitted = coverage >= MINIMUM_ANCHORS
    centered = values - medians[:, None]
    centered[~admitted] = np.nan
    return centered, medians, coverage, admitted


def prepare(source, output):
    output = Path(output)
    require(output.is_absolute() and ".." not in output.parts and not output.exists(), "New absolute output required")
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Symlink output rejected")
    paths = validate_source(source)
    conditions, runs, samples, excluded = metadata(
        read_rows(
            paths["meta_conditions.csv"], ["id", "description", "type", "description 2", "cond", "time"], "latin-1"
        ),
        read_rows(paths["meta_runs_samples.csv"], ["sample", "run", "batch", "block"]),
    )
    order, sites, original, rows, witness = parse_quantitative(
        read_rows(paths["data_psites.csv"], ["sample", "run", "reference", "var", "value"]), runs, samples
    )
    centered, medians, coverage, admitted = center_runs(original)
    classes = sorted({record["type"] for record in samples.values()})
    require(1 <= len(classes) <= 64, "Unsupported number of condition types")
    features = np.asarray(
        [[int(samples[sample]["type"] == label) for label in classes] for sample in order], dtype=float
    )
    validate_source(source)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in {
        "source.json": source,
        "conditions.json": conditions,
        "samples.json": samples,
        "sites.json": sites,
        "source_witness.json": witness,
        "exclusions.json": excluded,
        "run_centering.json": {
            sample: {
                "median": float(medians[i]) if np.isfinite(medians[i]) else None,
                "observed_sites": int(coverage[i]),
                "admitted": bool(admitted[i]),
            }
            for i, sample in enumerate(order)
        },
    }.items():
        atomic_json(output / name, value)
    for name, value in {"features.npy": features, "targets.npy": centered, "source_rows.npy": rows}.items():
        np.save(output / name, value, allow_pickle=False)
    report = {
        "schema": SCHEMA,
        "status": "source_reconstructed",
        "paper": PAPER,
        "source": source,
        "sample_order": order,
        "samples": len(order),
        "admitted_samples": int(admitted.sum()),
        "condition_codes": len({record["condition"] for record in samples.values()}),
        "site_scopes_before_task_admission": len(sites),
        "distinct_proteins": len({SITE.fullmatch(site)[1] for site in sites}),
        "finite_observations": int(np.isfinite(centered).sum()),
        "minimum_centering_sites": MINIMUM_ANCHORS,
        "target": {"name": TARGET, "unit": UNIT},
        "covariates": [
            {
                "name": "condition_type_" + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_"),
                "unit": "binary_indicator",
                "minimum": 0,
                "maximum": 1,
            }
            for label in classes
        ],
        "condition_type_order": classes,
        "preprocessing": "Author nonimputed log-intensity values minus the median of observed resolved sites in the same run; no cross-run fitted transformation or imputation",
        "limitations": [
            "Relative phosphosite log intensity, not phosphorylation occupancy, protein-adjusted abundance or kinase activity",
            "Source log base is not assumed; units follow the author export",
            "Short-duration laboratory BY4741 samples in author-retained batches only",
            "Relative labels use the same sample's observed phosphosite background; background measurements are not model inputs",
            "Coarse author condition classes do not identify chemical structure or dose",
            "No trained or published models; source preparation only",
        ],
        "models_trained": 0,
        "models_published": 0,
        "external_model_calls": 0,
    }
    require(len({item["name"] for item in report["covariates"]}) == len(classes), "Condition feature name collision")
    report["files_sha256"] = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    atomic_json(output / "manifest.json", report)
    return report
