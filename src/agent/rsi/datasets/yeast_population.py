"""Freeze a source-bound population split before reading yeast protein outcomes."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from .model_tasks import ordinary
from agent.rsi.engine.storage import atomic_json, sha256_file

VERSION = "yeast_food_population_source.v1"
DOI = "10.1073/pnas.2319211121"
SUPPLEMENT_SHA256 = "1106127d03bed1849ccbf314e75610e8d7fca6a35631abf61a1203070f64c50f"
SAMPLE_TABLE = "pnas.2319211121.sd01.xlsx"
FOOD_ORIGINS = frozenset(
    {
        "Bakery",
        "Wine",
        "Distillery",
        "Fermentation",
        "Dairy",
        "Beer",
        "Palm wine",
        "Sake",
        "Cider",
        "Bioethanol",
        "Flor",
    }
)
SPLIT_SEED = "yeast-food-population-v1"


def canonical_population(value):
    """Keep author population labels; wine subclades share a single partition."""
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if any(word in value.casefold() for word in ("mosaic", "mixed", "unknown")):
        return None
    if re.fullmatch(r"1\. Wine/European(?: \(subclade [1-4]\))?", value):
        return "1. Wine/European"
    if re.fullmatch(r"[0-9]+\. [A-Za-z][A-Za-z /-]+", value):
        return value
    return None


def canonical_sample(value):
    """Normalize only the documented SACE dot/underscore alias."""
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"[A-Z]{3}", value):
        return value
    if re.fullmatch(r"SACE[._][A-Z]{3}", value):
        return value.replace(".", "_")
    return None


def numeric_metadata(value, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError("Boolean is not quantitative metadata")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Missing numerical metadata") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError("Numerical metadata outside declared domain")
    return number


def population_plan(records):
    """Only sample metadata enters selection; no protein or RNA targets are read."""
    samples, excluded, seen = [], Counter(), set()
    for row in records:
        identifier = canonical_sample(row.get("Standardized_name"))
        if identifier is None:
            excluded["invalid_sample_identity"] += 1
            continue
        if identifier in seen:
            raise ValueError("Repeated standardized sample identity")
        seen.add(identifier)
        if row.get("Eco_Origin") not in FOOD_ORIGINS:
            excluded["outside_food_fermentation_origins"] += 1
            continue
        if row.get("in_Proteomic") != "Yes" or row.get("in_Transcriptomic") != "Yes":
            excluded["unpaired_measurements"] += 1
            continue
        population = canonical_population(row.get("Clades"))
        if population is None:
            excluded["mixed_mosaic_or_unassigned_population"] += 1
            continue
        try:
            od_rna = numeric_metadata(row.get("OD_RNA_seq"), 0.0, 10.0)
            od_protein = numeric_metadata(row.get("OD_proteomic"), 0.0, 10.0)
            ploidy = numeric_metadata(row.get("Ploidy"), 1.0, 8.0)
            if ploidy != int(ploidy) or min(od_rna, od_protein) <= 0:
                raise ValueError("Positive optical densities and integral ploidy required")
        except ValueError:
            excluded["incomplete_or_invalid_covariates"] += 1
            continue
        samples.append(
            {
                "id": identifier,
                "origin": row["Eco_Origin"],
                "population": population,
                "od_rna": od_rna,
                "od_protein": od_protein,
                "ploidy": ploidy,
            }
        )
    counts = Counter(s["population"] for s in samples)
    eligible = {name for name, count in counts.items() if count >= 10}
    retained = [s for s in samples if s["population"] in eligible]
    excluded["population_below_metadata_sample_floor"] = len(samples) - len(retained)
    if len(eligible) < 6:
        raise ValueError("Need at least six eligible populations for 4/1/1 independent groups")
    ordered = sorted(eligible, key=lambda name: (hashlib.sha256((SPLIT_SEED + ":" + name).encode()).hexdigest(), name))
    partition = dict.fromkeys(ordered, "training")
    partition[ordered[0]] = "validation"
    partition[ordered[1]] = "future"
    for sample in retained:
        sample["phase"] = partition[sample["population"]]
    retained.sort(key=lambda item: item["id"])
    rows = dict(Counter(s["phase"] for s in retained))
    if any(rows.get(p, 0) < n for p, n in {"training": 40, "validation": 10, "future": 10}.items()):
        raise ValueError("Insufficient metadata rows under fixed population split")
    return {
        "schema": VERSION,
        "study_doi": DOI,
        "samples": retained,
        "partition": dict(sorted(partition.items())),
        "rows": rows,
        "population_counts": {name: counts[name] for name in sorted(eligible)},
        "excluded": dict(sorted(excluded.items())),
        "split_seed": SPLIT_SEED,
        "grouping_level": "genetic_population",
        "new_trained_models": 0,
    }


def target_protocol():
    return {
        "schema": VERSION,
        "study_doi": DOI,
        "organism": "Saccharomyces cerevisiae",
        "food_origins": sorted(FOOD_ORIGINS),
        "sample_identity": "SACE.XXX in PRIDE run metadata is matched to SACE_XXX in author sample/RNA tables; no suffix-only merging",
        "selection": "Metadata only; >=10 paired samples per population; canonical wine population; no mixed/mosaic labels",
        "partition": "SHA256(seed:population): first validation, second future, remaining training; fixed for every protein",
        "grouping": "Author genetic population; not biological replication or verified whole-genome independence",
        "raw_signal": "Precursor.Quantity only; no Precursor.Normalised, globally imputed or quantile-normalized targets",
        "quality_fields": [
            "Q.Value",
            "PG.Q.Value",
            "Protein.Q.Value",
            "Global.Q.Value",
            "Global.PG.Q.Value",
            "Lib.PG.Q.Value",
            "Lib.Q.Value",
        ],
        "maximum_q_value": 0.01,
        "precursor_mapping": "Proteotypic precursors mapping unambiguously to one annotated yeast gene",
        "qc_panel": "Positive signal in >=80% of QC injections in each batch; linear sample CV <=0.3 in each batch",
        "qc_denominator": "Median positive raw precursor quantity among qualifying QC injections in the same batch",
        "target": "Median log2(raw precursor / same-batch QC median); >=2 distinct stripped peptides per protein/run",
        "technical_replicates": "Median quantified run-level target for the same strain; one row per strain/protein",
        "missing_values": "No imputation; missing training, validation and future coverage reported separately",
        "covariates": ["author_log2_tpm", "od_rna", "od_protein", "ploidy"],
        "feature_domains": [[-30.0, 40.0], [0.0, 10.0], [0.0, 10.0], [1.0, 8.0]],
        "published_endpoint_reproduced": False,
        "source_limitation": "Pooled QC comprises all isolates; the shared spectral library and MBR also span partitions",
        "interpretation": "Retrospective population-held-out prediction of a QC-relative assay endpoint, not prospective validation",
        "model_selection": "Training data only; never change fixed holdout populations for sparse or difficult tasks",
        "future_evaluation": "Once after the model and comparison protocol have been fixed",
        "model_calls": 0,
    }


def freeze_source(raw_root, output):
    """Verify the published supplement and freeze only its sample metadata."""
    import openpyxl

    raw_root, output = Path(raw_root).absolute(), Path(output).absolute()
    archive = ordinary(raw_root / "supplementary.zip")
    if sha256_file(archive) != SUPPLEMENT_SHA256:
        raise ValueError("Published supplement digest differs")
    table = ordinary(raw_root / SAMPLE_TABLE)
    with ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if Path(name).name == SAMPLE_TABLE]
        if len(members) != 1 or bundle.read(members[0]) != table.read_bytes():
            raise ValueError("Sample metadata differs from source supplement")
    workbook = openpyxl.load_workbook(table, read_only=True, data_only=True)
    try:
        rows = workbook.active.iter_rows(values_only=True)
        header = next(rows)
        plan = population_plan(dict(zip(header, values, strict=True)) for values in rows)
    finally:
        workbook.close()
    output.mkdir(parents=True, exist_ok=False)
    plan["metadata_source"] = {"path": str(table), "sha256": sha256_file(table)}
    plan["supplement_source"] = {"path": str(archive), "sha256": SUPPLEMENT_SHA256}
    atomic_json(output / "population_plan.json", plan)
    atomic_json(output / "target_protocol.json", target_protocol())
    return {
        "rows": plan["rows"],
        "populations": len(plan["partition"]),
        "metadata_plan_sha256": sha256_file(output / "population_plan.json"),
        "target_protocol_sha256": sha256_file(output / "target_protocol.json"),
        "new_trained_models": 0,
        "external_model_calls": 0,
    }


def validate_report_header(columns):
    """Require raw quantities and the actual DIA-NN quality fields, without fallback."""
    required = {
        "File.Name",
        "Run",
        "Genes",
        "Proteotypic",
        "Precursor.Id",
        "Stripped.Sequence",
        "Precursor.Quantity",
        *target_protocol()["quality_fields"],
    }
    if len(columns) != len(set(columns)) or not required <= set(columns):
        raise ValueError("Raw report lacks unique required signal or quality fields")
    return sorted(required)
