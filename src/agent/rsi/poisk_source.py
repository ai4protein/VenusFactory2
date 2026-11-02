"""Conservative, provenance-preserving POISK v1.1 antibody source adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .storage import atomic_json, sha256_file

DATA_SHA256 = "3ef1d109fe4f0a6af3c3f423eac72e5ee840d62ec817aecc7769e114b144f201"
SCHEMA_SHA256 = "a6fd2cf8f3d38e0c5edf9c361b58125fac8d68a803be735a82a4565d8636e2c8"
RECORD = "10.5281/zenodo.21775623"
TARGETS = {
    "P02768": "ALB",
    "P00533": "EGFR",
    "P04626": "ERBB2",
    "P08581": "MET",
    "Q16790": "CA9",
    "Q13421": "MSLN",
    "P16422": "EPCAM",
    "P22223": "CDH3",
    "P15941": "MUC1",
    "P15692": "VEGFA",
    "P61626": "LYZ",
}
AA = "ACDEFGHIKLMNPQRSTVWY"
NUMBER = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
QUANTITY = re.compile(
    rf"(?:(KD|Kd|K_D|IC50|EC50)\s*[:=]?\s*)?(<=|>=|<|>|=)?\s*({NUMBER})"
    r"(?:\s*[x*]\s*10\s*\^\s*([+-]?[0-9]+))?\s*(fM|pM|nM|uM|mM|M)"
)
SCALES = {"fM": 1e-15, "pM": 1e-12, "nM": 1e-9, "uM": 1e-6, "mM": 1e-3, "M": 1.0}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_quantity(value):
    """No endpoint inference from assay name; bounds remain bounds."""
    text = unicodedata.normalize("NFKC", value).strip().replace("μ", "u").replace("µ", "u")
    text = text.replace("−", "-").replace("×", "x").replace("≤", "<=").replace("≥", ">=")
    match = QUANTITY.fullmatch(text)
    if not match:
        return {"status": "unresolved_quantity", "raw": value}
    prefix, comparator, number, exponent, unit = match.groups()
    try:
        molar = float(number) * (10.0 ** int(exponent) if exponent else 1.0) * SCALES[unit]
    except (ValueError, OverflowError):
        return {"status": "invalid_concentration", "raw": value}
    if not math.isfinite(molar) or not 1e-18 <= molar <= 1:
        return {"status": "invalid_concentration", "raw": value}
    endpoint = "KD" if prefix in {"KD", "Kd", "K_D"} else prefix or "reported_molar_concentration"
    return {
        "status": "point" if comparator in {None, "="} else "censored",
        "endpoint": endpoint,
        "comparator": comparator or "=",
        "molar": molar,
        "negative_log10_molar": -math.log10(molar),
        "raw": value,
    }


def assay_method(value):
    if re.fullmatch(r"(?:SPR|Biacore)(?:\s*\([^()]+\))?", value.strip(), re.I):
        return "SPR"
    if re.fullmatch(r"(?:BLI|Octet)(?:\s*\([^()]+\))?", value.strip(), re.I):
        return "BLI"
    return None


def references(folder):
    result = {}
    for accession, gene in TARGETS.items():
        path = Path(folder) / f"{accession}.json"
        entry = json.loads(path.read_text())
        require(
            entry["primaryAccession"] == accession and entry["organism"]["taxonId"] == 9606, "Reference taxon differs"
        )
        require(gene in {g.get("geneName", {}).get("value") for g in entry.get("genes", [])}, "Reference gene differs")
        result[accession] = {
            "gene": gene,
            "sequence": entry["sequence"]["value"],
            "sha256": sha256_file(path),
            "path": str(path.resolve()),
        }
    return result


def normalize_row(row, refs):
    accession = row["target.uniprot_id"]
    if accession not in refs or row["target.organism"].strip().lower() not in {"human", "homo sapiens"}:
        return None, "outside_declared_human_target_panel"
    if row["audit.verdict"] != "CONFIRMED" or row["audit.flags"].strip():
        return None, "source_audit_not_confirmed_without_flags"
    kind = row["antibody_type"]
    if kind not in {"Conventional", "VHH"}:
        return None, "unsupported_complex_antibody_format"
    heavy, light = row["vh.sequence_from_patent"], row["vl.sequence_from_patent"]
    if row["vh.is_full_heavy_chain"] != "False" or not 80 <= len(heavy) <= 180 or set(heavy) - set(AA):
        return None, "unresolved_variable_heavy_chain"
    if kind == "VHH":
        if light:
            return None, "vhh_has_light_chain"
    elif row["vl.is_full_light_chain"] != "False" or not 70 <= len(light) <= 160 or set(light) - set(AA):
        return None, "unresolved_variable_light_chain"
    target = row["target.sequence"]
    if len(target) < 20 or target not in refs[accession]["sequence"] or set(target) - set(AA):
        return None, "target_reference_sequence_unresolved"
    method = assay_method(row["target.binding_affinity.method"])
    if not method:
        return None, "unsupported_or_mixed_measurement_method"
    quantity = parse_quantity(row["target.binding_affinity.affinity_value"])
    if quantity["status"] != "point":
        return None, quantity["status"]
    require(re.fullmatch(r"[0-9]+", row["patent_id"]) is not None, "Unexpected US patent identifier")
    return {
        "source_record": row["unique_record_id"],
        "patent": row["patent_id"],
        "antibody_type": kind,
        "heavy": heavy,
        "light": light,
        "antibody": digest([heavy, light]),
        "target_accession": accession,
        "target_gene": refs[accession]["gene"],
        "target_sequence_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "method": method,
        "method_raw": row["target.binding_affinity.method"],
        "condition": " ".join(row["target.binding_conditions"].split()),
        "endpoint": quantity["endpoint"],
        "molar": quantity["molar"],
        "target": quantity["negative_log10_molar"],
        "quantity_raw": quantity["raw"],
    }, None


def deduplicate(rows):
    buckets = defaultdict(list)
    for row in rows:
        key = tuple(
            row[k]
            for k in [
                "antibody",
                "antibody_type",
                "target_accession",
                "target_sequence_sha256",
                "method",
                "condition",
                "endpoint",
            ]
        )
        buckets[key].append(row)
    retained, conflicts = [], []
    for key, copies in sorted(buckets.items()):
        values = {format(row["molar"], ".12g") for row in copies}
        if len(values) != 1:
            conflicts.append(
                {
                    "key_sha256": digest(key),
                    "source_records": sorted(r["source_record"] for r in copies),
                    "reason": "conflicting_values_with_indistinguishable_recorded_context",
                }
            )
            continue
        row = dict(min(copies, key=lambda r: r["source_record"]))
        row["id"] = "poisk_" + digest([key, next(iter(values))])[:24]
        row["source_records"] = sorted(r["source_record"] for r in copies)
        row["patents"] = sorted({r["patent"] for r in copies})
        retained.append(row)
    return sorted(retained, key=lambda r: r["id"]), conflicts


def prepare(raw, reference_folder, output):
    raw, output = Path(raw), Path(output)
    require(output.is_absolute() and not output.exists(), "New absolute output required")
    require(not any(p.is_symlink() for p in [output, *output.parents]), "Symlink output rejected")
    require(sha256_file(raw / "poisk_dataset.csv") == DATA_SHA256, "POISK release checksum differs")
    require(sha256_file(raw / "poisk_dataset_column_description.csv") == SCHEMA_SHA256, "POISK schema differs")
    metadata = json.loads((raw / "poisk_dataset_croissant.json").read_text())
    require(
        metadata["version"] == "1.1" and metadata["license"] == "https://creativecommons.org/licenses/by/4.0/",
        "Release metadata differs",
    )
    refs = references(reference_folder)
    with (raw / "poisk_dataset_column_description.csv").open(newline="") as stream:
        columns = [r["column_name"] for r in csv.DictReader(stream)]
    admitted, excluded, seen = [], [], set()
    with (raw / "poisk_dataset.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == columns, "Column order or schema changed")
        for row in reader:
            require(None not in row and None not in row.values(), "Malformed CSV row")
            identifier = row["unique_record_id"]
            require(identifier not in seen, "Duplicate record ID")
            seen.add(identifier)
            parsed, reason = normalize_row(row, refs)
            if reason:
                excluded.append({"record": identifier, "reason": reason})
            else:
                admitted.append(parsed)
    rows, conflicts = deduplicate(admitted)
    output.mkdir(parents=True)
    atomic_json(output / "observations.json", rows)
    atomic_json(output / "exclusions.json", excluded)
    atomic_json(output / "conflicts.json", conflicts)
    report = {
        "schema": "poisk_source.v1",
        "record_doi": RECORD,
        "version": "1.1",
        "license": metadata["license"],
        "attribution": "Daria Tupikina, Andrea Roncoli, Alexander Bujotzek, Brennan Abanades; POISK v1.1",
        "transformations": "Explicit numeric parsing, declared human panel, source-audit filtering, reference mapping, conservative deduplication and conflict removal",
        "raw_sha256": DATA_SHA256,
        "raw_rows": len(seen),
        "point_records_before_deduplication": len(admitted),
        "retained_observations": len(rows),
        "conflicting_keys": len(conflicts),
        "exclusion_counts": dict(Counter(r["reason"] for r in excluded)),
        "endpoint_counts": dict(Counter(r["endpoint"] for r in rows)),
        "target_counts": dict(Counter(r["target_gene"] for r in rows)),
        "antibody_types": dict(Counter(r["antibody_type"] for r in rows)),
        "reference_bindings": {k: {a: b for a, b in v.items() if a != "sequence"} for k, v in refs.items()},
        "observations_sha256": sha256_file(output / "observations.json"),
        "external_model_calls": 0,
        "models_published": 0,
        "limitations": [
            "CONFIRMED is the source LLM auditor verdict, not an independent experimental verification.",
            "Unprefixed molar values remain reported_molar_concentration; they are not relabeled KD.",
            "Censored, interval, qualitative, mass-concentration and rate measurements are not point labels.",
            "A UniProt match establishes sequence identity, not the exact experimental construct.",
            "Absent experimental conditions remain unreported; they are not assumed equivalent.",
        ],
    }
    atomic_json(output / "manifest.json", report)
    return report
