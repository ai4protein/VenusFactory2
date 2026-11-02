"""Label-free, fixed descriptors for observed wildtype plant enzyme assays.

Categorical hashes encode source identities/text, not molecular fingerprints.
No vocabulary, standardization or parameters are fitted on this dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

NAME = "plant_assay_descriptors.v1"
AA = "ACDEFGHIKLMNPQRSTVWY"
DIMENSIONS = 205


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def bucket(namespace, value, size):
    return int.from_bytes(hashlib.sha256(f"{namespace}:{value}".encode()).digest()[:8], "big") % size


def indicators(namespace, values, size):
    result = [0.0] * size
    for value in values:
        result[bucket(namespace, value, size)] = 1.0
    return result


def interval(value, *, temperature=False):
    """Return lower, upper, missing and interval indicators without midpoint imputation."""
    if value is None or value == {}:
        return [0.0, 0.0, 1.0, 0.0]
    require(isinstance(value, dict), "Condition must be an explicit source interval")
    low, high = value.get("start_value"), value.get("end_value")
    require(finite(low), "Unresolved condition value")
    high = low if high is None else high
    require(finite(high) and low <= high, "Unresolved or reversed condition range")
    if temperature:
        require(value.get("unit") == "°C", "Source temperature must be in degrees Celsius")
        require(-100 <= low <= high <= 200, "Temperature outside declared domain")
    else:
        require(0 <= low <= high <= 14, "pH outside declared domain")
    return [float(low), float(high), 0.0, float(low != high)]


def substrate_identity(row):
    associated = row["associated_substrate"]
    if row["property"] == "kcat":
        require(associated is None, "Reaction-level kcat must not acquire an invented associated substrate")
        require(type(row["reaction"].get("id")) is int, "Source reaction identity required")
        return f"sabio_reaction:{row['reaction']['id']}"
    require(row["property"] in {"Km", "kcat/Km"}, "Unsupported kinetic endpoint")
    require(
        isinstance(associated, dict) and type(associated.get("id")) is int and associated["id"] > 0,
        "Measured substrate identity required",
    )
    return f"sabio_compound:{associated['id']}"


def encode(row, sequence):
    require(
        isinstance(sequence, str) and 1 <= len(sequence) <= 4096 and not set(sequence) - set(AA),
        "Full canonical reference sequence required",
    )
    counts = Counter(sequence)
    result = [counts[a] / len(sequence) for a in AA]
    pairs = [0.0] * 64
    for a, b in zip(sequence, sequence[1:]):
        pairs[bucket("dipeptide", a + b, 64)] += 1.0
    result += [value / max(1, len(sequence) - 1) for value in pairs] + [float(len(sequence))]
    ec = row["ec_number"]
    require(isinstance(ec, str) and re.fullmatch(r"[1-7]\.\d+\.\d+\.\d+", ec), "Complete source EC number required")
    segments = ec.split(".")
    result += indicators("ec", [".".join(segments[:i]) for i in range(1, 5)], 32)
    identity = substrate_identity(row)
    associated = [] if row["property"] == "kcat" else [identity]
    result += indicators("associated_substrate", associated, 24)
    substrates = [
        part["compound"]["id"]
        for part in row["reaction"]["species"]
        if part.get("role") == "Substrate"
        and type(part.get("compound", {}).get("id")) is int
        and part["compound"]["id"] > 0
    ]
    require(substrates, "Reaction substrate participants required")
    if associated:
        require(row["associated_substrate"]["id"] in substrates, "Associated compound is not a source substrate")
    result += indicators("reaction_substrate", sorted(set(substrates)), 24)
    conditions = row["conditions"]
    require(isinstance(conditions, dict), "Recorded conditions required")
    result += interval(conditions.get("envvar_ph"))
    result += interval(conditions.get("envvar_temperature"), temperature=True)
    # Deliberately exclude parameter comments, publications, accessions and every target field.
    text = json.dumps(
        {"buffer": conditions.get("buffer"), "other": conditions.get("envvars_other")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).casefold()
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
    result += indicators("condition_token", tokens, 32)
    require(len(result) == DIMENSIONS and all(finite(x) for x in result), "Invalid encoded assay features")
    return result


def covariates():
    def column(name, unit="fraction", low=0.0, high=1.0):
        return {"name": name, "unit": unit, "minimum": low, "maximum": high}

    result = [column(f"aa_{a}") for a in AA]
    result += [column(f"dipeptide_hash_{i:02d}") for i in range(64)]
    result += [column("reference_length", "amino_acid_residues", 1, 4096)]
    for family, size in [("ec_path", 32), ("associated_substrate", 24), ("reaction_substrate", 24)]:
        result += [column(f"{family}_hash_{i:02d}", "binary_indicator") for i in range(size)]
    for name, unit, low, high in [("ph", "pH", 0, 14), ("temperature", "degree_Celsius", -100, 200)]:
        result += [column(f"{name}_{bound}", unit, low, high) for bound in ["lower", "upper"]]
        result += [column(f"{name}_{flag}", "binary_indicator") for flag in ["missing", "is_range"]]
    result += [column(f"condition_token_hash_{i:02d}", "binary_indicator") for i in range(32)]
    require(len(result) == DIMENSIONS, "Descriptor declaration differs from encoder")
    return result
