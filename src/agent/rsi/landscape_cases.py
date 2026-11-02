"""Disjoint sequence-only partitions of released GB1 and TrpB landscapes."""

from __future__ import annotations

import csv
import json
import math

from .protein_model import digest, validate_rows
from .storage import sha256_file


def partition(rows, protein, sizes=(256, 512, 1024), seed=20261011, *, exclude_sequences=()):
    if len(sizes) != 3 or any(n <= 0 for n in sizes) or sum(sizes) > len(rows):
        raise ValueError("invalid partition sizes")
    if len({r["sequence"] for r in rows}) != len(rows):
        raise ValueError("duplicate sequence")
    if any(not math.isfinite(float(r["target"])) for r in rows):
        raise ValueError("nonfinite label")
    excluded = set(exclude_sequences)
    eligible = [row for row in rows if row["sequence"] not in excluded]
    if sum(sizes) > len(eligible):
        raise ValueError("not enough unseen sequences after exclusions")
    ordered = sorted(eligible, key=lambda row: digest({"seed": seed, "protein": protein, "sequence": row["sequence"]}))
    a, b, c = sizes
    return ordered[:a], ordered[a : a + b], ordered[a + b : a + b + c]


def prepare_landscape(protein, source, *, sizes=(256, 512, 1024), seed=20261011, exclude_sequences=()):
    if protein not in {"GB1", "TrpB"}:
        raise ValueError("unsupported landscape")
    path = source / (protein + "_fitness.csv")
    manifest = json.loads((source / "source_manifest.json").read_text())
    item = next(x for x in manifest["files"] if x["local_file"] == path.name)
    if sha256_file(path) != item["sha256"]:
        raise ValueError("landscape source changed")
    with path.open() as f:
        rows = [{"id": x["Combo"], "sequence": x["Combo"], "target": float(x["fitness"])} for x in csv.DictReader(f)]
    scope = {
        "protein": protein,
        "assay": "ALDE released " + protein + " combinatorial fitness landscape",
        "target": "fitness",
        "unit": "author-reported fitness",
        "sequence_length": 4,
    }
    for offset in range(0, len(rows), 5000):
        validate_rows(rows[offset : offset + 5000], scope, labels=True)
    excluded = set(exclude_sequences)
    train, val, future = partition(rows, protein, sizes, seed, exclude_sequences=excluded)
    provenance = {
        **item,
        "record": manifest["record"],
        "available_rows": len(rows),
        "zero_fitness_rows": sum(x["target"] == 0 for x in rows),
        "label_policy": "all released rows eligible, including recorded zeros; no absent sequences or imputed labels",
        "representation": "four aligned mutable sites, not full-length sequence",
    }
    split = {
        "rule": "sequence-hash, independent of target values",
        "seed": seed,
        "training": sizes[0],
        "acceptance": sizes[1],
        "later": sizes[2],
        "excluded_sequences": len(excluded),
        "exclusion_sha256": digest(sorted(excluded)),
        "purpose": "frozen-source transfer audit, not reproduction of ALDE active-learning campaigns",
    }
    return scope, train, val, future, provenance, split
