"""Frozen, label-free descriptor map for separate antibody variable chains."""

import hashlib
from collections import Counter

NAME = "separate_vh_vl_composition_dipeptide32_length_target_onehot.v1"
AA = "ACDEFGHIKLMNPQRSTVWY"
TARGETS = ("P00533", "P02768", "P04626", "P08581", "P15692", "P15941", "P16422", "P22223", "P61626", "Q13421", "Q16790")


def encode(row):
    values = []
    for key in ["heavy", "light"]:
        sequence = row[key]
        counts = Counter(sequence)
        values.extend(counts[a] / max(1, len(sequence)) for a in AA)
        bins = [0.0] * 32
        for i in range(len(sequence) - 1):
            position = int(hashlib.sha256(sequence[i : i + 2].encode()).hexdigest(), 16) % 32
            bins[position] += 1 / (len(sequence) - 1)
        values.extend(bins)
        values.append(len(sequence) / 200)
    values.append(float(bool(row["light"])))
    values.extend(float(row["protein"] == target) for target in TARGETS)
    return values


def covariates():
    names = []
    for chain in ("heavy", "light"):
        names += [f"{chain}_fraction_{a}" for a in AA]
        names += [f"{chain}_dipeptide_bin_{i}" for i in range(32)]
        names.append(chain + "_length_over_200")
    return [{"name": n, "unit": "fraction", "minimum": 0, "maximum": 1} for n in names] + [
        {"name": n, "unit": "binary_indicator", "minimum": 0, "maximum": 1}
        for n in ["light_present", *("target_" + t for t in TARGETS)]
    ]
