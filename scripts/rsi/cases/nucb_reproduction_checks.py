"""Independent table comparisons for the public NucB author replay."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

RTOL = 1e-8
ATOL = 1e-12
NESTED = {"mutations", "nuc_mutations", "sublibrary_names", "generations"}


def canonical(value):
    if pd.isna(value):
        return "<NA>"
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        # The author's encode_df leaves DNA mutation tuples as Python literals,
        # while encoding amino-acid mutations as JSON. Neither format is code.
        data = ast.literal_eval(value)
    if not isinstance(data, (tuple, list)):
        raise ValueError("Nested table field must be a list or tuple")
    return json.dumps(sorted(data), separators=(",", ":"))


def compare_tables(actual_path: Path, reference_path: Path) -> dict:
    actual, reference = pd.read_csv(actual_path), pd.read_csv(reference_path)
    key = "nuc_mutations" if "nuc_mutations" in actual else "mutations"
    report = {
        "actual_rows": len(actual),
        "reference_rows": len(reference),
        "key": key,
        "rtol": RTOL,
        "atol": ATOL,
        "differences": {},
    }
    if key not in actual or key not in reference:
        return {**report, "passed": False, "problem": "missing_key"}
    for frame in (actual, reference):
        for name in NESTED & set(frame):
            frame[name] = frame[name].map(canonical)
    if not actual[key].is_unique or not reference[key].is_unique:
        return {**report, "passed": False, "problem": "duplicate_key"}
    actual, reference = actual.set_index(key), reference.set_index(key)
    akeys, rkeys = set(actual.index), set(reference.index)
    report["actual_only_keys"] = len(akeys - rkeys)
    report["reference_only_keys"] = len(rkeys - akeys)
    report["actual_only_columns"] = sorted(set(actual) - set(reference))
    report["reference_only_columns"] = sorted(set(reference) - set(actual))
    common = sorted(akeys & rkeys)
    report["compared_rows"] = len(common)
    actual, reference = actual.loc[common], reference.loc[common]
    for name in sorted(set(actual) & set(reference)):
        a, b = actual[name], reference[name]
        numeric = all(pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) for s in (a, b))
        if numeric:
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            match = np.isclose(av, bv, rtol=RTOL, atol=ATOL, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            max_error = float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.0
        else:
            match = (a.fillna("<NA>").astype(str) == b.fillna("<NA>").astype(str)).to_numpy()
            max_error = None
        if not match.all():
            report["differences"][name] = {"mismatched_rows": int((~match).sum()), "max_absolute_error": max_error}
    report["passed"] = not any(
        report[k]
        for k in [
            "actual_only_keys",
            "reference_only_keys",
            "actual_only_columns",
            "reference_only_columns",
            "differences",
        ]
    )
    return report
