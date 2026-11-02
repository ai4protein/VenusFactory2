"""Recompute NucB hit-rate tables and purified-protein activity from author cells.

The input provenance distinguishes released processed tables from reconstructed
raw-count outputs. Notebook mathematics is unchanged; plot-only cells are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import io
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from prepare_nucb_assets import fingerprints, write
from run_nucb_author_reproduction import verify_source


def run_cells(notebook, indices, namespace):
    for index in indices:
        code = "".join(notebook["cells"][index]["source"])
        exec(compile(code, f"author-notebook:cell-{index}", "exec"), namespace)


def text_output(nb, index):
    return "".join(
        "".join(o.get("text", o.get("data", {}).get("text/plain", []))) for o in nb["cells"][index].get("outputs", [])
    )


def align_bootstrap_row_order(data, reference):
    """Match author positional resampling using identities, never reference values.

    Raw reconstruction may permute equivalent sequence rows. A fixed RNG seed
    then samples different sequences by position. Reordering restores the
    published sampling frame while retaining every reconstructed measurement.
    """
    keys = ["generation", "sequence"]
    actual = data.set_index(keys, verify_integrity=True)
    reference_index = reference.set_index(keys, verify_integrity=True).index
    if set(actual.index) != set(reference_index):
        raise ValueError("Bootstrap source identities differ from the published sampling frame")
    ordered = actual.loc[reference_index].reset_index()[data.columns]
    payload = json.dumps(reference_index.tolist(), separators=(",", ":"), ensure_ascii=False).encode()
    return ordered, {
        "policy": "published_identity_order_v1",
        "keys": keys,
        "rows": len(data),
        "reordered_rows": sum(a != b for a, b in zip(actual.index, reference_index, strict=True)),
        "identity_order_sha256": hashlib.sha256(payload).hexdigest(),
        "measurements": "reconstructed values retained; reference used for identity order only",
    }


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = verify_source(args.assets)
    cache = json.loads(args.asset_report.read_text())
    if cache["status"] != "verified":
        raise ValueError("Input cache incomplete")
    hashes = {x["path"]: x["sha256"] for x in cache["files"]}
    relevant = [f"processed_data/g{i}.csv" for i in range(1, 5)] + [
        "plate_data/genotypes.csv",
        "plate_data/2021-09-07-purified-protein-comparison-activity.xml",
    ]
    for name in relevant:
        if fingerprints(args.assets / "gcs" / name)[0] != hashes[name]:
            raise ValueError("Pinned source data changed")
    data_root = args.data_root or args.assets / "gcs"
    actual_inputs = {name: fingerprints(data_root / name)[0] for name in relevant if name.startswith("processed_data/")}
    protocol = {
        "source_commit": manifest["commit"],
        "source_manifest": manifest,
        "source_inputs": {name: hashes[name] for name in relevant},
        "analysis_inputs": actual_inputs,
        "execution_origin": "released_processed_snapshot"
        if args.data_root is None
        else "provided_reconstructed_processed_tables",
        "data_root": str(data_root),
        "runtime": {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "statsmodels"]},
        "hit_rate_cells": [13, 14, 15, 16, 17, 18],
        "bootstrap_seed": 226,
        "purified_cells": [5, 8, 9, 11, 12, 13, 16, 19],
        "scope": "Recompute author statistical analysis, not new wet-lab data, model training or agent-efficacy inference",
        "slope_replay_tolerance": {"rtol": 1e-10, "atol": 1e-9},
        "implementation_sha256": fingerprints(Path(__file__))[0],
    }
    dependencies = [
        Path(__file__),
        Path(__file__).with_name("run_nucb_author_reproduction.py"),
        Path(__file__).with_name("prepare_nucb_assets.py"),
        Path(__file__).with_name("nucb_reproduction_checks.py"),
    ]
    protocol["implementation"] = {p.name: fingerprints(p)[0] for p in dependencies}
    for dependency in dependencies:
        destination = args.output / "implementation" / dependency.name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(dependency.read_bytes())
    write(args.output / "protocol.json", protocol)
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    sys.path.insert(0, str(args.assets / "source"))
    import numpy as np
    import pandas as pd
    from nuclease_design import constants, utils
    from scipy import stats

    def local_gcs(url):
        prefix = constants.GCS_DATA_DIR + "/"
        if not url.startswith(prefix):
            raise ValueError("Unexpected data URL")
        name = url[len(prefix) :]
        if name not in hashes:
            raise ValueError("Unlisted source data")
        return (args.assets / "gcs" / name).open("rb")

    utils._read_from_gcs = local_gcs
    notebook = json.loads((args.assets / "source/notebooks/plot_hit_rates.ipynb").read_text())
    data = utils.load_all_data(str(data_root))
    reference = utils.load_all_data(str(args.assets / "gcs")) if args.data_root is not None else data
    data, protocol["bootstrap_row_order"] = align_bootstrap_row_order(data, reference)
    write(args.output / "protocol.json", protocol)
    expanded = utils.expand_sublibraries(data)
    ns = {"np": np, "pd": pd, "utils": utils, "constants": constants, "df": expanded, "RANDOM_SEED": 226}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        run_cells(notebook, protocol["hit_rate_cells"], ns)
    (args.output / "author_hit_rate_stdout.txt").write_text(captured.getvalue())
    table = ns["table"]
    table.to_csv(args.output / "hit_rates.csv", index=False)
    expected = pd.read_csv(io.StringIO(text_output(notebook, 18)), keep_default_na=False)
    observed = pd.read_csv(args.output / "hit_rates.csv", keep_default_na=False)
    expected = expected.sort_values("name").reset_index(drop=True)
    observed = observed.sort_values("name").reset_index(drop=True)
    differences = {}
    if list(observed.columns) == list(expected.columns) and len(observed) == len(expected):
        for column in observed:
            a, b = observed[column].astype(str), expected[column].astype(str)
            if not a.equals(b):
                differences[column] = int((a != b).sum())
    else:
        differences["shape_or_columns"] = True
    # Retain full precision estimates, not only rounded notebook display strings.
    for name, frame in ns["stats_dfs"].items():
        frame.to_csv(args.output / ("hit_rate_precision_" + name + ".csv"))
    report = {
        "status": "running",
        "hit_rates": {
            "sublibraries": len(table),
            "matches_notebook_rounded_table": not differences,
            "differences": differences,
            "printed_tests": captured.getvalue().split("generation,name", 1)[0],
        },
        "agent_orchestrated": False,
    }
    write(args.output / "report.json", report)
    try:
        notebook = json.loads((args.assets / "source/notebooks/plot_purified_protein_results.ipynb").read_text())
        ns = {
            "np": np,
            "pd": pd,
            "stats": stats,
            "ET": ET,
            "utils": utils,
            "constants": constants,
            "DATA_DIR": str(args.assets / "gcs"),
        }
        run_cells(notebook, protocol["purified_cells"], ns)
        velocities = ns["initial_velocity_df"]
        velocities.drop(columns="initial_velocity_resampled").to_csv(
            args.output / "purified_initial_velocity.csv", index=False
        )
        ns["df"].to_csv(args.output / "purified_time_series.csv", index=False)
        # Independent least-squares reconstruction of all 28 original well fits.
        checks = []
        for identity, frame in ns["df"].groupby(["strain", "genotype", "concentration"]):
            n = 100 if identity[1] == "wild type" else 10
            values = frame.sort_values("time").iloc[:n]
            x, y = values["time"].to_numpy(), values["fluorescence"].to_numpy()
            xm, ym = x - x.mean(), y - y.mean()
            slope = float(np.dot(xm, ym) / np.dot(xm, xm))
            intercept = float(y.mean() - slope * x.mean())
            residual = y - (slope * x + intercept)
            stderr = float(np.sqrt(np.dot(residual, residual) / (len(x) - 2) / np.dot(xm, xm)))
            original = velocities[
                (velocities["strain"] == identity[0])
                & (velocities["genotype"] == identity[1])
                & (velocities["concentration"] == identity[2])
            ].iloc[0]
            ok = bool(
                np.allclose(
                    [slope, intercept, stderr],
                    original[["initial_velocity", "intercept", "initial_velocity_stderr"]].to_numpy(dtype=float),
                    rtol=1e-10,
                    atol=1e-9,
                )
            )
            checks.append(
                {
                    "strain": identity[0],
                    "genotype": identity[1],
                    "concentration": float(identity[2]),
                    "points": n,
                    "passed": ok,
                }
            )
        if not all(x["passed"] for x in checks):
            raise ValueError("Independent kinetic fit reconstruction differs")
        # Fixed parametric bootstrap comes from author cell 19. The original WT
        # denominator-resampling cell is unseeded; report deterministic ratios of
        # fitted slopes separately instead of pretending to reproduce that RNG.
        wt = velocities[velocities["genotype"] == "wild type"].set_index("concentration")["initial_velocity"]
        ratios = velocities.drop(columns="initial_velocity_resampled").copy()
        ratios["ratio_of_fitted_slopes_to_wt"] = ratios["initial_velocity"] / ratios["concentration"].map(wt)
        ratios.to_csv(args.output / "purified_slope_ratios.csv", index=False)
        report["purified_protein"] = {
            "time_series_points": len(ns["df"]),
            "fits": len(velocities),
            "fit_reconstruction": checks,
            "bootstrap_draws_per_fit": 1000,
            "uncertainty_scope": "Parametric uncertainty of initial-velocity line fits; not independent biological replicates",
            "ratio_scope": "Ratios of fitted slopes at matched concentration; original unseeded WT-resampling visualization not claimed reproduced",
        }
        report["status"] = "replayed" if not differences else "reference_discrepancy"
        report["source_unchanged"] = verify_source(args.assets) == manifest
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        report["outputs"] = {
            p.name: fingerprints(p)[0]
            for p in args.output.iterdir()
            if p.is_file() and p.name not in {"report.json", "report.json.tmp"}
        }
        write(args.output / "report.json", report)
    print(
        json.dumps({"status": report["status"], "hit_rate_sublibraries": len(table), "kinetic_fits": len(velocities)}),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--asset-report", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
