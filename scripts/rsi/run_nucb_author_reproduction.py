"""Replay author's four-generation NucB raw-count analysis offline.

Author numerical functions and notebook cells are executed unchanged. The sole
I/O hook resolves hard-coded GCS reads into the verified local public snapshot.
This is controller-executed reproduction, not evidence of agent self-improvement.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as metadata
import json
import sys
import time
from pathlib import Path

from nucb_reproduction_checks import ATOL, RTOL, compare_tables
from prepare_nucb_assets import fingerprints, write


def load(path):
    return json.loads(path.read_text())


def verify_source(assets):
    manifest = load(assets / "download_manifest.json")
    for row in manifest["files"]:
        if fingerprints(assets / "source" / row["path"])[0] != row["sha256"]:
            raise ValueError("Author source changed")
    return manifest


def verify_inputs(assets, asset_report):
    report = load(asset_report)
    if report["status"] != "verified":
        raise ValueError("Public input cache is incomplete")
    hashes = {}
    for row in report["files"]:
        path = (assets / "gcs" / row["path"]).resolve()
        path.relative_to((assets / "gcs").resolve())
        sha, md5 = fingerprints(path)
        if sha != row["sha256"] or md5 != row["md5"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Frozen public input changed")
        hashes[row["path"]] = sha
    return hashes


def replay(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = verify_source(args.assets)
    inputs = verify_inputs(args.assets, args.asset_report)
    paths = [
        Path(__file__),
        Path(__file__).with_name("nucb_reproduction_checks.py"),
        Path(__file__).with_name("prepare_nucb_assets.py"),
    ]
    implementation = {}
    for path in paths:
        implementation[path.name] = fingerprints(path)[0]
        dest = args.output / "implementation" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
    protocol = {
        "source": source,
        "input_hashes": inputs,
        "implementation": implementation,
        "runtime": {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "statsmodels", "requests"]},
        "python": sys.version,
        "execution_order": ["g4", "g3", "g2", "g1"],
        "tolerances": {
            "rtol": RTOL,
            "atol": ATOL,
            "categorical": "exact",
            "row_order": "ignored",
            "nested_sets": "sorted, preserving multiplicity",
        },
        "io_adaptation": "GCS read helper redirected to the verified local snapshot; missing files fail; no scientific formula changes",
        "scope": "Author raw-count enrichment, fiducial p-values, FDR activity calls and consolidated landscape; not new sequencing or full TeleProt model/design reproduction",
        "agent_orchestrated": False,
    }
    write(args.output / "protocol.json", protocol)
    sys.path.insert(0, str(args.assets / "source"))
    import pandas as pd
    from nuclease_design import constants, preprocessing, utils

    access = []

    def local_gcs(url):
        prefix = constants.GCS_DATA_DIR + "/"
        if not url.startswith(prefix):
            raise ValueError("Unrecognised author data URL")
        relative = url[len(prefix) :]
        path = (args.assets / "gcs" / relative).resolve()
        path.relative_to((args.assets / "gcs").resolve())
        if relative not in inputs:
            raise ValueError("Author requested unlisted input")
        access.append(relative)
        return path.open("rb")

    utils._read_from_gcs = local_gcs
    output_data = args.output / "data"
    report = {
        "status": "running",
        "stages": [],
        "agent_orchestrated": False,
        "protocol_sha256": fingerprints(args.output / "protocol.json")[0],
    }
    write(args.output / "report.json", report)
    try:
        for generation in protocol["execution_order"]:
            stage = {"generation": generation, "status": "running"}
            report["stages"].append(stage)
            write(args.output / "report.json", report)
            then = time.monotonic()
            frame, fiducials = preprocessing.preprocess_generation(
                data_dir=str(args.assets / "gcs"), **getattr(constants, generation.upper() + "_INFO")
            )
            outputs = {constants.get_processed_data_file(generation): frame}
            outputs.update({constants.get_fiducial_filename(n, generation): df for n, df in fiducials.items()})
            checks = {}
            hashes = {}
            for name, df in outputs.items():
                p = output_data / name
                p.parent.mkdir(parents=True, exist_ok=True)
                utils.encode_df(df).to_csv(p, index=False)
                hashes[name] = fingerprints(p)[0]
            # Preserve every computed table before an audit can fail.
            for name in outputs:
                checks[name] = compare_tables(output_data / name, args.assets / "gcs" / name)
            stage.update(
                status="replayed",
                rows=len(frame),
                comparisons=checks,
                output_hashes=hashes,
                matches_author_tables=all(x["passed"] for x in checks.values()),
                wall_seconds=time.monotonic() - then,
            )
            write(args.output / "report.json", report)
            print(
                json.dumps({"generation": generation, "rows": len(frame), "matches": stage["matches_author_tables"]}),
                flush=True,
            )
            del frame, fiducials, outputs
            gc.collect()
        notebook = load(args.assets / "source/notebooks/get_landscape.ipynb")
        # Exact analysis cells from the pinned notebook; skip installation/magics and
        # replace only the declared input/output directory cell with local paths.
        namespace = {
            "utils": utils,
            "constants": constants,
            "pd": pd,
            "LOCAL_OUTPUT_DIR": output_data,
            "pathlib": __import__("pathlib"),
            "all_data_df": utils.load_all_data(data_dir=str(output_data)),
        }
        import itertools

        namespace["itertools"] = itertools
        for index in [9, 10, 11, 12]:
            text = "".join(notebook["cells"][index]["source"])
            exec(compile(text, f"author:get_landscape:cell_{index}", "exec"), namespace)
        landscape = output_data / constants.LANDSCAPE_PATH
        comparison = compare_tables(landscape, args.assets / "gcs" / constants.LANDSCAPE_PATH)
        frame = pd.read_csv(landscape)
        report["landscape"] = {
            "rows": len(frame),
            "unique_sequences": int(frame["sequence"].nunique()),
            "activity_counts": {str(k): int(v) for k, v in frame["activity_level"].value_counts().items()},
            "gcs_reference": comparison,
            "sha256": fingerprints(landscape)[0],
        }
        if args.git_landscape:
            report["landscape"]["git_reference"] = compare_tables(landscape, args.git_landscape)
            report["landscape"]["git_reference_sha256"] = fingerprints(args.git_landscape)[0]
        report["matches_gcs_reference"] = comparison["passed"] and all(
            x["matches_author_tables"] for x in report["stages"]
        )
        report["status"] = "replayed" if report["matches_gcs_reference"] else "reference_discrepancy"
        report["source_unchanged"] = verify_source(args.assets) == source
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        for recorded_stage in report["stages"]:
            if recorded_stage["status"] == "running":
                recorded_stage.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["redirected_public_reads"] = access
        write(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "landscape_rows": report["landscape"]["rows"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--asset-report", type=Path, required=True)
    p.add_argument("--git-landscape", type=Path)
    p.add_argument("--output", type=Path, required=True)
    replay(p.parse_args())
