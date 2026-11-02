"""Finish a verified four-generation NucB prefix without repeating raw counts."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from pathlib import Path

from audit_nucb_cnn import frame, require, sha
from nucb_reproduction_checks import compare_tables
from prepare_nucb_assets import write
from run_nucb_author_reproduction import verify_inputs, verify_source


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.processed_run / "report.json").read_text())
    previous = json.loads((args.processed_run / "protocol.json").read_text())
    require(parent["status"] in {"failed", "replayed", "reference_discrepancy"}, "Parent must be terminal")
    require(parent["protocol_sha256"] == sha(args.processed_run / "protocol.json"), "Parent binding changed")
    source = verify_source(args.assets)
    require(source == previous["source"], "Author source changed between stages")
    require(verify_inputs(args.assets, args.asset_report) == previous["input_hashes"], "Source input version changed")
    for filename, expected in previous["implementation"].items():
        require(
            sha(args.processed_run / "implementation" / filename) == expected, "Parent archived implementation changed"
        )
    require([s["generation"] for s in parent["stages"]] == ["g4", "g3", "g2", "g1"], "Incomplete generations")
    checks = {}
    for stage in parent["stages"]:
        require(stage["status"] == "replayed" and stage["matches_author_tables"], "Unverified generation")
        for filename, expected in stage["output_hashes"].items():
            path = args.processed_run / "data" / filename
            require(sha(path) == expected, "Processed prefix changed")
            check = compare_tables(path, args.assets / "gcs" / filename)
            require(check["passed"], "Processed table does not match the pinned reference")
            checks[filename] = {"sha256": expected, "reference_recheck": check}
    protocol = {
        "source": source,
        "parent": str(args.processed_run),
        "parent_report_sha256": sha(args.processed_run / "report.json"),
        "parent_terminal_status": parent["status"],
        "processed_inputs": checks,
        "continuation": "Continue after all four verified generations; restore pathlib required by original export cell. Raw calculations are not repeated; original failure remains.",
        "cells": [9, 10, 11, 12],
        "agent_orchestrated": False,
        "implementation": {},
    }
    for filename in [
        Path(__file__).name,
        "audit_nucb_cnn.py",
        "nucb_reproduction_checks.py",
        "prepare_nucb_assets.py",
        "run_nucb_author_reproduction.py",
    ]:
        path = Path(__file__).with_name(filename)
        protocol["implementation"][filename] = sha(path)
        destination = args.output / "implementation" / filename
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(path.read_bytes())
    write(args.output / "protocol.json", protocol)
    sys.path.insert(0, str(args.assets / "source"))
    import pandas as pd
    from nuclease_design import constants, utils

    notebook = json.loads((args.assets / "source/notebooks/get_landscape.ipynb").read_text())
    output_data = args.output / "data"
    namespace = {
        "utils": utils,
        "constants": constants,
        "pd": pd,
        "itertools": itertools,
        "pathlib": pathlib,
        "LOCAL_OUTPUT_DIR": output_data,
        "all_data_df": utils.load_all_data(str(args.processed_run / "data")),
    }
    for i in protocol["cells"]:
        exec(compile("".join(notebook["cells"][i]["source"]), f"author:get_landscape:cell-{i}", "exec"), namespace)
    landscape = output_data / constants.LANDSCAPE_PATH
    comparison = compare_tables(landscape, args.assets / "gcs" / constants.LANDSCAPE_PATH)
    require(comparison["passed"], "Reconstructed landscape differs from released GCS reference")
    restored = frame(landscape)
    git = frame(args.git_landscape)
    pd.testing.assert_frame_equal(
        restored.set_index("sequence").sort_index(), git.set_index("sequence").sort_index(), check_dtype=False
    )
    result = {
        "status": "replayed",
        "rows": len(restored),
        "unique_sequences": int(restored.sequence.nunique()),
        "gcs_reference": comparison,
        "git_scientific_fields_match": True,
        "serialization_notes": "Git's extra CSV index is metadata; nested tuples/JSON and published display-label aliases are canonicalized explicitly.",
        "activity_counts": {str(k): int(v) for k, v in restored.activity_level.value_counts().items()},
        "landscape_sha256": sha(landscape),
        "git_source_sha256": sha(args.git_landscape),
        "parent_report_sha256": protocol["parent_report_sha256"],
        "prefix_tables_verified": len(checks),
        "protocol_sha256": sha(args.output / "protocol.json"),
        "raw_counts_recomputed": False,
        "agent_orchestrated": False,
        "scope": "Completion of raw-count-to-landscape computation using verified original prefix; no new experiment or new biological replicate.",
    }
    write(args.output / "report.json", result)
    print(json.dumps({k: result[k] for k in ["status", "rows", "unique_sequences", "prefix_tables_verified"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["assets", "asset-report", "processed-run", "git-landscape", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
