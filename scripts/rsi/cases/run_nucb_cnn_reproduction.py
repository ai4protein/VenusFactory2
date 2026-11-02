"""Replay pinned NucB CNN weights and notebook outputs on the released data.

The author ensemble is preserved. Per-sequence normalization is a separately
labelled sensitivity check, not a silent change to the reference algorithm.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import importlib.metadata as metadata
import json
import os
import re
import sys
import time
from pathlib import Path

from scripts.rsi.cases.nucb_reproduction_checks import canonical
from scripts.rsi.cases.prepare_nucb_assets import fingerprints, write
from scripts.rsi.cases.run_nucb_author_reproduction import verify_source

ACTIVITY_ALIASES = {
    "non-functional": "non-functional",
    "activity > 0": "activity_greater_than_0",
    "activity > WT": "activity_greater_than_WT",
    "activity > A73R": "activity_greater_than_A73R",
    "activity_greater_than_0": "activity_greater_than_0",
    "activity_greater_than_WT": "activity_greater_than_WT",
    "activity_greater_than_A73R": "activity_greater_than_A73R",
}


def normalize_activity_labels(frame):
    result = frame.copy()
    if result["activity_level"].isna().any() or not set(result["activity_level"]).issubset(ACTIVITY_ALIASES):
        raise ValueError("Unknown or missing published activity category")
    result["activity_level"] = result["activity_level"].map(ACTIVITY_ALIASES)
    return result


def text_output(notebook, index):
    return "\n".join(
        "".join(o.get("data", {}).get("text/plain", [])) for o in notebook["cells"][index].get("outputs", [])
    )


def expected_array(notebook, index):
    match = re.search(r"array\(\[\[(.*?)\]\]", text_output(notebook, index), flags=re.S)
    if not match:
        raise ValueError("Missing author array output")
    return [float(x.strip()) for x in match.group(1).split(",") if x.strip()]


def expected_confusion(notebook):
    rows = {}
    for line in text_output(notebook, 19).splitlines():
        tokens = line.split()
        if len(tokens) == 5 and all(t.isdigit() for t in tokens[1:]):
            rows[tokens[0]] = [int(t) for t in tokens[1:]]
    if len(rows) != 4:
        raise ValueError("Missing author cross-tabulation output")
    return rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = verify_source(args.assets)
    files = json.loads(args.asset_report.read_text())
    if files["status"] != "verified":
        raise ValueError("Input cache incomplete")
    selected = [
        r for r in files["files"] if r["path"].startswith("cnn_models/") or r["path"] == "processed_data/landscape.csv"
    ]
    for row in selected:
        if fingerprints(args.assets / "gcs" / row["path"])[0] != row["sha256"]:
            raise ValueError("Frozen model/data input changed")
    git_manifest = json.loads((args.git_landscape.parent.parent / "source.json").read_text())
    git_record = next(x for x in git_manifest["files"] if x["path"] == "data/landscape.csv")
    if git_manifest["commit"] != source["commit"] or fingerprints(args.git_landscape)[0] != git_record["sha256"]:
        raise ValueError("Pinned Git landscape changed")
    notebook = json.loads((args.assets / "source/notebooks/analyze_cnn.ipynb").read_text())
    protocol = {
        "source_commit": source["commit"],
        "source_manifest": source,
        "inputs": selected,
        "git_landscape_sha256": fingerprints(args.git_landscape)[0],
        "runtime": {p: metadata.version(p) for p in ["tensorflow", "tf-keras", "numpy", "pandas", "scipy"]},
        "keras_mode": os.environ.get("TF_USE_LEGACY_KERAS"),
        "device": "CPU",
        "tolerance": {"rtol": 1e-4, "atol": 1e-7},
        "reference_wt_single": expected_array(notebook, 11),
        "reference_wt_ensemble": expected_array(notebook, 13),
        "reference_confusion": expected_confusion(notebook),
        "reference_parameters": int(text_output(notebook, 21).strip()),
        "evaluation": "Author published checkpoint replay, not retraining or a held-out generalization experiment",
        "input_serialization": "JSON arrays and safe tuple literals canonicalized before the unchanged author decoder; no label/sequence changes",
        "sensitivity": "Normalize author mean log-probabilities independently per row; compare labels and probability mass without selecting a new model",
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
    protocol["published_label_aliases"] = ACTIVITY_ALIASES
    write(args.output / "protocol.json", protocol)
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    sys.path.insert(0, str(args.assets / "source"))
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from nuclease_design import cnn_model, constants, utils
    from scipy import special

    if tf.config.list_physical_devices("GPU"):
        raise ValueError("This reproduction requires the declared CPU execution")
    models = cnn_model.load_cnn_ensemble(str(args.assets / "gcs"))
    single = cnn_model.get_predict_fn(models[0])
    ensemble = cnn_model.get_ensemble_predict_fn(models)
    wt = [constants.FULL_REFERENCE_SEQ]
    single_wt, ensemble_wt = single(wt), ensemble(wt)
    wt_pass = bool(
        np.allclose(single_wt[0], protocol["reference_wt_single"], rtol=1e-4, atol=1e-7)
        and np.allclose(ensemble_wt[0], protocol["reference_wt_ensemble"], rtol=1e-4, atol=1e-7)
    )
    params = [m.count_params() for m in models]
    if params != [protocol["reference_parameters"]] * 5:
        raise ValueError("Model architecture parameter count differs from notebook")
    report = {
        "status": "running",
        "wt_single": single_wt.tolist(),
        "wt_ensemble": ensemble_wt.tolist(),
        "wildtype_reference_matches": wt_pass,
        "parameters": params,
        "cohorts": [],
        "agent_orchestrated": False,
        "retraining_performed": False,
    }
    write(args.output / "report.json", report)
    try:
        # Both released landscape versions are evaluated explicitly, never choose
        # whichever happens to match a desired notebook result after inspection.
        for name, path in [("gcs", args.assets / "gcs/processed_data/landscape.csv"), ("git", args.git_landscape)]:
            encoded_frame = pd.read_csv(path)
            # Git CSV uses tuple literals whereas the GCS CSV uses JSON arrays.
            # Normalize only nested-field serialization before author decoding.
            for column in ["mutations", "sublibrary_names", "generations"]:
                encoded_frame[column] = encoded_frame[column].map(canonical)
            df = normalize_activity_labels(utils.decode_df(encoded_frame))
            if not df["sequence"].is_unique:
                raise ValueError("Duplicate sequence in published landscape")
            cohort = df[df["generations"] != ("g4",)].reset_index(drop=True)
            seqs = cohort["sequence"].tolist()
            encoded = cnn_model.int_encode_sequences([cnn_model.trim_sequence_to_variable_region(s) for s in seqs])
            individual = np.stack([m.predict(encoded, verbose=0) for m in models], axis=-1)
            mean_log = np.mean(np.log(individual + 1e-6), axis=-1)
            original = special.softmax(mean_log)  # exact expression in pinned author function
            row_normalized = special.softmax(mean_log, axis=-1)
            # Compare the factored computation against the unmodified entrypoint
            # on a bounded prefix; the global normalization is recomputed there.
            size = min(16, len(seqs))
            direct = ensemble(seqs[:size])
            expected_prefix = special.softmax(mean_log[:size])
            if not np.allclose(direct, expected_prefix, rtol=1e-4, atol=1e-7):
                raise ValueError("Factored full-cohort inference differs from author entrypoint")
            if not np.array_equal(original.argmax(axis=1), row_normalized.argmax(axis=1)):
                raise ValueError("Normalization sensitivity unexpectedly changes class ranking")
            predicted = [cnn_model.OUTPUT_CLASSES[i] for i in original.argmax(axis=1)]
            table = pd.crosstab(
                cohort["activity_level"], pd.Series(predicted, name="predicted_activity_level")
            ).reindex(
                index=list(constants.LANDSCAPE_ACTIVITY_LEVELS), columns=list(cnn_model.OUTPUT_CLASSES), fill_value=0
            )
            if int(table.to_numpy().sum()) != len(cohort):
                raise ValueError("Cross-tabulation omitted evaluated sequences")
            observed = {str(k): [int(v) for v in table.loc[k].values] for k in table.index}
            saved = args.output / name
            saved.mkdir()
            cohort.to_csv(saved / "cohort.csv", index=False)
            table.to_csv(saved / "cross_tabulation.csv")
            np.savez_compressed(
                saved / "predictions.npz", individual=individual, author=original, per_sequence=row_normalized
            )
            result = {
                "source": name,
                "rows": len(cohort),
                "landscape_rows": len(df),
                "source_sha256": fingerprints(path)[0],
                "cross_tabulation": observed,
                "notebook_cross_tabulation_matches": observed == protocol["reference_confusion"],
                "author_probability_total": float(original.sum()),
                "author_probability_row_sum_min": float(original.sum(axis=1).min()),
                "author_probability_row_sum_max": float(original.sum(axis=1).max()),
                "per_sequence_max_row_mass_error": float(np.max(np.abs(row_normalized.sum(axis=1) - 1))),
                "class_predictions_unchanged_by_row_normalization": True,
                "outputs": {f.name: fingerprints(f)[0] for f in saved.iterdir()},
            }
            report["cohorts"].append(result)
            write(args.output / "report.json", report)
            print(
                json.dumps(
                    {
                        "source": name,
                        "rows": len(cohort),
                        "notebook_matches": result["notebook_cross_tabulation_matches"],
                    }
                ),
                flush=True,
            )
        # A duplicated WT request exposes the batch dependence directly.
        duplicated = ensemble(wt * 2)
        report["batch_dependence"] = {
            "duplicated_wildtype": duplicated.tolist(),
            "max_vs_half_single_error": float(np.max(np.abs(duplicated - ensemble_wt / 2))),
            "row_mass": duplicated.sum(axis=1).tolist(),
        }
        report["status"] = (
            "replayed"
            if wt_pass and all(c["notebook_cross_tabulation_matches"] for c in report["cohorts"])
            else "reference_discrepancy"
        )
        report["source_unchanged"] = verify_source(args.assets) == source
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        write(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "wildtype_reference_matches": wt_pass}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--asset-report", type=Path, required=True)
    p.add_argument("--git-landscape", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
