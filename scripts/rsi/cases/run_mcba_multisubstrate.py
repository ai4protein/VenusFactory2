"""Replay every public McbA notebook prediction/model-selection task.

The declared starting point is author processed assay and zero-shot tables.
This is author-method reconstruction, not agent autonomy or upstream zero-shot training.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMMIT = "0882f67317f56dfd2a39474f8f548e73197c1e8d"
PRODUCTS = [
    "moclobemide",
    "metoclopramide",
    "itopride",
    "cinchocaine",
    "declopramide",
    "trimethobenzamide",
    "sulpiride",
    "procainamide",
    "troxipide",
]
SUBSETS = ["full", "BLOSUM1", "BLOSUM2", "BLOSUM3", "BLOSUM4", "NDT", "NRT", "AGCP"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def verify_source(source):
    manifest = load(source / "source.json")
    if manifest["status"] != "completed" or manifest["commit"] != COMMIT:
        raise ValueError("Complete pinned author source required")
    for entry in manifest["files"]:
        path = (source / entry["path"]).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("Author source changed: " + entry["path"])
    return manifest


def plain(cell):
    return "\n".join("".join(o.get("data", {}).get("text/plain", [])) for o in cell.get("outputs", []))


def ranking_reference(cell, count):
    rows = []
    for line in plain(cell).splitlines():
        match = re.fullmatch(r"\s*\d+\s+([A-Z]{4})\s+([-0-9.]+)\s*", line)
        if match:
            rows.append({"Mutation": match[1], "Prediction": float(match[2])})
    if len(rows) != count or len({r["Mutation"] for r in rows}) != count:
        raise ValueError("Incomplete cached notebook ranking")
    return rows


def metric_reference(cell):
    rows, alphas = [], {}
    for line in plain(cell).splitlines():
        m = re.fullmatch(r"\s*(\d+)\s+(.+?: .+?)\s+([-0-9.]+)\s+([-0-9.]+)\s*\\?", line)
        if m:
            rows.append(
                {"index": int(m[1]), "architecture": m[2].strip(), "spearman_r": float(m[3]), "NDCG": float(m[4])}
            )
        a = re.fullmatch(r"\s*(\d+)\s+Ridge\(alpha=([0-9.eE+-]+)\)\s*", line)
        if a:
            alphas[int(a[1])] = float(a[2])
    if [x["index"] for x in rows] != list(range(40)) or set(alphas) != set(range(40)):
        raise ValueError("Incomplete cached notebook architecture table")
    return [{**row, "alpha": alphas[row["index"]]} for row in rows]


def plan(source):
    selection = load(source / "notebooks/Model Selection.ipynb")["cells"]
    extended = load(source / "notebooks/Extended Predictions.ipynb")["cells"]
    tasks = []
    for i, product in enumerate(PRODUCTS):
        cell = selection[8 + i] if i < 2 else extended[3 + 2 * (i - 2)]
        tasks.append(
            {
                "id": "rank_" + product,
                "kind": "rank",
                "product": product,
                "subset": "full",
                "reference": ranking_reference(cell, 25 if i < 2 else 24),
                "status": "allocated",
            }
        )
    for i, product in enumerate(PRODUCTS[:2]):
        for subset in SUBSETS:
            tasks.append(
                {
                    "id": "compare_" + product + "_" + subset,
                    "kind": "compare",
                    "product": product,
                    "subset": subset,
                    "reference": metric_reference(selection[5 + i]) if subset == "full" else None,
                    "status": "allocated",
                }
            )
    return tasks


def input_paths(source, task):
    product = task["product"]
    prefix = "data/ML_validation/" if product in PRODUCTS[:2] else "data/HSS/"
    training = source / (prefix + product + "_train.xlsx")
    if task["subset"] != "full":
        training = source / ("data/curated_reduced_AAs/" + product + "_train_" + task["subset"] + ".xlsx")
    test = source / ("data/ML_validation/" + product + "_test.xlsx") if task["kind"] == "compare" else None
    ec = source / ("data/zero-shot_predictions/" + product + "_zeroshot_EC.csv")
    energy = source / ("data/zero-shot_predictions/" + product + "_zeroshot_maestro.csv") if test else None
    esm = source / ("data/zero-shot_predictions/" + product + "_zeroshot_ESM1b_nc.csv") if test else None
    return training, test, ec, energy, esm


def run_worker(args):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    protocol = load(args.output / "protocol.json")
    if sha(Path(__file__)) != protocol["runner_sha256"]:
        raise ValueError("Runner differs from frozen protocol")
    verify_source(args.source)
    task = next(x for x in protocol["tasks"] if x["id"] == args.task)
    folder = args.output / task["id"]
    folder.mkdir()
    sys.path.insert(0, str(args.source))
    mlde = importlib.import_module("src.AugmentedMLDE_Class")
    original = mlde.GridSearchCV
    searches = []

    def serial_grid(*pos, **kw):
        kw.update(n_jobs=1, verbose=0)
        search = original(*pos, **kw)
        searches.append(search)
        return search

    mlde.GridSearchCV = serial_grid
    paths = input_paths(args.source, task)
    training, test, ec, energy, esm = paths
    model = mlde.AugmentedMLDEmodel(
        str(training),
        str(test) if test else None,
        4,
        str(ec),
        str(energy) if energy else None,
        str(esm) if esm else None,
    )
    result = {
        "id": task["id"],
        "status": "running",
        "input_hashes": {str(p.relative_to(args.source)): sha(p) for p in paths if p},
        "training_rows": len(model._training_data),
        "test_rows": len(model._test_data) if test else 0,
    }
    started = time.monotonic()
    if task["kind"] == "rank":
        model.train_and_predict("AugmentedEC", "Georgiev")
        frame = model._predictions_df
        reference = task["reference"]
        top = frame.sort_values("Prediction", ascending=False).head(len(reference))
        expected_ids = [x["Mutation"] for x in reference]
        values = frame.set_index("Mutation")["Prediction"]
        error = max(abs(float(values[x["Mutation"]]) - x["Prediction"]) for x in reference)
        top.to_csv(folder / "top.csv", index=False)
        result.update(
            top_exact_order_match=top["Mutation"].tolist() == expected_ids,
            top_overlap=len(set(top["Mutation"]) & set(expected_ids)),
            max_reference_absolute_error=error,
        )
        result["reference_passed"] = result["top_exact_order_match"] and error <= protocol["ranking_tolerance"]
    else:
        model.compare_and_predict()
        frame = model._all_predictions
        metrics = model._model_metrics.copy()
        metrics["Tuned Alpha"] = [float(search.best_params_["alpha"]) for search in searches]
        metrics.to_csv(folder / "metrics.csv", index=False)
        model._test_predictions.to_csv(folder / "test_predictions.csv", index=False)
        # Independently recalculate the author's signed, standardized-activity DCG convention.
        actual = model._test_data["Activity"].to_numpy(dtype=float)
        actual = (actual - actual.mean()) / actual.std(ddof=1)
        recomputed = []
        for row in metrics.to_dict("records"):
            predicted = model._test_predictions[row["architecture"]].to_numpy(dtype=float)
            order = pd.Series(predicted).sort_values(ascending=False).index.to_numpy()
            ideal = pd.Series(actual).sort_values(ascending=False).index.to_numpy()
            discount = np.log2(np.arange(len(actual)) + 2)
            ndcg = float(np.sum(actual[order] / discount) / np.sum(actual[ideal] / discount))
            rho = float(spearmanr(actual, predicted).statistic)
            if not np.isclose(ndcg, row["NDCG"], rtol=1e-10, atol=1e-10) or not np.isclose(
                rho, row["spearman_r"], rtol=1e-10, atol=1e-10
            ):
                raise ValueError("Independent test metrics differ")
            recomputed.append({"architecture": row["architecture"], "NDCG": ndcg, "spearman_r": rho})
        write(folder / "metric_recalculation.json", recomputed)
        result.update(architectures=len(metrics), metric_recalculation_passed=True)
        if task["reference"] is not None:
            observed = metrics.set_index("architecture")
            errors = [
                {
                    "architecture": row["architecture"],
                    **{
                        key: abs(float(observed.loc[row["architecture"], key]) - row[key])
                        for key in ["NDCG", "spearman_r"]
                    },
                    "alpha": abs(float(observed.loc[row["architecture"], "Tuned Alpha"]) - row["alpha"]),
                }
                for row in task["reference"]
            ]
            write(folder / "reference_comparison.json", errors)
            result["max_reference_absolute_errors"] = {
                key: max(x[key] for x in errors) for key in ["NDCG", "spearman_r", "alpha"]
            }
            result["reference_passed"] = all(
                x[key] <= (1e-10 if key == "alpha" else protocol["metric_tolerance"])
                for x in errors
                for key in ["NDCG", "spearman_r", "alpha"]
            )
        else:
            result["reference_passed"] = None
            result["reference_limit"] = (
                "Reduced-training notebooks cache a figure without numerical table; internal metric recalculation is not historical numerical agreement."
            )
    if len(frame) != 160000 or frame["Mutation"].nunique() != 160000:
        raise ValueError("Incomplete combinatorial prediction grid")
    array = frame.drop(columns="Mutation").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite predictions")
    np.savez_compressed(
        folder / "predictions.npz",
        mutations=frame["Mutation"].to_numpy(dtype="U4"),
        columns=np.asarray(frame.columns[1:], dtype="U64"),
        predictions=array,
    )
    write(
        folder / "grid_search.json",
        [
            {
                "best_alpha": float(s.best_params_["alpha"]),
                "best_score": float(s.best_score_),
                "folds": s.n_splits_,
                "alpha_grid": list(s.param_grid["alpha"]),
            }
            for s in searches
        ],
    )
    result.update(
        status="completed",
        wall_seconds=time.monotonic() - started,
        prediction_rows=len(frame),
        searches=len(searches),
        output_hashes={f.name: sha(f) for f in folder.iterdir() if f.is_file()},
    )
    write(folder / "report.json", result)
    print(json.dumps({k: result[k] for k in ["id", "status", "reference_passed", "wall_seconds"]}), flush=True)


def main(args):
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    manifest = verify_source(args.source)
    tasks = plan(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": 1,
        "commit": COMMIT,
        "source_manifest_sha256": sha(args.source / "source.json"),
        "runner_sha256": sha(Path(__file__)),
        "tasks": tasks,
        "ranking_tolerance": 1e-5,
        "metric_tolerance": 1e-5,
        "scope": "All unique processed-input computational tasks in both author notebooks: nine ranked products, two 40-architecture comparisons and fourteen reduced-training comparisons.",
        "adaptations": [
            "Linux paths including lowercase maestro filenames",
            "single GridSearch worker and BLAS thread",
            "serialize outputs instead of inline plotting",
            "reuse full-training comparison for the identical full entry in reduced-set loops",
        ],
        "limits": [
            "author-provided zero-shot tables, not their upstream regeneration",
            "one protein family, not nine independent families",
            "author seed 42 and grid unchanged",
            "no new biological or autonomous-agent result",
            "author NDCG uses signed standardized activities; do not reinterpret as canonical nonnegative-relevance NDCG",
            "notebook cached outputs may precede current source expressions; differences retained",
        ],
        "budget": {"cpu_seconds_per_task": 3600, "wall_seconds_per_task": 4200, "memory_GiB": 8},
        "packages": {
            p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl"]
        },
    }
    write(args.output / "protocol.json", protocol)
    write(args.output / "source_manifest.json", manifest)
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    report = {
        "status": "preflight",
        "protocol_sha256": sha(args.output / "protocol.json"),
        "tasks": [dict(t) for t in tasks],
        "full_paper_reproduced": False,
        "agent_orchestrated": False,
    }
    write(args.output / "report.json", report)
    if not args.execute:
        print(json.dumps({"status": "preflight", "tasks": len(tasks), "source_files": len(manifest["files"])}))
        return
    started = time.monotonic()
    report["status"] = "running"
    for record in report["tasks"]:
        record["status"] = "running"
        write(args.output / "report.json", report)
        argv = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--source",
            str(args.source),
            "--output",
            str(args.output),
            "--task",
            record["id"],
        ]
        with (args.output / (record["id"] + ".log")).open("w") as log:
            try:
                process = subprocess.run(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=4200,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
                done = args.output / record["id"] / "report.json"
                if process.returncode == 0 and done.exists():
                    record.update(load(done))
                else:
                    record.update(status="failed", returncode=process.returncode)
            except subprocess.TimeoutExpired:
                record.update(status="failed", error_type="wall_timeout")
        write(args.output / "report.json", report)
        print(
            json.dumps(
                {"id": record["id"], "status": record["status"], "reference_passed": record.get("reference_passed")}
            ),
            flush=True,
        )
    report.update(status="campaign_completed", wall_seconds=time.monotonic() - started)
    write(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.task:
        try:
            run_worker(args)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1) from None
    else:
        main(args)
