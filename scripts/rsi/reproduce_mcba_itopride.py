"""Run the author's McbA itopride calculation, independently of the agent.

Reference: doi:10.1038/s41467-024-55399-0, Extended Predictions.ipynb.
Only worker parallelism and logging are overridden; algorithm parameters are kept.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import re
import resource
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    source = json.loads((root / "source.json").read_text())
    for entry in source["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("source asset path escaped")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("source asset hash mismatch")
    notebook = json.loads((root / "notebooks/Extended Predictions.ipynb").read_text())
    reference = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "ito._predictions_df" in "".join(cell["source"]):
            for output in cell.get("outputs", []):
                for line in output.get("data", {}).get("text/plain", []):
                    match = re.match(r"\s*\d+\s+([A-Z]{4})\s+([-0-9.]+)\s*$", line)
                    if match:
                        reference.append((match[1], float(match[2])))
    if len(reference) != 24:
        raise ValueError("expected the 24 cached author predictions")
    report = {
        "status": "running", "scope": "one published substrate; processed inputs to complete candidate ranking",
        "source_commit": source["commit"], "agent_run": False,
        "full_paper_reproduced": False, "wet_lab_reproduced": False,
        "predeclared_absolute_tolerance": 1e-5,
        "reference": "24 predictions printed to six decimals in the author notebook",
        "adaptations": ["Linux input paths", "GridSearchCV n_jobs=1 and verbose=0", "BLAS threads=1"],
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl"]},
    }
    (args.output / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.path.insert(0, str(root))
    mlde = importlib.import_module("src.AugmentedMLDE_Class")
    original_search = mlde.GridSearchCV
    searches = []

    def single_worker_search(*pos, **kw):
        kw["n_jobs"] = 1
        kw["verbose"] = 0
        obj = original_search(*pos, **kw)
        searches.append(obj)
        return obj

    mlde.GridSearchCV = single_worker_search
    started = time.monotonic()
    model = mlde.AugmentedMLDEmodel(
        str(root / "data/HSS/itopride_train.xlsx"), None, 4,
        str(root / "data/zero-shot_predictions/itopride_zeroshot_EC.csv"), None, None)
    model.train_and_predict("AugmentedEC", "Georgiev")
    predictions = model._predictions_df
    top = predictions.sort_values("Prediction", ascending=False).head(24)
    top_ids = top["Mutation"].tolist()
    reference_ids = [key for key, value in reference]
    by_id = predictions.set_index("Mutation")["Prediction"]
    error = max(abs(float(by_id[key]) - value) for key, value in reference)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    top.to_csv(args.output / "top24.csv", index=False)
    search = searches[-1]
    report.update(
        status="completed", elapsed_seconds=time.monotonic() - started,
        training_rows=len(model._training_data), prediction_rows=len(predictions),
        unique_candidates=int(predictions["Mutation"].nunique()),
        best_alpha=search.best_params_["alpha"], cv_folds=search.n_splits_,
        model="AugmentedEC", encoding="Georgiev", seed=model._random_seed,
        top24_exact_order_match=top_ids == reference_ids,
        top24_overlap=len(set(top_ids) & set(reference_ids)),
        max_abs_error_to_printed_reference=error,
        reference_check_passed=(top_ids == reference_ids and error <= 1e-5),
        predictions_sha256=hashlib.sha256((args.output / "predictions.csv").read_bytes()).hexdigest(),
        input_hashes={x["path"]: x["sha256"] for x in source["files"]},
    )
    (args.output / "baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "training_rows", "prediction_rows", "top24_exact_order_match", "max_abs_error_to_printed_reference", "reference_check_passed", "elapsed_seconds"]}))
    return 0 if report["reference_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
