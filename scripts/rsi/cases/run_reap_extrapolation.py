"""Author-code single-to-multiple extrapolation on the strict REAP cohort.

This retrospective computation retains both source discrepancies. It does not
claim exact historical splits, complete paper replication or agent efficacy.
"""
from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def allocations(rows):
    runs = []
    for family in sorted({x["family"] for x in rows}):
        singles = [i for i, x in enumerate(rows) if x["family"] == family and x["mutation_order"] == 1]
        future = [i for i, x in enumerate(rows) if x["family"] == family and x["mutation_order"] > 1]
        for seed in range(10):
            train, val = train_test_split(singles, test_size=0.2, random_state=seed, shuffle=True)
            groups = [{rows[i]["sequence"] for i in indices} for indices in (train, val, future)]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)) or not all(groups):
                raise ValueError("Empty split or full-sequence overlap")
            for loss in ["RankReg", "MSE", "Huber", "L1", "EVOLVEpro"]:
                runs.append({"id": f"{family}_seed{seed:02d}_{loss}", "family": family, "seed": seed, "loss": loss, "train": train, "validation": val, "future": future})
    return runs


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"status": "running", "fits": []}
    try:
        rows = json.loads((args.admission / "campaign.json").read_text())
        feature_record = json.loads((args.features / "report.json").read_text())
        if feature_record["status"] != "completed" or feature_record["input_sha256"] != sha(args.admission / "campaign.json") or feature_record["output_sha256"] != sha(args.features / "embeddings.npz"):
            raise ValueError("Feature identity changed")
        with np.load(args.features / "embeddings.npz", allow_pickle=False) as saved:
            X = saved["X"].copy()
            if saved["ids"].tolist() != [x["id"] for x in rows] or saved["sequence_sha256"].tolist() != [x["sequence_sha256"] for x in rows] or not np.isfinite(X).all():
                raise ValueError("Feature row alignment differs")
        y = np.asarray([x["target"] for x in rows], dtype=np.float64)
        frozen = json.loads((args.author_root / "freeze.json").read_text())
        for name, expected in frozen["hashes"].items():
            if sha(args.author_root / name) != expected:
                raise ValueError("Author implementation changed")
        planned = allocations(rows)
        protocol = {
            "scope": "Retrospective strict-cohort reproduction of the published single-to-multiple design; two SaSrtA source conflicts excluded; not historical-seed agreement or autonomous agent comparison",
            "input_sha256": sha(args.admission / "campaign.json"), "feature_sha256": sha(args.features / "embeddings.npz"), "author_freeze_sha256": sha(args.author_root / "freeze.json"), "driver_sha256": sha(Path(__file__)),
            "split_policy": "80/20 single mutants for train/validation; every higher-order mutant held out; seeds 0-9 selected before fitting; exact sequence overlap rejected",
            "label_standardization": "Training mean and population standard deviation only; frozen transform for validation/future",
            "head": "published PLM_RankReg MLP, 1280->256->128->1, dropout 0",
            "training": {"epochs": 1000, "patience": 50, "batch_size": 1024, "lr": 3e-4, "wd": 1e-5, "alpha": 0.8, "margin": 0.001, "rank_num_samples": None, "selection": "author validation Spearman, MSE tie-break"},
            "random_forest": {"n_estimators": 100, "n_jobs": 12, "random_state": "matched seed"},
            "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "scipy", "scikit-learn"]},
            "allocations": planned,
        }
        write(args.output / "protocol.json", protocol)
        write(args.output / "row_identity.json", [{k: x[k] for k in ["id", "family", "round", "key", "mutation_order", "sequence_sha256"]} for x in rows])
        report["protocol_sha256"] = sha(args.output / "protocol.json")
        report["planned_fits"] = len(planned)
        write(args.output / "report.json", report)
        sys.path.insert(0, str(args.author_root))
        from reap.training import (
            evaluate_mse_spearman,
            load_plm_checkpoint,
            predict_plm_model,
            train_evolvepro,
            train_plm_model,
        )
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for allocation in planned:
            folder = args.output / allocation["id"]
            folder.mkdir()
            t = time.monotonic()
            tr, va, te = [allocation[k] for k in ["train", "validation", "future"]]
            mu, sd = float(y[tr].mean()), float(y[tr].std())
            if not np.isfinite(sd) or sd <= 0:
                raise ValueError("Degenerate training labels")
            yz = ((y - mu) / sd).astype(np.float32)
            if allocation["loss"] == "EVOLVEpro":
                _, _, model = train_evolvepro(X[tr], yz[tr], X[te], yz[te], seed=allocation["seed"], n_jobs=12)
                predictions = model.predict(X[te])
                val_predictions = model.predict(X[va])
                import joblib
                joblib.dump(model, folder / "model.joblib")
                restored = joblib.load(folder / "model.joblib").predict(X[te])
            else:
                _, _, _, model = train_plm_model(X[tr], yz[tr], X[va], yz[va], epochs=1000, seed=allocation["seed"], save_path=str(folder / "model.pt"), model_type="mlp", loss_name=allocation["loss"], alpha=0.8, margin=0.001, rank_num_samples=None, patience=50, batch_size=1024, lr=3e-4, wd=1e-5, device="cuda:0", checkpoint_metadata={"protocol_sha256": report["protocol_sha256"], "allocation": allocation["id"], "target_mean": mu, "target_std": sd})
                predictions = predict_plm_model(model, X[te])
                val_predictions = predict_plm_model(model, X[va])
                restored = predict_plm_model(load_plm_checkpoint(folder / "model.pt", X.shape[1], device="cuda:0"), X[te])
            if not np.allclose(predictions, restored, rtol=1e-6, atol=1e-7) or not np.isfinite(predictions).all():
                raise ValueError("Checkpoint replay differs")
            mse, rho = evaluate_mse_spearman(yz[te], predictions)
            vmse, vrho = evaluate_mse_spearman(yz[va], val_predictions)
            np.savez_compressed(folder / "predictions.npz", future_indices=np.asarray(te), predicted_standardized=predictions, predicted=predictions * sd + mu, observed=y[te], validation_indices=np.asarray(va), validation_predicted_standardized=val_predictions)
            result = {"id": allocation["id"], "family": allocation["family"], "seed": allocation["seed"], "loss": allocation["loss"], "status": "completed", "n_train": len(tr), "n_validation": len(va), "n_future": len(te), "target_mean": mu, "target_std": sd, "future_spearman": rho, "future_standardized_mse": mse, "validation_spearman": vrho, "validation_standardized_mse": vmse, "checkpoint_replayed": True, "prediction_sha256": sha(folder / "predictions.npz"), "wall_seconds": time.monotonic() - t}
            write(folder / "result.json", result)
            report["fits"].append(result)
            report["wall_seconds"] = time.monotonic() - start
            write(args.output / "report.json", report)
            print(len(report["fits"]), len(planned), allocation["id"], round(rho, 4), flush=True)
        report.update(status="completed", wall_seconds=time.monotonic() - start)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), wall_seconds=time.monotonic() - start)
        write(args.output / "report.json", report)
        raise
    write(args.output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["admission", "features", "author-root", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
