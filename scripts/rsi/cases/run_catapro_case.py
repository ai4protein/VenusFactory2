"""Replay CataPro samples or complete released CV cohorts from pinned assets."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import csv
import fcntl
import gc
import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from agent.rsi.cases.catapro_case import (
    COLUMNS,
    DATASETS,
    FeatureBuilder,
    FeatureCache,
    author_models,
    checked_manifest,
    dataset_rows,
    encoder_binding,
    file_sha,
    numpy_forward,
)


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def metrics(prediction, target):
    from scipy.stats import pearsonr, spearmanr

    return {
        "rows": len(target),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "pcc": float(pearsonr(prediction, target).statistic),
        "scc": float(spearmanr(prediction, target).statistic),
    }


@contextmanager
def gpu_lease(path):
    inherited = os.environ.get("VENUS_RSI_CASE_LOCK_FD")
    if inherited is not None:
        fd = int(inherited)
        actual, expected = os.fstat(fd), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("Inherited GPU lease points to a different file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The broker owns this open file description; never unlock or close it.
        yield
    else:
        with path.open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


def main(args):
    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Released ActivityModel constructs CUDA submodules; choose an available CUDA device")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "preparing", "dataset": args.dataset, "started_unix": time.time()}
    write(out / "report.json", report)
    source = args.source.resolve()
    cache = None
    try:
        admitted = checked_manifest(source, "admission.json")
        checkpoints = checked_manifest(source, "checkpoints_admission_api.json")
        molecule_admission = json.loads((args.molecule_encoder / "admission.json").read_text())
        if molecule_admission["status"] != "admitted":
            raise ValueError("Molecular encoder admission incomplete")
        binding = {
            "protein": encoder_binding(args.protein_encoder),
            "molecule": encoder_binding(args.molecule_encoder),
            "pooling": "catapro.v1: first/last500; UZOB->X; exclude EOS; protein float64 mean; molecule float32 mean",
            "rdkit_version": __import__("rdkit").__version__,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "precision": "float32 encoders, TF32 disabled, single-input encoding",
            "implementation_sha256": file_sha(ROOT / "src/agent/rsi/cases/catapro_case.py"),
        }
        for row in molecule_admission["files"]:
            if binding["molecule"]["files"][row["rfilename"]] != row["sha256"]:
                raise ValueError("Molecular encoder hash differs from admission")
        rows = dataset_rows(source, args.dataset)
        protocol = {
            "source_commit": admitted["commit"],
            "source_admission_sha256": file_sha(source / "admission.json"),
            "checkpoint_admission_sha256": file_sha(source / "checkpoints_admission_api.json"),
            "checkpoints": {r["path"]: r["sha256"] for r in checkpoints["files"]},
            "feature_contract": binding,
            "rows": len(rows),
            "dataset": args.dataset,
            "input_sha256": file_sha(
                source
                / ("samples/sample_inp.csv" if args.dataset == "sample" else "datasets/" + DATASETS[args.dataset])
            ),
            "output_units": COLUMNS,
            "target_conversion": "Raw Km(M) multiplied by 1000 before log10; activity=log10(kcat)-log10(Km_mM), inferred from released output headers; original training-feature construction is absent.",
            "scope": "Released computation replay. CV folds also selected original checkpoints, so this is not an unbiased new holdout or autonomous model discovery.",
            "tolerances": {
                "numpy_checkpoint_atol": 1e-4,
                "numpy_checkpoint_rtol": 1e-5,
                "sample_feature_atol": 5e-5,
                "published_sample_atol": 1e-4,
            },
        }
        write(out / "protocol.json", protocol)
        for original in [Path(__file__), ROOT / "src/agent/rsi/cases/catapro_case.py"]:
            dest = out / "implementation" / original.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(original.read_bytes())
        with gpu_lease(ROOT / "research_runs/rsi/.case_gpu.lock"):
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            cache = FeatureCache(args.cache, binding)
            builder = FeatureBuilder(args.protein_encoder, args.molecule_encoder, cache, args.device)
            features = np.empty((len(rows), 1959), dtype=np.float32)
            report.update(status="encoding", rows=len(rows), rows_encoded=0)
            write(out / "report.json", report)
            for i, row in enumerate(rows):
                features[i] = builder.row(row)
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    report["rows_encoded"] = i + 1
                    write(out / "report.json", report)
                    print(json.dumps({"rows_encoded": i + 1, "rows": len(rows)}), flush=True)
            np.save(out / "features.npy", features, allow_pickle=False)
            del builder
            cache.close()
            cache = None
            gc.collect()
            torch.cuda.empty_cache()
            if args.dataset == "sample":
                spec = importlib.util.spec_from_file_location("catapro_reference_utils", source / "inference/utils.py")
                reference = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(reference)
                reference_features = np.concatenate(
                    [
                        reference.Seq_to_vec([r["sequence"] for r in rows], str(args.protein_encoder)),
                        reference.get_molT5_embed([r["smiles"] for r in rows], str(args.molecule_encoder)),
                        reference.GetMACCSKeys([r["smiles"] for r in rows]),
                    ],
                    axis=1,
                ).astype(np.float32)
                feature_check = {
                    "max_abs_difference": float(np.max(np.abs(features - reference_features))),
                    "passed": bool(np.allclose(features, reference_features, atol=5e-5, rtol=1e-5)),
                    "comparison": "Optimized cached implementation versus unmodified released feature functions, same pinned encoders",
                }
                write(out / "feature_parity.json", feature_check)
                if not feature_check["passed"]:
                    raise ValueError("Released feature function replay differs")
            modules = author_models(source)
            kinds = ("kcat", "km", "activity") if args.dataset == "sample" else (args.dataset,)
            predictions = np.full((len(rows), len(kinds)), np.nan, dtype=np.float32)
            per_fold = []
            folders = {"kcat": "kcat_models", "km": "Km_models", "activity": "act_models"}
            constructors = {
                "kcat": modules["model"].KcatModel,
                "km": modules["model"].KmModel,
                "activity": modules["act_model"].ActivityModel,
            }
            for k, kind in enumerate(kinds):
                ensemble = []
                for fold in range(10):
                    indices = (
                        np.arange(len(rows))
                        if args.dataset == "sample"
                        else np.array([i for i, r in enumerate(rows) if r["fold"] == fold])
                    )
                    data = features[indices]
                    state = torch.load(
                        source / "models" / folders[kind] / f"{fold}_bestmodel.pth",
                        map_location="cpu",
                        weights_only=True,
                    )
                    model = constructors[kind](device=args.device)
                    model.load_state_dict(state, strict=True)
                    model.eval()
                    values = []
                    with torch.no_grad():
                        for start in range(0, len(data), 512):
                            tensor = torch.from_numpy(data[start : start + 512]).to(args.device)
                            pred = model(tensor[:, :1024], tensor[:, 1024:])
                            if kind == "activity":
                                pred = pred[-1]
                            values.append(pred.cpu().numpy().ravel())
                    pred = np.concatenate(values)
                    independent = np.concatenate(
                        [
                            numpy_forward(data[start : start + 512], state, activity=kind == "activity")
                            for start in range(0, len(data), 512)
                        ]
                    )
                    parity = bool(np.allclose(pred, independent, atol=1e-4, rtol=1e-5))
                    per_fold.append(
                        {
                            "kind": kind,
                            "fold": fold,
                            "rows": len(indices),
                            "independent_numpy_passed": parity,
                            "max_abs_difference": float(np.max(np.abs(pred - independent))),
                        }
                    )
                    if not parity:
                        write(out / "checkpoint_parity.json", per_fold)
                        raise ValueError("Independent checkpoint replay failed")
                    if args.dataset == "sample":
                        ensemble.append(pred)
                    else:
                        predictions[indices, k] = pred
                    del model, state
                    torch.cuda.empty_cache()
                if args.dataset == "sample":
                    predictions[:, k] = np.mean(np.stack(ensemble, axis=1), axis=1)
            write(out / "checkpoint_parity.json", per_fold)
            if not np.isfinite(predictions).all():
                raise ValueError("Incomplete/nonfinite predictions")
            np.save(out / "predictions.npy", predictions, allow_pickle=False)
            if args.dataset == "sample":
                with (source / "inference/catapro_test-pred.csv").open() as stream:
                    reference_rows = list(csv.DictReader(stream))
                if len(reference_rows) != len(rows) or any(
                    a["id"] != b["fasta_id"] or a["smiles"] != b["smiles"]
                    for a, b in zip(rows, reference_rows, strict=True)
                ):
                    raise ValueError("Published sample rows do not match")
                published = np.array([[float(r[c]) for c in COLUMNS] for r in reference_rows])
                result = {
                    "published_sample_matches": bool(np.allclose(predictions, published, atol=1e-4, rtol=1e-5)),
                    "max_abs_difference": float(np.max(np.abs(predictions - published))),
                    "per_output_max_abs_difference": np.max(np.abs(predictions - published), axis=0).tolist(),
                }
            else:
                target = np.array([r["targets"][args.dataset] for r in rows])
                result = {
                    "metrics": metrics(predictions[:, 0], target),
                    "fold_metrics": {
                        str(f): metrics(
                            predictions[[r["fold"] == f for r in rows], 0], target[[r["fold"] == f for r in rows]]
                        )
                        for f in range(10)
                    },
                }
            write(out / "results.json", result)
            report.update(
                status="completed",
                result=result,
                features_sha256=file_sha(out / "features.npy"),
                predictions_sha256=file_sha(out / "predictions.npy"),
                finished_unix=time.time(),
            )
            write(out / "report.json", report)
            print(json.dumps({"status": report["status"], "result": result}), flush=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, message=str(exc)[:500], finished_unix=time.time())
        write(out / "report.json", report)
        raise
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protein-encoder", type=Path, required=True)
    parser.add_argument("--molecule-encoder", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["sample", *DATASETS], required=True)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
