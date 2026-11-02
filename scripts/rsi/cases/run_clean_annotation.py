"""New CPU ESM-1b embeddings and both released CLEAN heads on all three cohorts.

This software uses or is derived from the CLEAN software developed by the
Department of Chemical and Biomolecular Engineering at the University of
Illinois Urbana-Champaign. This is a reconstruction wrapper, not an author modification.
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
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import clean_annotation as case  # noqa: E402


def run(output):
    output = output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    output.mkdir(parents=True, exist_ok=False)
    assets = ROOT / "research_runs/rsi/case_assets/CLEAN"
    report = {
        "status": "running",
        "case": "clean_sequence_annotation.v1",
        "source_commit": case.COMMIT,
        "execution_origin": "new_sequence_embedding_and_author_inference",
        "full_paper_reproduced": False,
        "original_clean_training_reproduced": False,
        "new_model_developed": False,
        "asset_manifest_sha256": case.MANIFEST_SHA256,
        "allocation": case.ALLOCATION,
        "completed_embedding_records": 0,
        "results": {},
        "artifact_sha256": {},
    }
    started = time.monotonic()
    case.save(output / "report.json", report)
    try:
        versions = {k: importlib.metadata.version(k) for k in case.ENVIRONMENT}
        if versions != case.ENVIRONMENT:
            raise ValueError("Pinned CLEAN reconstruction environment required")
        manifest = case.verify_assets(assets)
        encoder = manifest["encoder"]
        for item in encoder.values():
            if case.sha(Path(item["path"])) != item["sha256"]:
                raise ValueError("Pinned ESM-1b encoder changed")
        import esm
        import numpy as np
        import pandas as pd
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(1234)
        sys.path.insert(0, str(assets / "source/app/src"))
        from CLEAN.distance_map import get_dist_map_test
        from CLEAN.evaluate import get_eval_metrics, get_pred_probs, write_max_sep_choices
        from CLEAN.model import LayerNormNet

        sources = [Path(__file__), Path(case.__file__)]
        for path in sources:
            shutil.copy2(path, output / path.name)
        source_hashes = {p.name: case.sha(p) for p in sources}
        cohorts = {n: case.dataset(assets, n) for n in case.COHORTS}
        heads = {}
        for name in case.MODELS:
            model = LayerNormNet(512, 128, torch.device("cpu"), torch.float32)
            state = torch.load(assets / "pretrained" / (name + ".pth"), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            train = case.read_rows(assets / "source/app/data" / (name + ".csv"))
            index = case.class_index(train)
            embeddings = torch.load(
                assets / "pretrained" / (name.removeprefix("split") + ".pt"), map_location="cpu", weights_only=True
            ).detach()
            case.centers(embeddings.numpy(), index)
            case.forward({k: v.detach().numpy() for k, v in state.items()}, np.zeros((1, 1280)))
            heads[name] = (model, train, index, embeddings)
        original_price, _ = case.read_predictions(
            assets / "source/app/results/inputs/price_maxsep.csv", [r["Entry"] for r in cohorts["price"]]
        )
        protocol = {
            "source_commit": case.COMMIT,
            "asset_manifest_sha256": case.MANIFEST_SHA256,
            "environment": versions,
            "python": sys.version,
            "allocation": report["allocation"],
            "encoder": encoder,
            "device": "cpu",
            "threads": 4,
            "seed": 1234,
            "representation": "ESM-1b layer 33 mean over all residues, without BOS/EOS; no truncation",
            "label_rule": "Author maximum separation; first_grad=True, use_max_grad=False, gmm=None",
            "source_hashes": source_hashes,
            "audit_boundary": "Independent post-embedding NumPy network, EC center, distance, label and metric replay; no second ESM transformer run",
            "reference_policy": "Retain all mismatches to saved author predictions; do not tune on the test cohorts",
            "omissions": [
                "original contrastive training and cross-validation",
                "p-value inference",
                "wet-lab enzyme mining",
                "causal RSI benefit",
            ],
        }
        case.save(output / "protocol.json", protocol)
        with torch.serialization.safe_globals([argparse.Namespace]):
            esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(encoder["model"]["path"])
        esm_model.eval()
        converter = alphabet.get_batch_converter()
        for name, rows in cohorts.items():
            features = []
            for row in rows:
                seq = row["Sequence"]
                _, _, tokens = converter([(row["Entry"], seq)])
                if tokens.shape != (1, len(seq) + 2):
                    raise ValueError("ESM tokenization changed the declared sequence span")
                with torch.inference_mode():
                    result = esm_model(tokens, repr_layers=[33], return_contacts=False)
                vector = result["representations"][33][0, 1 : len(seq) + 1].mean(dim=0).numpy()
                if vector.shape != (1280,) or not np.isfinite(vector).all():
                    raise ValueError("Invalid ESM result")
                features.append(vector)
                report["completed_embedding_records"] += 1
                if report["completed_embedding_records"] % 20 == 0:
                    case.save(output / "report.json", report)
                    print(
                        json.dumps({"stage": "esm", "completed": report["completed_embedding_records"], "total": 578}),
                        flush=True,
                    )
            np.save(output / (name + ".esm.npy"), np.stack(features), allow_pickle=False)
            case.save(
                output / (name + ".ids.json"),
                [
                    {
                        "id": r["Entry"],
                        "sequence_sha256": hashlib.sha256(r["Sequence"].encode()).hexdigest(),
                        "length": len(r["Sequence"]),
                    }
                    for r in rows
                ],
            )
        del esm_model
        import gc

        gc.collect()
        for model_name in case.MODELS:
            model, train, index, embeddings = heads[model_name]
            train_ids = {r["Entry"] for r in train}
            train_sequences = {r["Sequence"] for r in train}
            for name, rows in cohorts.items():
                prefix = model_name + "." + name
                means = np.load(output / (name + ".esm.npy"), allow_pickle=False)
                ids = {r["Entry"]: r["EC number"].split(";") for r in rows}
                with torch.inference_mode():
                    projected = model(torch.from_numpy(means))
                    mapping = get_dist_map_test(embeddings, projected, index, ids, torch.device("cpu"), torch.float32)
                frame = pd.DataFrame.from_dict(mapping)
                if list(frame.columns) != list(ids) or list(frame.index) != list(index):
                    raise ValueError("Author distance identities changed")
                dist = frame.to_numpy().T
                np.save(output / (prefix + ".distances.npy"), dist, allow_pickle=False)
                write_max_sep_choices(frame, str(output / prefix), first_grad=True, use_max_grad=False, gmm=None)
                labels, _ = case.read_predictions(output / (prefix + "_maxsep.csv"), list(ids))
                if labels != case.select_labels(dist, list(index)):
                    raise ValueError("Independent label selector differs")
                truth = list(ids.values())
                all_labels = set().union(*map(set, truth))
                probabilities = get_pred_probs(str(output / prefix), pred_type="_maxsep")
                author_scores = get_eval_metrics(labels, probabilities, truth, all_labels)
                scores = dict(
                    zip(
                        ["weighted_precision", "weighted_recall", "weighted_f1", "weighted_auc", "exact_match"],
                        map(float, author_scores),
                    )
                )
                full = case.metrics(truth, labels)
                record = {
                    "rows": len(rows),
                    "author_truth_only_label_space_metrics": scores,
                    "full_union_metrics": full,
                    "train_id_overlap": sum(r["Entry"] in train_ids for r in rows),
                    "train_exact_sequence_overlap": sum(r["Sequence"] in train_sequences for r in rows),
                }
                if name == "price":
                    record["released_price_prediction_comparison"] = {
                        "same_label_records": sum(set(a) == set(b) for a, b in zip(original_price, labels)),
                        "rows": len(rows),
                        "reference_training_head_not_declared_in_file": True,
                    }
                report["results"][prefix] = record
                case.save(output / "report.json", report)
        report["artifact_sha256"] = {
            p.name: case.sha(p) for p in output.iterdir() if p.name != "report.json" and p.is_file()
        }
        if source_hashes != {p.name: case.sha(p) for p in sources}:
            raise ValueError("CLEAN reconstruction implementation changed")
        report["status"] = "completed"
        report["independent_replay"] = case.verify_artifacts(report, output, assets)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        case.save(output / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    r = run(args.output)
    print(
        json.dumps(
            {
                "status": r["status"],
                "allocations": len(r["results"]),
                "embedding_records": r["completed_embedding_records"],
            }
        )
    )
