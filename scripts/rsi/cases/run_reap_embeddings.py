"""Execute the published REAP embedding function against verified local assets.

This is controller-run feature computation, not evaluated-agent orchestration.
The independent author package is provided as an isolated versioned snapshot.
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"status": "running", "scope": "Controller-run author embedding computation on the strict replicate-verified cohort; not autonomous orchestration, author-training agreement or efficacy evidence"}
    try:
        rows = json.loads((args.admission / "campaign.json").read_text())
        summary = json.loads((args.admission / "summary.json").read_text())
        if len(rows) != summary["admitted_campaign"] or len({x["id"] for x in rows}) != len(rows):
            raise ValueError("Admission count or identity disagreement")
        freeze = json.loads((args.author_root / "freeze.json").read_text())
        for name, expected in freeze["hashes"].items():
            if sha(args.author_root / name) != expected:
                raise ValueError("Author source changed: " + name)
        checkpoint = args.checkpoint.resolve()
        contact = checkpoint.with_name(checkpoint.stem + "-contact-regression.pt")
        if not checkpoint.is_file() or not contact.is_file():
            raise ValueError("Both local ESM2 checkpoints required; no network fallback")
        report.update({
            "records": len(rows), "input_sha256": sha(args.admission / "campaign.json"),
            "admission_summary_sha256": sha(args.admission / "summary.json"),
            "driver_sha256": sha(Path(__file__)), "author_freeze_sha256": sha(args.author_root / "freeze.json"),
            "checkpoints": {str(p): sha(p) for p in (checkpoint, contact)},
            "batch_size": args.batch_size, "device": args.device, "dtype": "float32",
            "pooling": "author function; final layer 33, residues only, no truncation", "truncation": False,
            "lengths": sorted({len(x["sequence"]) for x in rows}),
            "packages": {name: importlib.metadata.version(name) for name in ["torch", "fair-esm", "numpy", "pandas"]},
        })
        write(args.output / "report.json", report)
        sys.path.insert(0, str(args.author_root))
        import esm
        from reap.embeddings import extract_esm2_embedding
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint)
        model.eval().float().to(args.device)
        converter = alphabet.get_batch_converter()
        chunks = []
        for offset in range(0, len(rows), args.batch_size):
            batch_rows = rows[offset:offset + args.batch_size]
            batch = [(row["id"], row["sequence"]) for row in batch_rows]
            X = extract_esm2_embedding(batch, model, converter, torch.device(args.device), repr_layers=33)
            if X.shape != (len(batch), 1280) or not np.isfinite(X).all():
                raise ValueError("Invalid author embedding output")
            chunks.append(X)
            if offset % 32 == 0 or offset + len(batch) == len(rows):
                report.update(completed=offset + len(batch), wall_seconds=time.monotonic() - start)
                write(args.output / "report.json", report)
                print(report["completed"], len(rows), round(report["wall_seconds"], 2), flush=True)
        features = np.concatenate(chunks)
        np.savez_compressed(args.output / "embeddings.npz", X=features, ids=np.asarray([x["id"] for x in rows]), sequence_sha256=np.asarray([x["sequence_sha256"] for x in rows]))
        with np.load(args.output / "embeddings.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["X"], features) or saved["ids"].tolist() != [x["id"] for x in rows]:
                raise ValueError("Saved feature identity or numerical round trip differs")
        report.update(status="completed", shape=list(features.shape), output_sha256=sha(args.output / "embeddings.npz"), wall_seconds=time.monotonic() - start)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), wall_seconds=time.monotonic() - start)
        write(args.output / "report.json", report)
        raise
    write(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Positive batch size required")
    main(args)
