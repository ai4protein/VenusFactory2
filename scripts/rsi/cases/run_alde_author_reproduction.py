"""Reproduce full four-round ALDE campaigns with pinned author implementations.

A declared subset of the paper: one-hot, five-network ensemble, all three
acquisition rules, paired author seeds, both released complete landscapes.
CPU preserves the author's PyTorch version; no model source is rewritten.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMMIT = "d0b3593dd17987c9864830d36f5976b1ad04a619"
SOURCE = ROOT / "research_runs/rsi/case_assets/ALDE" / COMMIT / "source"
ACQUISITIONS = ["GREEDY", "UCB", "TS"]


def check_indices(values, total, expected=480):
    if len(values) != expected or any(type(x) is not int or not 0 <= x < total for x in values):
        raise ValueError("invalid author query trajectory")
    return {
        "query_events": len(values),
        "unique_sequences": len(set(values)),
        "initial_unique_sequences": len(set(values[:96])),
        "later_repeated_queries": len(values[96:]) - len(set(values[96:]) - set(values[:96])),
    }


def worker(args):
    # Imports resolve from the pinned author tree, not the VenusFactory src tree.
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import torch
    from src import utils
    from src.objectives import Combo
    from src.optimize import BO_ARGS, BayesianOptimization

    torch.set_num_threads(1)
    os.chdir(args.workspace)
    seed = int((SOURCE / "src/rndseed.txt").read_text().splitlines()[args.seed_index])
    obj = Combo(args.protein, "onehot")
    x, y = obj.get_points()
    start_x, start_y, start_indices = utils.samp_discrete(96, obj, seed)
    prefix = str(args.output / "author_")
    args.output.mkdir(parents=True, exist_ok=False)
    kwargs = BO_ARGS(
        mtype="DNN_ENSEMBLE",
        kernel="RBF",
        acq_fn=args.acquisition,
        xi=4,
        architecture=[x.shape[1], 30, 30, 1],
        activation="lrelu",
        min_noise=1e-6,
        trainlr=1e-3,
        train_iter=300,
        dropout=0,
        mcdropout=0,
        verbose=0,
        bb_fn=obj.objective,
        domain=obj.get_domain(),
        disc_X=x,
        disc_y=y,
        noise_std=0,
        n_rand_init=0,
        budget=384,
        query_cost=1,
        queries_x=start_x,
        queries_y=start_y,
        indices=start_indices,
        savedir=prefix,
        batch_size=96,
    )
    BayesianOptimization.run(kwargs, seed)
    indices = torch.load(prefix + "indices.pt", weights_only=True).numpy()
    if not np.isfinite(indices).all() or not np.equal(indices, indices.astype(int)).all():
        raise ValueError("non-integral query index")
    indices = indices.astype(int).tolist()
    audit = check_indices(indices, len(y))
    if indices[:96] != start_indices.tolist():
        raise ValueError("author initial sample changed")
    # Preserve the source driver's random control exactly, including repeated
    # indices caused by drawing with replacement and restarting the same seed.
    _, _, random_tail = utils.samp_discrete(384, obj, seed)
    author_random = start_indices.tolist() + random_tail.tolist()
    # A separately labelled equal-budget unique control addresses that behavior.
    rng = np.random.default_rng(seed)
    available = np.setdiff1d(np.arange(len(y)), np.unique(start_indices.numpy()))
    unique_random = start_indices.tolist() + rng.choice(available, size=384, replace=False).tolist()
    trajectories = {}
    for name, ids in [
        ("author_model", indices),
        ("author_random", author_random),
        ("unique_random_control", unique_random),
    ]:
        trajectories[name] = {
            **check_indices(ids, len(y)),
            "indices": ids,
            "best_normalized_fitness": [float(y[ids[:n]].max()) for n in [96, 192, 288, 384, 480]],
        }
    result = {
        "status": "completed",
        "protein": args.protein,
        "acquisition": args.acquisition,
        "seed_index": args.seed_index,
        "seed": seed,
        "author_query_audit": audit,
        "normalization": "author landscape maximum; reported retrospectively, not an unseen-target predictive test",
        "trajectories": trajectories,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2))


def prepare_data(output):
    import csv
    import hashlib

    import numpy as np
    import torch

    fitness = ROOT / "research_runs/rsi/case_assets/alde_landscapes"
    encodings = ROOT / "research_runs/rsi/case_assets/alde_onehot"
    fm = json.loads((fitness / "source_manifest.json").read_text())
    em = json.loads((encodings / "source_manifest.json").read_text())
    records = {}
    for protein in ["GB1", "TrpB"]:
        table, tensor = fitness / (protein + "_fitness.csv"), encodings / (protein + "_onehot_x.pt")
        ft = next(v for v in fm["files"] if v["local_file"] == table.name)
        et = next(v for v in em["files"] if v["file"] == tensor.name)
        for path, expected in [(table, ft["sha256"]), (tensor, et["sha256"])]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("author input checksum changed")
        with table.open() as stream:
            rows = list(csv.DictReader(stream))
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        seqs = [row["Combo"] for row in rows]
        x = torch.load(tensor, map_location="cpu", weights_only=True)
        if x.shape != (len(seqs), 80) or not torch.isfinite(x).all():
            raise ValueError("invalid encoding shape or values")
        blocks = x.reshape(-1, 4, 20)
        if not torch.all((blocks == 0) | (blocks == 1)) or not torch.all(blocks.sum(-1) == 1):
            raise ValueError("invalid one-hot block")
        decoded = ["".join(alphabet[i] for i in row) for row in blocks.argmax(-1).tolist()]
        if decoded != seqs or len(set(seqs)) != len(seqs):
            raise ValueError("encoding and fitness row order disagree")
        labels = np.array([float(row["fitness"]) for row in rows])
        if not np.isfinite(labels).all():
            raise ValueError("nonfinite author label")
        destination = output / "workspace/data" / protein
        destination.mkdir(parents=True)
        (destination / "fitness.csv").symlink_to(table)
        (destination / "onehot_x.pt").symlink_to(tensor)
        records[protein] = {
            "rows": len(rows),
            "zero_fitness_rows": int((labels == 0).sum()),
            "fitness_sha256": ft["sha256"],
            "encoding_sha256": et["sha256"],
            "encoding_dtype": str(x.dtype),
            "sequence_alignment_verified": True,
        }
    return records


def main(args):
    import hashlib

    if not 1 <= args.seeds <= 70:
        raise ValueError("predeclare 1..70 author seeds")
    # Exercise the optimizer, not just imports: old Torch requires pkg_resources.
    import torch

    probe = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.Adam(probe.parameters())
    probe(torch.ones(2, 2).double()).square().mean().backward()
    optimizer.step()
    if len(set(args.acquisitions)) != len(args.acquisitions):
        raise ValueError("duplicate acquisition condition")
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "research_runs/rsi")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.rglob("*.py"))
    }
    for name in ["src/rndseed.txt", "alde.yml"]:
        source_hashes[name] = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
    seed_values = [int(v) for v in (SOURCE / "src/rndseed.txt").read_text().splitlines()[: args.seeds]]
    allocations = [
        {"protein": p, "acquisition": a, "seed_index": s}
        for s in range(args.seeds)
        for p in ["GB1", "TrpB"]
        for a in (args.acquisitions if s % 2 == 0 else args.acquisitions[::-1])
    ]
    protocol = {
        "schema": 1,
        "author_repository": "https://github.com/jsunn-y/ALDE",
        "commit": COMMIT,
        "allocations": allocations,
        "seeds": seed_values,
        "source_hashes": source_hashes,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": prepare_data(args.output),
        "initial_queries": 96,
        "rounds": 4,
        "batch_size": 96,
        "model": "author DNN_ENSEMBLE",
        "ensemble_size": 5,
        "bootstrap_size": 0.9,
        "architecture": [80, 30, 30, 1],
        "max_training_iterations": 300,
        "scope": "complete temporal computational campaigns for declared one-hot/DNN subset; not all encodings/models/70 seeds or new wet-lab replication",
        "agent_orchestrated": False,
        "environment": {
            d: importlib.metadata.version(d)
            for d in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "botorch", "gpytorch", "pyro-ppl"]
        },
        "environment_difference": "CPU build of author Torch 2.1.1/Python 3.11, selected compatible versions for dependencies unpinned by author",
        "author_sampling_audit": "initial sampling uses replacement; author's random tail restarts same seed; preserve exactly and add labelled unique-tail control",
        "timeout_s_per_arm": 3600,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    shutil.copy2(__file__, args.output / "implementation.py")
    report = {"status": "running", "arms": [], "full_paper_reproduced": False}

    def save():
        temp = args.output / ".report.tmp"
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.output / "report.json")

    save()
    for allocation in allocations:
        if any(
            hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != checksum
            for name, checksum in source_hashes.items()
        ):
            raise ValueError("author source drift")
        folder = args.output / allocation["protein"] / str(allocation["seed_index"]) / allocation["acquisition"]
        folder.parent.mkdir(parents=True, exist_ok=True)
        log = folder.parent / (allocation["acquisition"] + ".log")
        entry = {**allocation, "status": "running"}
        report["arms"].append(entry)
        save()
        print(json.dumps(entry), flush=True)
        started = time.monotonic()
        argv = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--output",
            str(folder),
            "--workspace",
            str(args.output / "workspace"),
            "--protein",
            allocation["protein"],
            "--acquisition",
            allocation["acquisition"],
            "--seed-index",
            str(allocation["seed_index"]),
        ]
        try:
            with log.open("w") as stream:
                result = subprocess.run(
                    argv,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    env={
                        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                        "HOME": str(args.output / "workspace"),
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "CUDA_VISIBLE_DEVICES": "",
                    },
                )
            if result.returncode != 0:
                raise ValueError("author worker failed; inspect archived log")
            outcome = json.loads((folder / "report.json").read_text())
            entry.update(status="completed", result=outcome)
        except Exception as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
        entry["wall_seconds"] = round(time.monotonic() - started, 3)
        save()
        print(json.dumps({k: v for k, v in entry.items() if k != "result"}), flush=True)
    report["status"] = "campaign_completed"
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--acquisitions", nargs="+", choices=ACQUISITIONS, default=ACQUISITIONS)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--protein", choices=["GB1", "TrpB"])
    parser.add_argument("--acquisition", choices=ACQUISITIONS)
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
