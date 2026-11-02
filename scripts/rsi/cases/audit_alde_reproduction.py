"""Independently recompute ALDE trajectories and compare source aggregates."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def curve(indices, labels):
    ids = np.asarray(indices)
    if ids.ndim != 1 or len(ids) != 480 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("expected 480 integer query indices")
    if np.any(ids < 0) or np.any(ids >= len(labels)):
        raise ValueError("index outside released landscape")
    y = np.asarray(labels, dtype=float)
    if not np.isfinite(y).all() or y.max() <= 0:
        raise ValueError("invalid landscape fitness")
    return np.maximum.accumulate(y[ids] / y.max())[[95, 191, 287, 383, 479]]


def main(args):
    import torch

    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    if report["status"] != "campaign_completed" or len(report["arms"]) != len(protocol["allocations"]):
        raise ValueError("campaign is not terminal with all allocations")
    source = ROOT / "research_runs/rsi/case_assets/ALDE" / protocol["commit"] / "source"
    published_path = source / "analysis/all_results.csv"
    with published_path.open() as stream:
        published = list(csv.DictReader(stream))
    args.output.mkdir(parents=True, exist_ok=False)
    labels = {}
    for protein, identity in protocol["inputs"].items():
        path = ROOT / "research_runs/rsi/case_assets/alde_landscapes" / (protein + "_fitness.csv")
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity["fitness_sha256"]:
            raise ValueError("fitness input changed")
        with path.open() as stream:
            labels[protein] = np.array([float(row["fitness"]) for row in csv.DictReader(stream)])
    groups = {}
    audited = []
    for arm in report["arms"]:
        if arm["status"] != "completed":
            audited.append({k: v for k, v in arm.items() if k != "result"})
            continue
        protein, acquisition, seed_index = (arm[k] for k in ["protein", "acquisition", "seed_index"])
        folder = run / protein / str(seed_index) / acquisition
        original = torch.load(folder / "author_indices.pt", map_location="cpu", weights_only=True).numpy()
        if not np.equal(original, original.astype(int)).all():
            raise ValueError("non-integral author index artifact")
        trajectories = arm["result"]["trajectories"]
        if original.astype(int).tolist() != trajectories["author_model"]["indices"]:
            raise ValueError("reported query indices disagree with saved author artifact")
        for name, trajectory in trajectories.items():
            values = curve(trajectory["indices"], labels[protein])
            if not np.allclose(values, trajectory["best_normalized_fitness"], rtol=0, atol=1e-12):
                raise ValueError("reported trajectory disagrees with independent label lookup")
            key = (protein, acquisition if name == "author_model" else name)
            by_seed = groups.setdefault(key, {})
            # Random controls repeat across acquisition conditions: count each
            # seed once, never inflate n from three identical control copies.
            if seed_index in by_seed and by_seed[seed_index] != values.tolist():
                raise ValueError("paired control differs across acquisitions")
            by_seed[seed_index] = values.tolist()
        audited.append(
            {
                "protein": protein,
                "acquisition": acquisition,
                "seed_index": seed_index,
                "status": "verified",
                "unique_queries": len(set(original.tolist())),
                "indices_sha256": hashlib.sha256((folder / "author_indices.pt").read_bytes()).hexdigest(),
            }
        )
    summaries = []
    for (protein, condition), values in groups.items():
        matrix = np.array(list(values.values()))
        summary = {
            "protein": protein,
            "condition": condition,
            "n_seeds": len(matrix),
            "seed_indices": sorted(values),
            "per_seed": values,
            "mean": matrix.mean(0).tolist(),
            "sample_sd": matrix.std(0, ddof=1).tolist() if len(matrix) > 1 else None,
            "sem": (matrix.std(0, ddof=1) / np.sqrt(len(matrix))).tolist() if len(matrix) > 1 else None,
        }
        if condition in ["GREEDY", "UCB", "TS"]:
            matches = [
                v
                for v in published
                if v["Protein"] == protein
                and v["Encoding"] == "onehot"
                and v["Model"] == "DNN Ensemble"
                and v["Acquisition"] == condition
                and int(v["Timestep"]) in [96, 192, 288, 384, 480]
            ]
            matches.sort(key=lambda v: int(v["Timestep"]))
            if len(matches) != 5:
                raise ValueError("published summary points missing")
            summary["published_mean"] = [float(v["Mean"]) for v in matches]
            summary["published_sem"] = [float(v["Std"]) for v in matches]
            summary["endpoint_difference_to_published_mean"] = summary["mean"][-1] - summary["published_mean"][-1]
        summaries.append(summary)
    result = {
        "status": "completed",
        "campaign_report_sha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
        "source_summary_sha256": hashlib.sha256(published_path.read_bytes()).hexdigest(),
        "allocations": len(protocol["allocations"]),
        "completed_and_verified": sum(a["status"] == "verified" for a in audited),
        "query_budgets": [96, 192, 288, 384, 480],
        "arms": audited,
        "summaries": summaries,
        "published_spread_semantics": "column Std is SEM: verified from author load_tensors and tabulate_regret notebook functions",
        "scope": "complete author four-round onehot/DNN campaigns over declared seeds; subset reproduction, not full publication or autonomous-agent execution",
        "comparison_limit": "declared author seed subset versus published all-run aggregates; without per-seed author artifacts, mean differences are not exact trajectory replication error or significance tests",
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["status", "allocations", "completed_and_verified"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
