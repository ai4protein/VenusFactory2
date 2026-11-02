"""Recompute author NucB diversity curves with exact dendrogram reuse.

Each distinct ordered bootstrap sample is clustered once. Reusing that same
complete-linkage tree across cut heights preserves the author's numerical method.
The unseeded G1 input permutation is explicitly fixed, so exact historical G1
bootstrap agreement is not assumed.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
import time
from pathlib import Path

from prepare_nucb_assets import fingerprints, write
from run_nucb_author_reproduction import verify_source


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = verify_source(args.assets)
    cache = json.loads(args.asset_report.read_text())
    if cache["status"] != "verified":
        raise ValueError("Unverified data snapshot")
    hashes = {x["path"]: x["sha256"] for x in cache["files"]}
    paths = [f"processed_data/g{i}.csv" for i in range(1, 5)] + [
        "analysis/num_clusters_g4_df.csv",
        "analysis/num_clusters_zero_shot_df.csv",
    ]
    for name in paths:
        if fingerprints(args.assets / "gcs" / name)[0] != hashes[name]:
            raise ValueError("Frozen author data changed")
    protocol = {
        "source": source,
        "inputs": {p: hashes[p] for p in paths},
        "runtime": {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "statsmodels"]},
        "radius_g4": [0, 25],
        "radius_g1": [0, 20],
        "bootstraps": 10,
        "author_bootstrap_seed": 0,
        "g1_input_permutation_seed": 0,
        "g1_reference_caveat": "Original notebook permutes G1 rows without a seed; exact saved bootstrap samples are unavailable",
        "optimization": "Reuse identical complete-linkage dendrogram for an identical ordered bootstrap sample across cut heights",
        "spot_checks": "First 12 unique ordered samples compared with original author num_sequence_clusters at encountered radius",
        "entry_sha256": fingerprints(Path(__file__))[0],
        "agent_orchestrated": False,
    }
    dependencies = [
        Path(__file__),
        Path(__file__).with_name("run_nucb_author_reproduction.py"),
        Path(__file__).with_name("prepare_nucb_assets.py"),
        Path(__file__).with_name("nucb_reproduction_checks.py"),
    ]
    protocol["implementation"] = {p.name: fingerprints(p)[0] for p in dependencies}
    for path in dependencies:
        dest = args.output / "implementation" / path.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
    write(args.output / "protocol.json", protocol)
    sys.path.insert(0, str(args.assets / "source"))
    import numpy as np
    import pandas as pd
    from nuclease_design import metrics, utils
    from scipy.cluster import hierarchy
    from scipy.spatial import distance

    data = utils.load_all_data(str(args.assets / "gcs"))
    df = utils.expand_sublibraries(data)
    df["campaign"] = df["sublibrary_name"].map(lambda s: "HR" if s.endswith("recombination") else "ML")
    nb = json.loads((args.assets / "source/notebooks/compute_sequence_clusters.ipynb").read_text())
    ns = {"np": np, "pd": pd, "utils": utils}
    exec(compile("".join(nb["cells"][10]["source"]), "author-cluster-notebook:cell-10", "exec"), ns)
    original = utils.num_sequence_clusters
    trees, spot_checks = {}, []

    def cached_clusters(sequences, max_intra_cluster_hamming_distance):
        key = tuple(sequences)
        if key not in trees:
            encoded = np.vstack([utils._encode_sequence(s) for s in key])
            pairwise = metrics.pairwise_hamming_distance(encoded)
            trees[key] = hierarchy.linkage(distance.squareform(pairwise, checks=False), method="complete")
            if len(spot_checks) < 12:
                count = int(
                    np.max(hierarchy.fcluster(trees[key], t=max_intra_cluster_hamming_distance, criterion="distance"))
                )
                expected = int(original(sequences, max_intra_cluster_hamming_distance))
                if count != expected:
                    raise ValueError("Dendrogram reuse differs from author calculation")
                spot_checks.append(
                    {
                        "sample_rows": len(key),
                        "radius": max_intra_cluster_hamming_distance,
                        "clusters": count,
                        "matched": True,
                    }
                )
        return np.max(hierarchy.fcluster(trees[key], t=max_intra_cluster_hamming_distance, criterion="distance"))

    utils.num_sequence_clusters = cached_clusters
    g4 = df[df["sublibrary_name"].isin(["g4_mbo_dnn", "g4_hit_recombination"])]
    np.random.seed(0)
    g1 = df[df.generation == "g1"].reset_index(drop=True).sample(frac=1.0)
    zero = pd.concat([df[df["sublibrary_name"] == "zero_shot"], g1])
    report = {"status": "running", "datasets": [], "agent_orchestrated": False}
    write(args.output / "report.json", report)
    try:
        for name, frame, hue, radii, references, reference_file in [
            ("g4", g4, "campaign", range(26), ["neg_control", "wt", "a73r"], "analysis/num_clusters_g4_df.csv"),
            (
                "zero_shot",
                zero,
                "sublibrary_name",
                range(21),
                ["neg_control", "wt"],
                "analysis/num_clusters_zero_shot_df.csv",
            ),
        ]:
            rows = []
            item = {"dataset": name, "status": "running", "completed_radii": [], "input_rows": len(frame)}
            report["datasets"].append(item)
            for radius in radii:
                result = ns["get_num_clusters_df"](
                    frame,
                    [
                        {"reference_name": ref, "expected_false_discovery_rate": utils.EXPECTED_FDR}
                        for ref in references
                    ],
                    max_intra_cluster_hamming_distance=radius,
                    hue_feature=hue,
                )
                rows.append(result)
                item["completed_radii"].append(radius)
                report["unique_trees"] = len(trees)
                write(args.output / "report.json", report)
                print(json.dumps({"dataset": name, "radius": radius, "trees": len(trees)}), flush=True)
            table = pd.concat(rows, ignore_index=True)
            table["activity_label"] = table["activity_label"].map(lambda x: "pre-sort" if x == "all" else x)
            table["population"] = (
                table["activity_label"]
                .astype(str)
                .str.replace("a73r", "A73R")
                .str.replace("neg_control", "negative control")
                .str.replace("wt", "parent")
            )
            dest = args.output / (name + "_clusters.csv")
            table.to_csv(dest, index=False)
            expected = pd.read_csv(args.assets / "gcs" / reference_file)
            shared_columns = sorted(set(table) & set(expected))
            # Full rows (including bootstrap replicate ids) are compared, not just
            # the mean curve; shape and field differences cannot be discarded.
            same_schema = set(table) == set(expected) and len(table) == len(expected)
            a = table[shared_columns].sort_values(shared_columns).reset_index(drop=True)
            b = expected[shared_columns].sort_values(shared_columns).reset_index(drop=True)
            equal = same_schema and all(a[c].astype(str).equals(b[c].astype(str)) for c in shared_columns)
            item.update(
                status="replayed",
                rows=len(table),
                matches_saved_reference=bool(equal),
                same_schema=same_schema,
                actual_columns=list(table),
                reference_columns=list(expected),
                sha256=fingerprints(dest)[0],
            )
            write(args.output / "report.json", report)
        report["status"] = (
            "replayed" if all(x["matches_saved_reference"] for x in report["datasets"]) else "reference_discrepancy"
        )
        report["source_unchanged"] = verify_source(args.assets) == source
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["optimization_spot_checks"] = spot_checks
        write(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "unique_trees": len(trees)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--asset-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
