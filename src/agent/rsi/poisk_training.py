"""Paired-chain source grouping and training-only POISK feasibility controls."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .poisk_source import AA, TARGETS, digest, require
from .storage import atomic_json, sha256_file

ENCODING = "separate_vh_vl_composition_dipeptide32_length_target_onehot.v1"


def encode(row):
    values = []
    for key in ["heavy", "light"]:
        seq = row[key]
        counts = Counter(seq)
        values.extend(counts[a] / max(1, len(seq)) for a in AA)
        bins = [0.0] * 32
        for i in range(len(seq) - 1):
            pos = int(hashlib.sha256(seq[i : i + 2].encode()).hexdigest(), 16) % 32
            bins[pos] += 1 / (len(seq) - 1)
        values.extend(bins)
        values.append(len(seq) / 200)
    values.append(float(bool(row["light"])))
    values.extend(float(row["target_accession"] == k) for k in sorted(TARGETS))
    return values


def components(rows):
    """Unite patents, either exact chain, and pairs within 5% edit distance."""
    from Bio.Align import PairwiseAligner

    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    antibodies = {}
    for row in rows:
        node = "antibody:" + row["antibody"]
        antibodies[row["antibody"]] = (row["heavy"], row["light"])
        for chain in [row["heavy"], row["light"]]:
            if chain:
                union(node, "chain:" + digest(chain))
        for patent in row["patents"]:
            union(node, "patent:" + patent)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 0, -1
    aligner.open_gap_score = aligner.extend_gap_score = -1

    def close(a, b):
        if not a or not b:
            return a == b
        maximum = int(0.05 * max(len(a), len(b)))
        return abs(len(a) - len(b)) <= maximum and aligner.score(a, b) >= -maximum

    ids = sorted(antibodies)
    edges = []
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            a, b = antibodies[left], antibodies[right]
            if close(a[0], b[0]) and close(a[1], b[1]):
                union("antibody:" + left, "antibody:" + right)
                edges.append([left, right])
    members = defaultdict(list)
    for node in sorted(parent):
        members[find(node)].append(node)
    names = {k: "pg_" + digest(v)[:24] for k, v in members.items()}
    groups = {r["id"]: names[find("antibody:" + r["antibody"])] for r in rows}
    return groups, {
        "policy": "connected components of shared patent ID, either identical chain, or paired chains each within5percent global edit distance",
        "near_clone_edges": edges,
        "groups": groups,
        "patent_family_resolution": "Patent IDs plus sequence links; complete legal patent-family identity is not established",
    }


def choose_split(rows):
    counts = Counter(r["group"] for r in rows)
    if len(rows) < 24 or len(counts) < 5:
        return None
    groups = sorted(counts)
    hold = max(1, round(len(groups) * 0.2))
    if len(groups) - 2 * hold < 3:
        return None
    best = None
    rng = random.Random(20261028)
    for _ in range(2048):
        order = groups[:]
        rng.shuffle(order)
        assignment = {
            g: "validation" if i < hold else "future" if i < 2 * hold else "training" for i, g in enumerate(order)
        }
        totals = Counter()
        for g, count in counts.items():
            totals[assignment[g]] += count
        if not 12 <= totals["training"] <= 600 or min(totals["validation"], totals["future"]) < 4:
            continue
        score = sum(
            (totals[p] / len(rows) - fraction) ** 2
            for p, fraction in [("training", 0.6), ("validation", 0.2), ("future", 0.2)]
        )
        key = (score, tuple(assignment[g] for g in groups))
        if best is None or key < best[0]:
            best = key, assignment
    return best[1] if best else None


def prepare(source, output):
    source, output = Path(source), Path(output)
    require(output.is_absolute() and not output.exists(), "New absolute task folder required")
    manifest = json.loads((source / "manifest.json").read_text())
    require(sha256_file(source / "observations.json") == manifest["observations_sha256"], "Source changed")
    rows = json.loads((source / "observations.json").read_text())
    groups, evidence = components(rows)
    buckets = defaultdict(list)
    for row in rows:
        # Keep known conditions for later explicitly conditioned tasks. An absent
        # condition remains unreported and is never equated with a standard assay.
        if row["condition"]:
            continue
        key = (row["target_accession"], row["antibody_type"], row["method"], row["endpoint"])
        buckets[key].append(
            {
                "id": row["id"],
                "group": groups[row["id"]],
                "values": encode(row),
                "target": row["target"],
                "source_records": row["source_records"],
            }
        )
    output.mkdir(parents=True)
    atomic_json(output / "group_evidence.json", evidence)
    tasks, excluded = [], []
    for key, items in sorted(buckets.items()):
        items.sort(key=lambda r: r["id"])
        assignment = choose_split(items)
        identity = {
            "target_accession": key[0],
            "target_gene": TARGETS[key[0]],
            "antibody_type": key[1],
            "method": key[2],
            "endpoint": key[3],
            "condition": "unreported_in_source",
        }
        if assignment is None:
            excluded.append(
                {
                    "scope": identity,
                    "rows": len(items),
                    "groups": len({r["group"] for r in items}),
                    "reason": "insufficient_independent_groups_or_partition_rows",
                }
            )
            continue
        task_id = "poisk_" + digest(identity)[:20]
        folder = output / task_id
        folder.mkdir()
        counts, files = {}, {}
        for phase in ["training", "validation", "future"]:
            part = [r for r in items if assignment[r["group"]] == phase]
            atomic_json(folder / f"{phase}.json", part)
            counts[phase] = {"rows": len(part), "groups": len({r["group"] for r in part})}
            files[phase] = {"path": str(folder / f"{phase}.json"), "sha256": sha256_file(folder / f"{phase}.json")}
        task = {
            "id": task_id,
            "scope": identity,
            "representation": ENCODING,
            "dimensions": len(items[0]["values"]),
            "data": files,
            "counts": counts,
            "split_assignment": assignment,
            "minimum_training_groups": 3,
            "split_selection": "2048 seeded component permutations; row-count balance only, no label magnitudes",
            "status": "offline_feasibility_task",
            "rsi_plugin_registered": False,
        }
        atomic_json(folder / "task.json", task)
        tasks.append(task)
    report = {
        "schema": "poisk_feasibility_tasks.v1",
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "encoder_source_sha256": sha256_file(Path(__file__)),
        "tasks": tasks,
        "excluded_scopes": excluded,
        "task_count": len(tasks),
        "global_components": len(set(groups.values())),
        "source_observations": len(rows),
        "new_autonomous_models": 0,
        "external_model_calls": 0,
        "final_labels_scored": False,
    }
    atomic_json(output / "manifest.json", report)
    return report


def train_controls(tasks, output):
    """Only task manifests and training partitions are read by this routine."""
    import joblib
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    tasks, output = Path(tasks), Path(output)
    require(output.is_absolute() and not output.exists(), "New absolute control output required")
    manifest = json.loads((tasks / "manifest.json").read_text())
    require(
        manifest["encoder_source_sha256"] == sha256_file(Path(__file__)), "Training preparation implementation changed"
    )
    output.mkdir(parents=True)
    results = []
    factories = {
        "training_mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=0, n_jobs=1
        ),
        "svr": lambda: make_pipeline(StandardScaler(), SVR(C=1, epsilon=0.1)),
    }
    for task in manifest["tasks"]:
        bound = task["data"]["training"]
        path = Path(bound["path"])
        require(sha256_file(path) == bound["sha256"], "Training data changed")
        rows = json.loads(path.read_text())
        x = np.asarray([r["values"] for r in rows])
        y = np.asarray([r["target"] for r in rows])
        groups = np.asarray([r["group"] for r in rows])
        folds = list(GroupKFold(n_splits=min(3, len(set(groups)))).split(x, y, groups))
        folder = output / task["id"]
        folder.mkdir()
        for name, factory in factories.items():
            predictions = np.full(len(rows), np.nan)
            for train, feedback in folds:
                require(not set(groups[train]) & set(groups[feedback]), "Group leakage")
                model = factory()
                model.fit(x[train], y[train])
                predictions[feedback] = model.predict(x[feedback])
            require(np.isfinite(predictions).all(), "Nonfinite training-feedback predictions")
            model = factory()
            model.fit(x, y)
            artifact = folder / f"{name}.joblib"
            joblib.dump(model, artifact)
            result = {
                "task": task["id"],
                "scope": task["scope"],
                "method": name,
                "training_rows": len(rows),
                "training_groups": len(set(groups)),
                "internal_group_cv_folds": len(folds),
                "internal_group_cv_rmse": float(np.sqrt(np.mean((y - predictions) ** 2))),
                "artifact_sha256": sha256_file(artifact),
                "source_training_sha256": bound["sha256"],
                "predictions": [
                    {"id": r["id"], "prediction": float(p)} for r, p in zip(rows, predictions, strict=True)
                ],
                "baseline_only": True,
                "registered_as_rsi_model": False,
            }
            atomic_json(folder / f"{name}.json", result)
            results.append(result)
    report = {
        "status": "completed",
        "task_count": len(manifest["tasks"]),
        "final_fitted_control_models": len(results),
        "internal_cv_fits": sum(r["internal_group_cv_folds"] for r in results),
        "outer_labels_read": False,
        "new_autonomous_models": 0,
        "external_model_calls": 0,
        "results": [{k: v for k, v in r.items() if k != "predictions"} for r in results],
    }
    atomic_json(output / "report.json", report)
    return report
