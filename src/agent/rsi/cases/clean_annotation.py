"""CLEAN inference reconstruction and independent post-embedding numerical audit.

This software uses or is derived from the CLEAN software developed by the
Department of Chemical and Biomolecular Engineering at the University of
Illinois Urbana-Champaign. Author software stays in separately licensed research assets.
It is not redistributed or modified by this module. See the asset license.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

COMMIT = "f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15"
MANIFEST_SHA256 = "b673aaf0eb9af1b70f03061a2589e857bf0f2182a2405f08c4928f9d51acc4c8"
COHORTS = {
    "new": ("app/data/new.csv", 392, "33e1ff5356c206b99449b3b2a3903bff41b1f30317be057f66f821d32534e259"),
    "price": ("app/data/datasets/price.csv", 149, "670924baf57826eafb43a3264ba2bc01a74caa72ced73279f1cc64f13ef2adc8"),
    "halogenase": (
        "app/data/datasets/halogenase.csv",
        37,
        "ebbcee122ea8209f7dd5ff65f31e7097279b8e7a459d9b683b3bb2a0aa96df5f",
    ),
}
MODELS = ("split70", "split100")
ENVIRONMENT = {
    "torch": "2.8.0+cu128",
    "fair-esm": "2.0.0",
    "numpy": "1.26.4",
    "pandas": "2.2.1",
    "scikit-learn": "1.4.2",
}
ALLOCATION = {"cohorts": list(COHORTS), "models": list(MODELS), "sequence_records": 578, "prediction_records": 1156}


def artifact_names():
    return (
        {"protocol.json", "run_clean_annotation.py", "clean_annotation.py"}
        | {name + suffix for name in COHORTS for suffix in (".ids.json", ".esm.npy")}
        | {
            model + "." + name + suffix
            for model in MODELS
            for name in COHORTS
            for suffix in (".distances.npy", "_maxsep.csv")
        }
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def ordinary(root, name):
    root = Path(root)
    p = root / name
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid CLEAN artifact path")
    cur = p
    while cur != root:
        if cur.is_symlink():
            raise ValueError("Symlinked CLEAN artifact")
        cur = cur.parent
    if not p.is_file():
        raise ValueError("Missing CLEAN artifact: " + str(name))
    return p


def verify_assets(root):
    path = ordinary(root, "manifest.json")
    if sha(path) != MANIFEST_SHA256:
        raise ValueError("CLEAN asset admission differs")
    manifest = json.loads(path.read_text())
    if manifest["source_commit"] != COMMIT:
        raise ValueError("CLEAN source version differs")
    for name, expected in manifest["files"].items():
        if sha(ordinary(root, name)) != expected:
            raise ValueError("CLEAN asset changed: " + name)
    return manifest


def read_rows(path):
    with Path(path).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["Entry", "EC number", "Sequence"]:
            raise ValueError("Unexpected CLEAN columns")
        rows = list(reader)
    if not rows or len({r["Entry"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate sequence identities")
    for row in rows:
        if not row["Entry"] or not row["EC number"] or not 1 <= len(row["Sequence"]) <= 1022:
            raise ValueError("Invalid CLEAN row")
    return rows


def dataset(root, cohort):
    name, count, expected = COHORTS[cohort]
    p = ordinary(Path(root) / "source", name)
    if sha(p) != expected:
        raise ValueError("CLEAN cohort differs from the fixed author release")
    rows = read_rows(p)
    if len(rows) != count:
        raise ValueError("Incomplete CLEAN cohort")
    return rows


def class_index(rows):
    """Author CSV first-appearance EC order; duplicated memberships count once."""
    index = {}
    for row in rows:
        for ec in row["EC number"].split(";"):
            index.setdefault(ec, set()).add(row["Entry"])
    return index


def centers(embeddings, index):
    matrix = np.asarray(embeddings, dtype=np.float64)
    sizes = [len(v) for v in index.values()]
    if matrix.shape != (sum(sizes), 128) or not np.isfinite(matrix).all():
        raise ValueError("Released embeddings do not cover the ordered EC memberships")
    rows, offset = [], 0
    for n in sizes:
        rows.append(matrix[offset : offset + n].mean(axis=0))
        offset += n
    return np.asarray(rows)


def forward(weights, x):
    """NumPy replay of the fixed released network, independent of torch modules."""
    expected = {f"fc{i}.{p}" for i in [1, 2, 3] for p in ["weight", "bias"]} | {
        f"ln{i}.{p}" for i in [1, 2] for p in ["weight", "bias"]
    }
    if set(weights) != expected:
        raise ValueError("Unexpected CLEAN network state")
    h = np.asarray(x, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 1280 or not np.isfinite(h).all():
        raise ValueError("Invalid ESM-1b means")
    for i in [1, 2, 3]:
        h = h @ np.asarray(weights[f"fc{i}.weight"], dtype=np.float64).T + weights[f"fc{i}.bias"]
        if i != 3:
            h = (h - h.mean(axis=1, keepdims=True)) / np.sqrt(h.var(axis=1, keepdims=True) + 1e-5)
            h = h * weights[f"ln{i}.weight"] + weights[f"ln{i}.bias"]
            h = np.maximum(h, 0)
    if h.shape[1] != 128 or not np.isfinite(h).all():
        raise ValueError("Invalid CLEAN projection")
    return h


def distances(projected, centroid):
    return np.asarray([np.sqrt(np.sum((centroid - row) ** 2, axis=1)) for row in projected])


def select_labels(distance, labels):
    """Independently express the released first-gradient max-separation rule."""
    distance = np.asarray(distance)
    if distance.ndim != 2 or distance.shape[1] != len(labels) or len(labels) < 10 or not np.isfinite(distance).all():
        raise ValueError("Invalid EC distance map")
    answers = []
    for row in distance:
        indices = np.argsort(row, kind="stable")[:10]
        values = row[indices]
        tail_mean = (values[1:].sum() + 10 * values[-1]) / 19
        deltas = np.abs(np.diff(np.abs(values - tail_mean)))
        eligible = np.flatnonzero(deltas > deltas.mean())
        if not len(eligible):
            raise ValueError("The released max-separation rule is undefined for equal gradients")
        end = int(eligible[0])
        if end >= 5:
            end = 0
        answers.append([labels[i] for i in indices[: end + 1]])
    return answers


def read_predictions(path, ids):
    with Path(path).open() as f:
        rows = list(csv.reader(f))
    if [r[0] for r in rows] != ids or any(len(r) < 2 for r in rows):
        raise ValueError("Prediction identity/order differs")
    labels, distance = [], []
    for row in rows:
        labels.append([v.split("/")[0].removeprefix("EC:") for v in row[1:]])
        distance.append([float(v.split("/")[1]) for v in row[1:]])
        if len(set(labels[-1])) != len(labels[-1]) or not all(np.isfinite(distance[-1])):
            raise ValueError("Invalid or repeated predicted labels")
    return labels, distance


def metrics(truth, predictions):
    """Full-union multilabel metrics; out-of-cohort predicted ECs count as errors."""
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Prediction cohort incomplete")
    truth, predictions = [set(x) for x in truth], [set(x) for x in predictions]
    labels = sorted(set().union(*truth, *predictions))
    a = np.asarray([[v in row for v in labels] for row in truth], dtype=bool)
    b = np.asarray([[v in row for v in labels] for row in predictions], dtype=bool)
    tp, fp, fn = (a & b).sum(0), (~a & b).sum(0), (a & ~b).sum(0)

    def div(x, y):
        return np.divide(x, y, out=np.zeros_like(x, dtype=float), where=y != 0)

    weights = a.sum(0) / a.sum()
    return {
        "rows": len(a),
        "truth_labels": len(set().union(*truth)),
        "union_labels": len(labels),
        "weighted_precision": float((weights * div(tp, tp + fp)).sum()),
        "weighted_recall": float((weights * div(tp, tp + fn)).sum()),
        "weighted_f1": float((weights * div(2 * tp, 2 * tp + fp + fn)).sum()),
        "micro_precision": float(div(np.array(tp.sum()), np.array((tp + fp).sum()))),
        "micro_recall": float(div(np.array(tp.sum()), np.array((tp + fn).sum()))),
        "micro_f1": float(div(np.array(2 * tp.sum()), np.array((2 * tp + fp + fn).sum()))),
        "exact_match": float(np.mean(np.all(a == b, axis=1))),
        "predicted_labels_absent_from_truth": sorted(set().union(*predictions) - set().union(*truth)),
    }


def verify_artifacts(report, output, assets):
    """Reconstruct predictions from fresh ESM means and the released state tensors.

    The ESM transformer itself is executed by the runner; this audit starts at
    the bound per-sequence means and does not claim a second transformer run.
    """
    import torch

    manifest = verify_assets(assets)
    if (
        report.get("status") != "completed"
        or report.get("execution_origin") != "new_sequence_embedding_and_author_inference"
    ):
        raise ValueError("Incomplete CLEAN computation")
    if report.get("asset_manifest_sha256") != MANIFEST_SHA256 or report.get("allocation") != ALLOCATION:
        raise ValueError("CLEAN scope changed")
    if report.get("completed_embedding_records") != 578 or report.get("source_commit") != COMMIT:
        raise ValueError("Incomplete CLEAN sequence execution")
    if any(
        report.get(k) is not False
        for k in ("full_paper_reproduced", "original_clean_training_reproduced", "new_model_developed")
    ):
        raise ValueError("Unsupported CLEAN scientific claim")
    if set(report.get("artifact_sha256", {})) != artifact_names():
        raise ValueError("Incomplete CLEAN artifact coverage")
    for name, expected in report["artifact_sha256"].items():
        if sha(ordinary(output, name)) != expected:
            raise ValueError("CLEAN output changed: " + name)
    protocol = json.loads(ordinary(output, "protocol.json").read_text())
    expected_protocol = {
        "source_commit": COMMIT,
        "asset_manifest_sha256": MANIFEST_SHA256,
        "environment": ENVIRONMENT,
        "allocation": ALLOCATION,
        "encoder": manifest["encoder"],
        "device": "cpu",
        "threads": 4,
        "seed": 1234,
    }
    if any(protocol.get(k) != v for k, v in expected_protocol.items()):
        raise ValueError("CLEAN execution protocol differs")
    if protocol.get("source_hashes") != {
        n: report["artifact_sha256"][n] for n in ("run_clean_annotation.py", "clean_annotation.py")
    }:
        raise ValueError("CLEAN implementation binding differs")
    if set(report.get("results", {})) != {m + "." + n for m in MODELS for n in COHORTS}:
        raise ValueError("Incomplete CLEAN evaluation allocations")
    replay = {}
    for model in MODELS:
        train = read_rows(Path(assets) / "source/app/data" / (model + ".csv"))
        index = class_index(train)
        train_ids = {r["Entry"] for r in train}
        train_sequences = {r["Sequence"] for r in train}
        state = torch.load(Path(assets) / "pretrained" / (model + ".pth"), map_location="cpu", weights_only=True)
        weights = {k: v.detach().numpy() for k, v in state.items()}
        raw = torch.load(
            Path(assets) / "pretrained" / (model.removeprefix("split") + ".pt"), map_location="cpu", weights_only=True
        )
        centroid = centers(raw.detach().numpy(), index)
        for cohort in COHORTS:
            rows = dataset(assets, cohort)
            saved_ids = json.loads(ordinary(output, cohort + ".ids.json").read_text())
            if saved_ids != [
                {
                    "id": r["Entry"],
                    "sequence_sha256": hashlib.sha256(r["Sequence"].encode()).hexdigest(),
                    "length": len(r["Sequence"]),
                }
                for r in rows
            ]:
                raise ValueError("ESM sequence mapping changed")
            mean = np.load(ordinary(output, cohort + ".esm.npy"), allow_pickle=False)
            if mean.shape != (len(rows), 1280):
                raise ValueError("Incomplete ESM representations")
            prefix = model + "." + cohort
            projection = forward(weights, mean)
            dist = distances(projection, centroid)
            saved = np.load(ordinary(output, prefix + ".distances.npy"), allow_pickle=False)
            if saved.shape != dist.shape or not np.isfinite(saved).all():
                raise ValueError("Independent CLEAN distance shape/finiteness differs")
            error = float(np.max(np.abs(saved - dist)))
            if error > 5e-5:
                raise ValueError("Independent CLEAN numerical replay differs")
            labels, recorded_distances = read_predictions(
                ordinary(output, prefix + "_maxsep.csv"), [r["Entry"] for r in rows]
            )
            if labels != select_labels(saved, list(index)) or labels != select_labels(dist, list(index)):
                raise ValueError("Independent EC selection differs")
            positions = {ec: i for i, ec in enumerate(index)}
            for i, (ecs, values) in enumerate(zip(labels, recorded_distances)):
                if any(abs(saved[i, positions[ec]] - v) > 5.1e-5 for ec, v in zip(ecs, values)):
                    raise ValueError("Saved CLEAN prediction distances differ")
            scored = metrics([r["EC number"].split(";") for r in rows], labels)
            if scored != report["results"][prefix]["full_union_metrics"]:
                raise ValueError("Independent label metrics differ")
            record = report["results"][prefix]
            if (
                record.get("rows") != len(rows)
                or record.get("train_id_overlap") != sum(r["Entry"] in train_ids for r in rows)
                or record.get("train_exact_sequence_overlap") != sum(r["Sequence"] in train_sequences for r in rows)
            ):
                raise ValueError("CLEAN training overlap or row count differs")
            if cohort == "price":
                original, _ = read_predictions(
                    Path(assets) / "source/app/results/inputs/price_maxsep.csv", [r["Entry"] for r in rows]
                )
                comparison = {
                    "same_label_records": sum(set(a) == set(b) for a, b in zip(original, labels)),
                    "rows": len(rows),
                    "reference_training_head_not_declared_in_file": True,
                }
                if record.get("released_price_prediction_comparison") != comparison:
                    raise ValueError("Released CLEAN reference comparison differs")
            replay[prefix] = {"rows": len(rows), "max_distance_error": error, "metrics": scored}
    return {
        "status": "passed",
        "allocations": replay,
        "transformer_rerun_by_auditor": False,
        "source_commit": manifest["source_commit"],
    }


def canonical_clean(report):
    """Bounded scientific claims; the job gate separately replays native artifacts."""
    if (
        report.get("status") != "completed"
        or report.get("allocation") != ALLOCATION
        or report.get("completed_embedding_records") != 578
    ):
        raise ValueError("Incomplete CLEAN scope")
    if (
        report.get("execution_origin") != "new_sequence_embedding_and_author_inference"
        or report.get("source_commit") != COMMIT
        or report.get("asset_manifest_sha256") != MANIFEST_SHA256
    ):
        raise ValueError("CLEAN computation provenance differs")
    if any(
        report.get(k) is not False
        for k in ("full_paper_reproduced", "original_clean_training_reproduced", "new_model_developed")
    ):
        raise ValueError("Unsupported CLEAN scientific claim")
    expected = {m + "." + n for m in MODELS for n in COHORTS}
    audit = report.get("independent_replay", {})
    if (
        audit.get("status") != "passed"
        or audit.get("transformer_rerun_by_auditor") is not False
        or audit.get("source_commit") != COMMIT
        or set(audit.get("allocations", {})) != expected
        or set(report.get("results", {})) != expected
    ):
        raise ValueError("Incomplete CLEAN independent replay")
    result = {}
    for key in sorted(expected):
        record = report["results"][key]
        replay = audit["allocations"][key]
        n = COHORTS[key.split(".", 1)[1]][1]
        measured = record["full_union_metrics"]
        if (
            record.get("rows") != n
            or measured.get("rows") != n
            or replay.get("rows") != n
            or replay.get("metrics") != measured
        ):
            raise ValueError("CLEAN replay metric binding differs")
        error = replay.get("max_distance_error")
        if type(error) not in (int, float) or not np.isfinite(error) or not 0 <= error <= 5e-5:
            raise ValueError("CLEAN numerical replay tolerance exceeded")
        for field in (
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "exact_match",
        ):
            value = measured.get(field)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1 + 1e-12:
                raise ValueError("Invalid CLEAN metric")
        result[key] = measured
    return {
        "computational_scope": "clean_full_released_cohorts_from_sequences_to_ec_annotation",
        "execution_origin": report["execution_origin"],
        "source_commit": COMMIT,
        "sequence_records": 578,
        "prediction_records": 1156,
        "metrics": result,
        "new_model_developed": False,
        "original_clean_training_reproduced": False,
        "transformer_independently_rerun": False,
        "new_enzyme_discovery_verified": False,
        "recursive_improvement_established": False,
    }
