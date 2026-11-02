"""Audited sequence inputs for the complete released CLEAN split10 training.

This software uses or is derived from CLEAN developed by the Department of
Chemical and Biomolecular Engineering, University of Illinois Urbana-Champaign.
Author code remains in separately licensed, immutable research assets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

COMMIT = "f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15"
MANIFEST_SHA256 = "8fb4c580015ce05039d10b7e6a06ced626743798c36ddc2433aa8b6ab417dfa1"
COUNTS = {"train": 7757, "augmentation": 14190, "new": 392, "price": 149, "halogenase": 37}
TOTAL = sum(COUNTS.values())
PLAN_SHA256 = "990d1ab6f59aedf99f7ba78e1888a47496dcc5c0ce2165c8d9647264fa64f527"
ENVIRONMENT = {
    "torch": "2.8.0+cu128",
    "fair-esm": "2.0.0",
    "numpy": "1.26.4",
    "pandas": "2.2.1",
    "scikit-learn": "1.4.2",
}
EVALUATION = {
    "new": ("app/data/new.csv", "33e1ff5356c206b99449b3b2a3903bff41b1f30317be057f66f821d32534e259"),
    "price": ("app/data/datasets/price.csv", "670924baf57826eafb43a3264ba2bc01a74caa72ced73279f1cc64f13ef2adc8"),
    "halogenase": (
        "app/data/datasets/halogenase.csv",
        "ebbcee122ea8209f7dd5ff65f31e7097279b8e7a459d9b683b3bb2a0aa96df5f",
    ),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def ordinary(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid input artifact path")
    path = root / relative
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError("Symlinked input artifact")
        current = current.parent
    if not path.is_file():
        raise ValueError("Missing input artifact: " + name)
    return path


def verify_assets(assets):
    if sha(ordinary(assets, "manifest.json")) != MANIFEST_SHA256:
        raise ValueError("Training asset admission changed")
    manifest = json.loads((assets / "manifest.json").read_text())
    if manifest["source_commit"] != COMMIT:
        raise ValueError("Author source changed")
    for name, expected in manifest["files"].items():
        if sha(ordinary(assets, name)) != expected:
            raise ValueError("Training source or data changed: " + name)
    return manifest


def read_rows(path):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["Entry", "EC number", "Sequence"]:
            raise ValueError("Unexpected author data columns")
        rows = list(reader)
    if not rows or len({r["Entry"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate author identities")
    for row in rows:
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+", row["Entry"])
            or not 1 <= len(row["Sequence"]) <= 1022
            or not row["EC number"]
        ):
            raise ValueError("Unsupported author record")
    return rows


def orphan_ids(rows):
    index = {}
    for row in rows:
        for ec in row["EC number"].split(";"):
            index.setdefault(ec, set()).add(row["Entry"])
    return set().union(*(ids for ids in index.values() if len(ids) == 1))


def read_fasta(path):
    result, identity, parts = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if identity is not None:
                result[identity] = "".join(parts)
            identity, parts = line[1:], []
            if identity in result:
                raise ValueError("Repeated augmentation identity")
        elif line.strip():
            if identity is None:
                raise ValueError("Sequence precedes FASTA identity")
            parts.append(line.strip())
    if identity is not None:
        if identity in result:
            raise ValueError("Repeated augmentation identity")
        result[identity] = "".join(parts)
    return result


def validate_augmentation(rows, augmented):
    orphan = orphan_ids(rows)
    parents = {r["Entry"]: r for r in rows if r["Entry"] in orphan}
    expected = {f"{identity}_{i}" for identity in parents for i in range(10)}
    if set(augmented) != expected:
        raise ValueError("Augmentation does not cover every orphan and all ten replicas")
    records = []
    for key, sequence in augmented.items():
        parent = key.rsplit("_", 1)[0]
        original = parents[parent]["Sequence"]
        letters = sequence.replace("<mask>", "*")
        if (
            len(letters) != len(original)
            or letters[0] != original[0]
            or any(a not in (b, "*") for a, b in zip(letters, original))
        ):
            raise ValueError("Augmentation changed residues outside author masking")
        records.append(
            {
                "key": key,
                "id": key,
                "parent": parent,
                "cohort": "augmentation",
                "sequence": sequence,
                "tokens": len(letters),
                "labels": parents[parent]["EC number"].split(";"),
            }
        )
    return records


def prepare_plan(work, assets, evaluation_source):
    """Run unchanged author masking; include every train, orphan and evaluation row."""
    manifest = verify_assets(assets)
    train = read_rows(assets / "source/app/data/split10.csv")
    if len(train) != COUNTS["train"] or len(orphan_ids(train)) != 1419:
        raise ValueError("Released training cohort changed")
    (work / "data/esm_data").mkdir(parents=True, exist_ok=False)
    shutil.copy2(assets / "source/app/data/split10.csv", work / "data/split10.csv")
    sys.path.insert(0, str(assets / "source/app/src"))
    from CLEAN.utils import mutate_single_seq_ECs, seed_everything

    previous = Path.cwd()
    try:
        os.chdir(work)
        seed_everything(1234)
        fasta_name = mutate_single_seq_ECs("split10")
    finally:
        os.chdir(previous)
    augmented = validate_augmentation(train, read_fasta(work / "data" / (fasta_name + ".fasta")))
    records = [
        {
            "key": row["Entry"],
            "id": row["Entry"],
            "cohort": "train",
            "sequence": row["Sequence"],
            "tokens": len(row["Sequence"]),
            "labels": row["EC number"].split(";"),
        }
        for row in train
    ]
    records += augmented
    eval_sources = {}
    for cohort, (name, expected) in EVALUATION.items():
        path = ordinary(evaluation_source, name)
        if sha(path) != expected:
            raise ValueError("Evaluation cohort changed")
        shutil.copy2(path, work / "data" / ("eval_" + cohort + ".csv"))
        eval_sources[cohort] = expected
        for row in read_rows(path):
            records.append(
                {
                    "key": "eval_" + cohort + "_" + hashlib.sha256(row["Entry"].encode()).hexdigest()[:24],
                    "id": row["Entry"],
                    "cohort": cohort,
                    "sequence": row["Sequence"],
                    "tokens": len(row["Sequence"]),
                    "labels": row["EC number"].split(";"),
                }
            )
    plan = {
        "schema": 1,
        "source_commit": COMMIT,
        "asset_manifest_sha256": MANIFEST_SHA256,
        "seed": 1234,
        "counts": COUNTS,
        "evaluation_sources": eval_sources,
        "records": records,
        "encoder": manifest["encoder"],
    }
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if (
        plan.get("source_commit") != COMMIT
        or plan.get("asset_manifest_sha256") != MANIFEST_SHA256
        or plan.get("counts") != COUNTS
        or plan.get("seed") != 1234
    ):
        raise ValueError("Training plan admission changed")
    records = plan["records"]
    if len(records) != TOTAL or len({x["key"] for x in records}) != TOTAL:
        raise ValueError("Incomplete or repeated planned inputs")
    if {c: sum(x["cohort"] == c for x in records) for c in COUNTS} != COUNTS:
        raise ValueError("Planned cohort coverage differs")
    for row in records:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["key"]) or row["key"] in {".", ".."}:
            raise ValueError("Unsafe tensor identity")
        if row["tokens"] != len(row["sequence"].replace("<mask>", "*")) or not 1 <= row["tokens"] <= 1022:
            raise ValueError("Input token span differs")
    if digest(plan) != PLAN_SHA256:
        raise ValueError("Full admitted sequence plan changed")


def receipt_payload(row, path, encoder_hash, protocol_hash):
    return {
        "input_sha256": digest(row),
        "encoder_sha256": encoder_hash,
        "protocol_sha256": protocol_hash,
        "tensor_sha256": sha(path),
        "shape": [1280],
        "dtype": "float32",
        "representation": "layer33_residue_mean_excluding_bos_eos",
        "computed": True,
    }


def verify_receipt(work, receipts, row, encoder_hash, protocol_hash):
    import torch

    key = row["key"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or key in {".", ".."}:
        raise ValueError("Unsafe tensor identity")
    path = ordinary(work, "data/esm_data/" + key + ".pt")
    receipt = json.loads(ordinary(receipts, key + ".json").read_text())
    if receipt != receipt_payload(row, path, encoder_hash, protocol_hash):
        raise ValueError("Embedding receipt or input binding differs")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (1280,)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
    ):
        raise ValueError("Invalid embedding tensor")
    return value


def verify_completed(output):
    report = json.loads(ordinary(output, "report.json").read_text())
    protocol = json.loads(ordinary(output, "protocol.json").read_text())
    plan = json.loads(ordinary(output, "plan.json").read_text())
    validate_plan(plan)
    if (
        report.get("status") != "completed"
        or report.get("completed_records") != TOTAL
        or any(
            report.get(k) is not False
            for k in ["training_completed", "autonomous_model_developed", "full_paper_reproduced"]
        )
    ):
        raise ValueError("Incomplete or overclaimed training inputs")
    if sha(output / "plan.json") != protocol["plan_sha256"] or digest(protocol) != report["protocol_sha256"]:
        raise ValueError("Input plan or protocol changed")
    for name, expected in protocol["implementation"].items():
        if sha(ordinary(output, name)) != expected:
            raise ValueError("Preparation implementation changed")
    keys = {x["key"] for x in plan["records"]}
    if {p.stem for p in (output / "receipts").glob("*.json")} != keys or {
        p.stem for p in (output / "work/data/esm_data").glob("*.pt")
    } != keys:
        raise ValueError("Tensor or receipt coverage differs")
    for row in plan["records"]:
        verify_receipt(output / "work", output / "receipts", row, plan["encoder"]["model"]["sha256"], digest(protocol))
    return {
        "status": "passed",
        "records": TOTAL,
        "counts": COUNTS,
        "transformer_reexecuted": False,
        "training_completed": False,
    }
