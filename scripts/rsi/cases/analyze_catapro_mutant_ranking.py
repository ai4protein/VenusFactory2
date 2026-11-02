#!/usr/bin/env python3
"""Recompute Figure 4 mutant ranking from an existing complete CataPro replay.

No inference or training occurs. Public author predictions/metrics are reproduction
references, never held-out evaluation data for development or model selection.
"""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi.cases.catapro_case import DATASETS, log_targets
from agent.rsi.cases.catapro_ranking import ranking_metrics, reaction_groups, summarize_groups


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Hash mismatch: {path.name}")


def references(archive):
    with zipfile.ZipFile(archive) as z:
        data = z.read("Source Data/Fig4.xlsx")
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    result = {}
    for kind, a, b in [("kcat", "a", "d"), ("km", "b", "e"), ("activity", "c", "f")]:
        scc = {r[0]: float(r[1]) for r in wb[f"fig.4{a}"].iter_rows(min_row=2, values_only=True)}
        acc = {r[0]: float(r[1]) for r in wb[f"fig.4{b}"].iter_rows(min_row=2, values_only=True)}
        if set(scc) != set(acc):
            raise ValueError("Official SCC and accuracy group identities differ")
        result[kind] = {key: {"scc": scc[key], "strict_pair_accuracy": acc[key]} for key in scc}
    wb.close()
    return result


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    replay = args.replay.resolve()
    source = args.source.resolve()
    paper = args.paper_assets.resolve()
    archive = paper / "41467_2025_58038_MOESM7_ESM.zip"
    receipt = json.loads((paper / "source_receipt.json").read_text())
    entry = next(r for r in receipt["files"] if r["name"] == archive.name)
    checked(archive, entry["sha256"])
    execution = json.loads((replay / "execution.json").read_text())
    if execution["status"] != "completed":
        raise ValueError("Full replay is incomplete")
    bound = {
        str(archive): sha(archive),
        str(paper / "source_receipt.json"): sha(paper / "source_receipt.json"),
        str(replay / "execution.json"): sha(replay / "execution.json"),
    }
    loaded = {}
    for kind in ["kcat", "km", "activity"]:
        folder = replay / kind
        e = next(r for r in execution["cohorts"] if r["name"] == kind)
        if e["status"] != "completed" or e["returncode"] != 0:
            raise ValueError("Replay cohort did not complete")
        checked(folder / "report.json", e["report_sha256"])
        report = json.loads((folder / "report.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        input_path = source / "datasets" / DATASETS[kind]
        checked(input_path, protocol["input_sha256"])
        checked(folder / "predictions.npy", report["predictions_sha256"])
        if report["status"] != "completed" or protocol["dataset"] != kind:
            raise ValueError("Invalid replay status or cohort")
        with input_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        prediction = np.load(folder / "predictions.npy", allow_pickle=False)
        if prediction.shape != (len(rows), 1) or len(rows) != protocol["rows"]:
            raise ValueError("Replay row count or prediction shape differs")
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite replay predictions")
        for p in [input_path, folder / "predictions.npy", folder / "report.json", folder / "protocol.json"]:
            bound[str(p)] = sha(p)
        loaded[kind] = (rows, prediction[:, 0])
    implementations = [
        Path(__file__),
        ROOT / "src/agent/rsi/cases/catapro_ranking.py",
        ROOT / "src/agent/rsi/cases/catapro_case.py",
    ]
    for p in implementations:
        bound[str(p)] = sha(p)
        dest = out / "implementation" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    write(
        out / "protocol.json",
        {
            "started_unix": started,
            "input_hashes": bound,
            "minimum_group_sizes": [20, 30],
            "grouping": "Exact raw UniProtID-SMILES pair; retain all author rows including wild type; no resplitting",
            "prediction_order": "Original CSV row order, bound by original replay input and prediction hashes",
            "scc": "Spearman with average ranks for ties; undefined constant groups are explicit",
            "accuracy": "Strict concordant pairs / all N*(N-1)/2 pairs; any prediction or target tie is not a strict success",
            "secondary_accuracy": "Strict concordant pairs / pairs with unequal experimental targets; diagnostic only",
            "published_metric_absolute_tolerance": 1e-6,
            "scope": "Post-hoc scientific analysis of released-checkpoint CV replay, not new holdout, new inference, autonomous discovery or wet lab",
            "versions": {
                "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "openpyxl": openpyxl.__version__,
            },
        },
    )
    reference = references(archive)
    all_groups = {}
    summary = {}
    for kind, (rows, prediction) in loaded.items():
        target = np.array([log_targets(row)[kind] for row in rows])
        groups = []
        for key, indices in sorted(reaction_groups(rows).items()):
            if len(indices) < 20:
                continue
            m = ranking_metrics(prediction[indices], target[indices])
            row = {
                "reaction": key,
                "uniprot_id": rows[indices[0]]["UniProtID"],
                "sample_ids": [rows[i][""] for i in indices],
                "distinct_sequences": len({rows[i]["Sequence"] for i in indices}),
                "folds": sorted({int(float(rows[i]["fold"])) for i in indices}),
                "metrics": m,
            }
            if key in reference[kind]:
                row["published"] = reference[kind][key]
                row["absolute_differences"] = {
                    metric: abs(m[metric] - value) if m[metric] is not None else None
                    for metric, value in reference[kind][key].items()
                }
            groups.append(row)
        all_groups[kind] = groups
        by_key = {r["reaction"]: r for r in groups}
        differences = [v for r in groups for v in r.get("absolute_differences", {}).values() if v is not None]
        parity = set(by_key) == set(reference[kind]) and all(
            r["metrics"][metric] is not None and abs(r["metrics"][metric] - value) <= 1e-6
            for key, values in reference[kind].items()
            if key in by_key
            for metric, value in values.items()
            for r in [by_key[key]]
        )
        summary[kind] = {
            "n_ge_20": summarize_groups(groups),
            "n_ge_30": summarize_groups([r for r in groups if r["metrics"]["rows"] >= 30]),
            "source_group_identity_matches": set(by_key) == set(reference[kind]),
            "missing_source_groups": sorted(set(reference[kind]) - set(by_key)),
            "additional_groups": sorted(set(by_key) - set(reference[kind])),
            "published_metric_parity_passed": parity,
            "maximum_absolute_metric_difference": max(differences) if differences else None,
            "multi_fold_groups": sum(len(r["folds"]) > 1 for r in groups),
        }
    write(out / "groups.json", all_groups)
    for path, digest in bound.items():
        checked(Path(path), digest)
    result = {
        "status": "completed",
        "input_hashes_unchanged": True,
        "external_model_calls": 0,
        "new_predictions": 0,
        "new_training_fits": 0,
        "summary": summary,
        "published_metric_parity_passed": all(r["published_metric_parity_passed"] for r in summary.values()),
        "groups_sha256": sha(out / "groups.json"),
        "protocol_sha256": sha(out / "protocol.json"),
        "seconds": time.time() - started,
    }
    write(out / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["replay", "source", "paper-assets", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
