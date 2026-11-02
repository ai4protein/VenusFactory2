"""Frozen REAP embeddings and measured labels for generated-model experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agent.rsi.engine.generated_features import validate_rows
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import sha256_file


def prepare_reap_model_case(admission, features, baseline, family, seed=0):
    admission, features, baseline = map(Path, (admission, features, baseline))
    rows = json.loads((admission / "campaign.json").read_text())
    summary = json.loads((admission / "summary.json").read_text())
    report = json.loads((features / "report.json").read_text())
    baseline_report = json.loads((baseline / "report.json").read_text())
    baseline_protocol = json.loads((baseline / "protocol.json").read_text())
    if report["status"] != "completed" or baseline_report["status"] != "completed":
        raise ValueError("Completed feature and baseline records required")
    if (
        report["input_sha256"] != sha256_file(admission / "campaign.json")
        or report["admission_summary_sha256"] != sha256_file(admission / "summary.json")
        or baseline_protocol["input_sha256"] != report["input_sha256"]
    ):
        raise ValueError("Admission identity differs")
    if (
        report["output_sha256"] != sha256_file(features / "embeddings.npz")
        or baseline_protocol["feature_sha256"] != report["output_sha256"]
        or baseline_report["protocol_sha256"] != sha256_file(baseline / "protocol.json")
    ):
        raise ValueError("Baseline or feature identity differs")
    with np.load(features / "embeddings.npz", allow_pickle=False) as saved:
        X = saved["X"].copy()
        if (
            X.shape != (len(rows), 1280)
            or len(rows) != summary["admitted_campaign"]
            or saved["ids"].tolist() != [r["id"] for r in rows]
            or saved["sequence_sha256"].tolist() != [r["sequence_sha256"] for r in rows]
        ):
            raise ValueError("Feature row identity or dimension differs")
    arms = [a for a in baseline_protocol["allocations"] if a["family"] == family and a["seed"] == seed]
    if len(arms) != 5 or any(any(a[k] != arms[0][k] for k in ["train", "validation", "future"]) for a in arms):
        raise ValueError("One complete common baseline partition required")
    split = arms[0]
    encoder = {
        "name": "esm2_t33_650M_UR50D",
        "layer": 33,
        "pooling": "residue_mean_no_truncation",
        "dtype": report["dtype"],
        "author_source_sha256": report["author_freeze_sha256"],
        "checkpoints": {Path(k).name: v for k, v in report["checkpoints"].items()},
    }
    lengths = {len(r["sequence"]) for r in rows if r["family"] == family}
    if len(lengths) != 1:
        raise ValueError("Aligned full-sequence family required")
    scope = {
        "protein": family,
        "assay": "REAP published relative enzyme activity; initial and automated rounds retained",
        "target": "fold_change_activity",
        "unit": "fold change versus batch-matched reference",
        "sequence_length": lengths.pop(),
        "representation": {
            "name": "ESM2-650M final-layer mean FP32",
            "dimensions": 1280,
            "encoder_sha256": digest(encoder),
        },
    }
    subsets = []
    for role in ["train", "validation", "future"]:
        indices = split[role]
        if any(rows[i]["family"] != family or ((rows[i]["mutation_order"] > 1) != (role == "future")) for i in indices):
            raise ValueError("Family or mutation-order partition mismatch")
        subset = [
            {
                "id": rows[i]["id"],
                "sequence": rows[i]["sequence"],
                "target": rows[i]["target"],
                "features": X[i].tolist(),
            }
            for i in indices
        ]
        validate_rows(subset, scope, labels=True)
        subsets.append(subset)
    groups = [{r["sequence"] for r in subset} for subset in subsets]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Development/future sequence overlap")
    source = {
        "paper": "https://doi.org/10.1038/s41467-026-76264-2",
        "admission": str(admission),
        "features": str(features),
        "baseline": str(baseline),
        "campaign_sha256": report["input_sha256"],
        "feature_sha256": report["output_sha256"],
        "baseline_protocol_sha256": baseline_report["protocol_sha256"],
        "encoder": encoder,
        "source_conflicts_retained": summary["excluded_campaign_plots"],
    }
    partition = {
        "family": family,
        "seed": seed,
        "training": len(subsets[0]),
        "acceptance": len(subsets[1]),
        "future": len(subsets[2]),
        "policy": "Single-mutant train/acceptance; every higher-order mutant is future-only. Fixed baseline seed partition; public retrospective data, not blind.",
        "baseline_comparison_limit": "Author baselines select epochs on the acceptance subset; generated code gets two internal training-only feedback attempts and one external acceptance. Feature/input sets align but optimizer budgets and selection access are not identical; do not infer full algorithmic fairness.",
    }
    return scope, *subsets, source, partition
