"""Frozen native Scientist cohort from the published DeepLoc 2.0 folds.

All original training folds are retained. Validation/test are preselected by
sequence hash under the native 512-row limit; HPA is reserved externally.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from pathlib import Path

from .deeploc_source import LABELS, canonical, regular_file, require, sha256
from .deeploc_source import SCHEMA as SOURCE_SCHEMA

SCHEMA = "deeploc_native_protocol.v1"
SELECTION_SALT = "deeploc2-native-v1:"
FOLDS = {"train": (0, 1, 2), "validation": (3,), "test": (4,)}
HOLDOUT_LIMIT = 512
PROVENANCE = "DeepLoc 2.0, DOI 10.1093/nar/gkac278; original Swiss-Prot localization annotations."
GROUPING = (
    "Published folds 0/1/2 train, fold 3 validation, fold 4 test; "
    "sequence SHA-256 groups reject exact overlap. Original folds are not individual homology clusters. "
    "Authors report 30% sequence similarity separation; not independently recomputed here."
)


def read_curated(source_path, expected_sha256):
    require(
        isinstance(expected_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", expected_sha256),
        "Supply the frozen curated source SHA-256",
    )
    source_path = Path(source_path).absolute()
    raw = regular_file(source_path, 1024**2)
    require(sha256(raw) == expected_sha256, "Curated source manifest checksum changed")
    report = json.loads(raw)
    require(
        report.get("schema") == SOURCE_SCHEMA
        and report.get("state") == "curated_source_not_admitted"
        and report.get("labels") == list(LABELS),
        "Unexpected curated localization source",
    )
    observations = report.get("observations", {})
    require(set(observations) == {"swissprot", "hpa"}, "Both curated cohorts must be bound")
    swissprot = None
    for cohort in ("swissprot", "hpa"):
        asset = observations[cohort]
        require(asset.get("file") == cohort + ".jsonl", "Unexpected curated asset path")
        content = regular_file(source_path.parent / asset["file"], 64 * 1024**2)
        require(
            len(content) == asset.get("bytes") and sha256(content) == asset.get("sha256"),
            "Curated asset checksum or size changed",
        )
        require(
            type(asset.get("rows")) is int and asset["rows"] == len(content.splitlines()),
            "Curated asset row count changed",
        )
        if cohort == "swissprot":
            swissprot = [json.loads(line) for line in content.splitlines()]
    require({row["published_partition"] for row in swissprot} == set(range(5)), "Preserve all published folds")
    return report, swissprot


def selected_splits(rows):
    """Selection depends on sequence identity only, never labels or model scores."""
    result = {}
    for split, folds in FOLDS.items():
        ordered = sorted(
            (row for row in rows if row["published_partition"] in folds),
            key=lambda row: (sha256((SELECTION_SALT + row["sequence_sha256"]).encode()), row["id"]),
        )
        result[split] = ordered if split == "train" else ordered[:HOLDOUT_LIMIT]
    return result


def export_protocol(source_path, expected_sha256, output):
    from agent.harness.research_evaluation import evaluation_bytes, validate_data

    report, rows = read_curated(source_path, expected_sha256)
    chosen = selected_splits(rows)
    native = {
        "task": "multilabel",
        "labels": sorted(LABELS),
        "provenance": PROVENANCE,
        "grouping": GROUPING,
        "splits": {
            split: [
                {
                    "id": row["id"],
                    "group": row["sequence_sha256"],
                    "inputs": {"sequence": row["sequence"]},
                    "target": row["labels"],
                }
                for row in values
            ]
            for split, values in chosen.items()
        },
    }
    validate_data(native, "supervised")
    native_size = len(evaluation_bytes(native))
    selected = {row["id"] for values in chosen.values() for row in values}
    membership = [
        {
            "id": row["id"],
            "sequence_sha256": row["sequence_sha256"],
            "published_partition": row["published_partition"],
            "split": next(name for name, folds in FOLDS.items() if row["published_partition"] in folds),
            "selected": row["id"] in selected,
        }
        for row in sorted(rows, key=lambda row: row["id"])
    ]
    buffer = io.StringIO(newline="")
    fields = ["id", "sequence_sha256", "split", "published_partition", "sequence", "localization"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for split, values in chosen.items():
        for row in values:
            writer.writerow(
                {
                    "id": row["id"],
                    "sequence_sha256": row["sequence_sha256"],
                    "split": split,
                    "published_partition": row["published_partition"],
                    "sequence": row["sequence"],
                    "localization": canonical(row["labels"]),
                }
            )
    dataset = buffer.getvalue().encode()
    require(len(dataset) <= 20 * 1024**2, "Selected table exceeds native protected-data limit")
    parameters = {
        "data_file": "inputs/localization.csv",
        "target_column": "localization",
        "input_columns": ["sequence"],
        "group_column": "sequence_sha256",
        "id_column": "id",
        "split_column": "split",
        "task": "multilabel",
        "provenance": PROVENANCE,
        "grouping": GROUPING,
    }
    protocol = {
        "schema": SCHEMA,
        "state": "prepared_not_trained",
        "source_sha256": expected_sha256,
        "source_dataset_id": report["source_dataset_id"],
        "property_family": report["property_family"],
        "source_assets": report["assets"],
        "curated_observations": report["observations"],
        "publication": report["publication"],
        "labels": sorted(LABELS),
        "model_task_count": 1,
        "selection": {
            "method": "ascending SHA256(salt + sequence_sha256), then protein ID",
            "salt": SELECTION_SALT,
            "uses_targets": False,
            "all_training_folds_retained": True,
            "holdout_row_limit": HOLDOUT_LIMIT,
            "folds": FOLDS,
        },
        "cohorts": {
            name: {
                "available": sum(row["published_partition"] in FOLDS[name] for row in rows),
                "selected": len(values),
                "omitted": sum(row["published_partition"] in FOLDS[name] for row in rows) - len(values),
                "published_fold_counts": dict(
                    sorted(Counter(str(row["published_partition"]) for row in values).items())
                ),
            }
            for name, values in chosen.items()
        },
        "external_hpa": {
            "rows": report["observations"]["hpa"]["rows"],
            "used_for_fit_or_selection": False,
            "policy": "Reserved external evaluation after candidate and protocol are frozen; not supplied to Scientist.",
        },
        "scope": "One joint ten-output model task. A bounded validation/test cohort, not published five-fold cross-validation or a full DeepLoc reproduction.",
        "metrics": {
            "primary": "macro_brier",
            "direction": "minimize",
            "secondary": ["macro_average_precision", "micro_average_precision", "macro_f1_at_0.5"],
            "missing_labels": "null cells excluded; source zero is not independent experimental absence",
            "rare_labels": "Report per-label positives/negatives and undefined metrics; no rare-label utility claim without adequate evidence.",
        },
        "reference_methods": ["training_prevalence", "aac_logistic", "terminal_logistic"],
        "native_prepare": parameters,
        "native_evaluation_bytes_before_preparation_metadata": native_size,
        "data_terms": {
            "dataset": "DTU downloadable research data; separate asset licensing is not stated on the retrieved download page.",
            "article": "CC BY 4.0; applies to the article, not assumed to cover all separate assets.",
            "upstream_code": "CC BY-NC-SA 4.0 in the authors' GitHub LICENSE; no upstream model code or weights are bundled here.",
            "upstream_license_url": "https://github.com/teevee112/DeepLoc-2.0/blob/main/LICENSE",
            "publication": "Private research staging only; retain source attribution and resolve asset redistribution terms before public release.",
        },
        "trained_models": 0,
        "external_model_requests": 0,
    }
    output = Path(output).absolute()
    require(
        not output.exists() and not any(p.is_symlink() for p in [output, *output.parents]),
        "Output must be a new directory without symlinks",
    )
    output.mkdir(parents=True, mode=0o700)
    artifacts = {
        "dataset.csv": dataset,
        "membership.jsonl": ("\n".join(canonical(row) for row in membership) + "\n").encode(),
    }
    for filename, content in artifacts.items():
        file = output / filename
        with file.open("xb") as stream:
            stream.write(content)
        file.chmod(0o600)
    protocol["artifacts"] = {
        name: {"sha256": sha256(content), "bytes": len(content)} for name, content in artifacts.items()
    }
    file = output / "protocol.json"
    with file.open("x") as stream:
        stream.write(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    file.chmod(0o600)
    return protocol
