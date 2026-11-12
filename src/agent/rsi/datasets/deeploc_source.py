"""Preserve DeepLoc localization labels and published partitions before admission.

This offline adapter consumes a hash-bound download snapshot. It neither fits a
model nor converts localization labels into separate model registrations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "deeploc_source.v1"
SOURCE = "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/"
LABELS = (
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
)
HPA_LABELS = tuple(label for label in LABELS if label not in {"Extracellular", "Plastid"})
FILES = {
    "swissprot": "Swissprot_Train_Validation_dataset.csv",
    "hpa": "hpa_testset.csv",
}
ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
MAX_SOURCE_BYTES = 64 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def regular_file(path, maximum):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in [path, *path.parents]), "Source path must not use symlinks")
    require(path.is_file() and 0 < path.stat().st_size <= maximum, "Missing or oversized source file")
    return path.read_bytes()


def table(raw, cohort):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = reader.fieldnames
    required = (
        {"ACC", "Kingdom", "Partition", "Membrane", "Sequence", *LABELS}
        if cohort == "swissprot"
        else {"sid", "Lengths", "fasta", *HPA_LABELS}
    )
    allowed = required | ({""} if cohort == "swissprot" else set())
    require(
        fields and len(fields) == len(set(fields)) and required <= set(fields) <= allowed,
        "Unexpected or duplicated DeepLoc columns",
    )
    require("" not in fields or fields[0] == "", "Only the original leading CSV index may be unnamed")
    rows = []
    for number, row in enumerate(reader, 2):
        require(number <= 50001, "Source exceeds 50,000 rows; review the changed release")
        require(None not in row and all(v is not None for v in row.values()), f"Malformed CSV line {number}")
        rows.append(row)
    require(rows, "Empty source cohort")
    return rows


def read_snapshot(manifest_path, expected_sha256):
    require(
        isinstance(expected_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", expected_sha256),
        "Supply the frozen source manifest SHA-256",
    )
    path = Path(manifest_path).absolute()
    raw = regular_file(path, 1024**2)
    require(sha256(raw) == expected_sha256, "Source manifest checksum changed")
    manifest = json.loads(raw)
    require(
        isinstance(manifest, dict)
        and manifest.get("schema") == "property_source_snapshot.v1"
        and isinstance(manifest.get("files"), list),
        "Unknown source manifest",
    )
    selected = [item for item in manifest["files"] if item.get("source") == "deeploc2"]
    require(
        len(selected) == 2 and {item.get("file") for item in selected} == set(FILES.values()),
        "Snapshot requires exactly the Swiss-Prot and HPA assets",
    )
    assets, cohorts = {}, {}
    for cohort, filename in FILES.items():
        item = next(item for item in selected if item["file"] == filename)
        url = SOURCE + "data/" + filename
        require(item.get("url") == url and item.get("final_url", url) == url, "Unexpected source URL")
        require(item.get("status") == "downloaded_not_admitted", "Unexpected source snapshot state")
        content = regular_file(path.parent / filename, MAX_SOURCE_BYTES)
        require(
            item.get("bytes") == len(content) and item.get("sha256") == sha256(content),
            "Source asset checksum or size changed",
        )
        rows = table(content, cohort)
        require(type(item.get("rows")) is int and item["rows"] == len(rows), "Source row count changed")
        assets[cohort] = {
            "file": filename,
            "url": url,
            "sha256": sha256(content),
            "bytes": len(content),
            "rows": len(rows),
        }
        cohorts[cohort] = rows
    return assets, cohorts


def binary_label(value, label, line):
    require(value in {"0", "0.0", "1", "1.0"}, f"Invalid {label} label at CSV line {line}")
    return int(float(value))


def normalize(cohort, rows):
    require(cohort in FILES, "Unknown localization cohort")
    observations, ids = [], set()
    for line, row in enumerate(rows, 2):
        key, sequence_key = ("ACC", "Sequence") if cohort == "swissprot" else ("sid", "fasta")
        identity, sequence = row[key], row[sequence_key]
        require(identity and identity == identity.strip() and identity not in ids, "Empty or duplicate protein ID")
        require(
            sequence and sequence == sequence.strip() and set(sequence) <= ALPHABET,
            f"Invalid amino-acid sequence at CSV line {line}; no sequence is silently changed",
        )
        ids.add(identity)
        observed = LABELS if cohort == "swissprot" else HPA_LABELS
        labels = {label: binary_label(row[label], label, line) if label in observed else None for label in LABELS}
        require(any(value == 1 for value in labels.values()), f"Protein has no annotated localization at line {line}")
        metadata = {}
        if cohort == "swissprot":
            require(row["Partition"] in {"0", "1", "2", "3", "4"}, "Invalid published partition")
            partition = int(row["Partition"])
            require(bool(row["Kingdom"].strip()), "Missing source kingdom")
            metadata = {"kingdom": row["Kingdom"], "membrane": binary_label(row["Membrane"], "Membrane", line)}
            if "" in row:
                metadata["source_index"] = row[""]
        else:
            partition = None
            require(row["Lengths"].isdigit() and int(row["Lengths"]) == len(sequence), "HPA sequence length mismatch")
        observations.append(
            {
                "id": f"deeploc2:{cohort}:{identity}",
                "protein_id": identity,
                "protein_namespace": "uniprot" if cohort == "swissprot" else "ensembl_protein",
                "source_cohort": cohort,
                "csv_line": line,
                "published_partition": partition,
                "sequence": sequence,
                "sequence_sha256": sha256(sequence.encode()),
                "labels": labels,
                "observed_labels": list(observed),
                "metadata": metadata,
            }
        )
    return observations


def summarize(cohorts):
    """Audit exact overlap; preserve the author's partition, not a new homology claim."""
    swissprot, hpa = cohorts["swissprot"], cohorts["hpa"]
    require(
        {row["published_partition"] for row in swissprot} == set(range(5)),
        "The source must retain all five published partitions",
    )
    by_sequence = {}
    duplicates = 0
    for row in swissprot:
        key = row["sequence_sha256"]
        if key in by_sequence:
            previous = by_sequence[key]
            require(
                previous["published_partition"] == row["published_partition"],
                "Identical sequence crosses published partitions",
            )
            require(previous["labels"] == row["labels"], "Conflicting labels for an identical sequence")
            duplicates += 1
        else:
            by_sequence[key] = row
    require(
        not set(by_sequence) & {row["sequence_sha256"] for row in hpa}, "Identical sequence overlaps Swiss-Prot and HPA"
    )
    result = {}
    for name, rows in cohorts.items():
        result[name] = {
            "rows": len(rows),
            "protein_ids": len({row["protein_id"] for row in rows}),
            "unique_sequences": len({row["sequence_sha256"] for row in rows}),
            "multilabel_rows": sum(sum(value == 1 for value in row["labels"].values()) > 1 for row in rows),
            "sequence_length": {
                "minimum": min(len(row["sequence"]) for row in rows),
                "maximum": max(len(row["sequence"]) for row in rows),
            },
            "labels": {
                label: {
                    "observed": sum(row["labels"][label] is not None for row in rows),
                    "positive": sum(row["labels"][label] == 1 for row in rows),
                    "source_zero": sum(row["labels"][label] == 0 for row in rows),
                    "unobserved": sum(row["labels"][label] is None for row in rows),
                }
                for label in LABELS
            },
        }
    return {
        "cohorts": result,
        "published_partition_counts": dict(
            sorted(Counter(str(row["published_partition"]) for row in swissprot).items())
        ),
        "swissprot_duplicate_sequence_rows_retained": duplicates,
        "exact_sequence_overlap": 0,
        "source_dataset_count": 1,
        "cohort_count": 2,
        "admitted_tasks": 0,
        "trained_models": 0,
        "external_model_requests": 0,
    }


def export_source(manifest_path, expected_sha256, output):
    assets, tables = read_snapshot(manifest_path, expected_sha256)
    cohorts = {name: normalize(name, rows) for name, rows in tables.items()}
    summary = summarize(cohorts)
    report = {
        "schema": SCHEMA,
        "state": "curated_source_not_admitted",
        "source_manifest_sha256": expected_sha256,
        "assets": assets,
        "publication": "https://doi.org/10.1093/nar/gkac278",
        "source_dataset_id": "deeploc2_localization",
        "property_family": "localization_transport",
        "property": "subcellular_localization",
        "labels": list(LABELS),
        "label_semantics": {
            "one": "Localization annotated in the source table.",
            "zero": "Source-encoded zero; not a separate experimentally confirmed absence assay.",
            "null": "Location unavailable in this cohort; exclude from training loss and scoring.",
            "membrane": "Source annotation metadata, never a sequence input feature or an extra localization class.",
        },
        "partitioning": {
            "swissprot": "Keep the original 0..4 partition values; they are folds, not individual homology-cluster identifiers.",
            "hpa": "Independent external cohort. Do not select, tune or fit with its labels.",
            "published_homology_threshold": "The authors report a 30% sequence-identity boundary.",
            "local_audit": "Exact identity and fold consistency only; no new homology alignment was run.",
        },
        "admission_requirements": [
            "Native multilabel training, probability-output contracts and observed-label masks through independent Review.",
            "Class-aware probability metrics, training-only controls and validation-only threshold selection.",
            "A declared full-fold or pre-frozen bounded-cohort protocol; no silent truncation to native row limits.",
            "Document data reuse terms and all source/target, assay and homology limitations before publication.",
        ],
        "summary": summary,
    }
    output = Path(output).absolute()
    require(
        not output.exists() and not any(p.is_symlink() for p in [output, *output.parents]),
        "Output must be a new directory without symlinks",
    )
    output.mkdir(parents=True, mode=0o700)
    artifacts = {}
    for name, rows in cohorts.items():
        file = output / (name + ".jsonl")
        with file.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
        file.chmod(0o600)
        artifacts[name] = {
            "file": file.name,
            "sha256": sha256(file.read_bytes()),
            "bytes": file.stat().st_size,
            "rows": len(rows),
        }
    report["observations"] = artifacts
    # Written last: a partially written directory never has a successful manifest.
    file = output / "source.json"
    with file.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    file.chmod(0o600)
    return report
