"""Deterministic, source-bound retrospective tasks for published sequence assays.

These adapters prepare computational prediction tasks. They neither reproduce
the papers' physical experiments nor publish models or call a language model.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path

from .model_tasks import PHASES, load_task
from agent.rsi.engine.storage import sha256_file

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
DESIGN_CLUSTERS = frozenset({"HHH", "HEEH", "EHEE", "EEHEE", "EEHH", "hall"})
PARTITION_VERSION = "published_sequence_tasks.v1"
ESOL_ATTRIBUTION = (
    "eSOL © Hideki Taguchi (Tokyo Institute of Technology) licensed under CC Attribution-Share Alike 2.1 Japan"
)
PAPERS = {
    "esol2009": {
        "title": "Bimodal protein solubility distribution revealed by an aggregation analysis "
        "of the entire ensemble of Escherichia coli proteins",
        "doi": "10.1073/pnas.0811922106",
    },
    "esol2012": {
        "title": "Global analysis of chaperone effects using a reconstituted cell-free translation system",
        "doi": "10.1073/pnas.1201380109",
    },
    "tsuboyama2023": {
        "title": "Mega-scale experimental analysis of protein folding stability in biology and design",
        "doi": "10.1038/s41586-023-06328-6",
    },
}
ESOL_ENDPOINTS = [
    ("esol2009_solubility", "esol2009", "solubility", "Solubility (%)", "%"),
    ("esol2009_expression", "esol2009", "expression", "Yield (uM)", "uM"),
] + [
    (f"esol2012_{condition.lower()}_{stage}", "esol2012", stage, f"{condition} {column}", unit)
    for condition in ("Minus", "TF", "GroE", "KJE")
    for stage, column, unit in (("solubility", "Sol (%)", "%"), ("expression", "Yield (uM)", "uM"))
]


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sequence(value):
    return isinstance(value, str) and 20 <= len(value) <= 2048 and set(value) <= AA


def _write(path, data):
    """Never overwrite an earlier preparation or its failure evidence."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def bound(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": sha256_file(path)}


def deduplicate(records):
    """Exclude every member of a duplicate class, without examining targets."""
    ids = collections.Counter(row["id"] for row in records)
    sequences = collections.Counter(row["sequence"] for row in records)
    kept, excluded = [], []
    for row in records:
        reason = "duplicate_identity" if ids[row["id"]] > 1 else "duplicate_sequence"
        if ids[row["id"]] > 1 or sequences[row["sequence"]] > 1:
            excluded.append({"id": row["id"], "reason": reason})
        else:
            kept.append(row)
    return kept, excluded


def esol_records(csv_path, annotations_path):
    """Join only annotated enzymes to W3110 CDS by agreeing JW/ECK/b IDs."""
    annotations = json.loads(Path(annotations_path).read_text())
    index = collections.defaultdict(list)
    for cds in annotations:
        for jw in set(re.findall(r"\bJW\d{4}\b", " ".join(cds["note"]))):
            index[jw].append(cds)
    kept, excluded = [], []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"JW_ID", "ECK number", "B number", "Type of gene product", "Gene product description"}
        required.update(item[3] for item in ESOL_ENDPOINTS)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("eSOL source schema changed")
        for row in reader:
            if row["Type of gene product"] != "e":
                continue
            identity = row["JW_ID"]
            matches = index[identity]
            reason = None
            if len(matches) != 1:
                reason = "ambiguous_or_missing_w3110_cds"
            else:
                cds = matches[0]
                tokens = set(re.findall(r"\b(?:ECK\d{4}|b\d{4})\b", " ".join(cds["note"])))
                if any(row[key] and row[key] not in tokens for key in ("ECK number", "B number")):
                    reason = "cross_identifier_mismatch"
                elif not _sequence(cds["sequence"]):
                    reason = "noncanonical_or_length"
            if reason:
                excluded.append({"id": identity, "reason": reason})
                continue
            endpoints = {column: _number(row[column]) for _, _, _, column, _ in ESOL_ENDPOINTS}
            if not any(value is not None for value in endpoints.values()):
                excluded.append({"id": identity, "reason": "no_finite_measurement"})
                continue
            kept.append(
                {
                    "id": identity,
                    "sequence": cds["sequence"],
                    "group": identity,
                    "endpoints": endpoints,
                    "locus_tag": cds["locus_tag"],
                    "protein_id": cds["protein_id"],
                    "description": row["Gene product description"],
                }
            )
    kept, duplicate_exclusions = deduplicate(kept)
    return kept, excluded + duplicate_exclusions


def designed_records(archive_path):
    """Read only G0 designed scaffolds and finite author-filtered dG_ML labels."""
    prefix = "Processed_K50_dG_datasets/"
    excluded = collections.Counter()
    kept = []
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(prefix + "Single_DMS_list.csv") as handle:
            parents = {
                row["name"]: row
                for row in csv.DictReader(io.TextIOWrapper(handle))
                if row["DMS_group"] == "G0" and row["WT_cluster"] in DESIGN_CLUSTERS
            }
        if not parents:
            raise ValueError("No source-declared G0 designed scaffolds")
        with archive.open(prefix + "Tsuboyama2023_Dataset2_Dataset3_20230416.csv") as handle:
            for row in csv.DictReader(io.TextIOWrapper(handle)):
                parent = parents.get(row["WT_name"])
                if parent is None:
                    continue
                if row["WT_cluster"] != parent["WT_cluster"]:
                    raise ValueError("Designed scaffold group changed between author tables")
                value = _number(row["dG_ML"])
                if value is None:
                    excluded["author_filtered_or_nonfinite_dG_ML"] += 1
                elif not _sequence(row["aa_seq"]):
                    excluded["noncanonical_or_length"] += 1
                else:
                    kept.append(
                        {
                            "id": row["name"],
                            "sequence": row["aa_seq"],
                            "group": row["WT_name"],
                            "topology": row["WT_cluster"],
                            "endpoints": {"dG_ML": value},
                        }
                    )
    kept, duplicate_exclusions = deduplicate(kept)
    excluded.update(item["reason"] for item in duplicate_exclusions)
    return kept, {"counts": dict(excluded), "duplicate_records": duplicate_exclusions}, parents


def group_assignments(records, groups=None):
    """One target-independent group assignment shared by every endpoint."""
    identities = {row["id"] for row in records}
    if groups is not None and set(groups) != identities:
        raise ValueError("Grouping must cover exactly the admitted identities")
    result = {}
    for row in records:
        group = groups[row["id"]] if groups is not None else row["group"]
        if not isinstance(group, str) or not group:
            raise ValueError("Invalid partition group")
        bucket = int(hashlib.sha256(f"{PARTITION_VERSION}:{group}".encode()).hexdigest(), 16) % 100
        phase = "training" if bucket < 60 else "validation" if bucket < 80 else "future"
        result[row["id"]] = {"group": group, "phase": phase}
    return result


def partition_rows(records, endpoint, assignments, limits=(600, 250, 250)):
    if set(assignments) != {row["id"] for row in records}:
        raise ValueError("Assignment coverage mismatch")
    outputs = {phase: [] for phase in PHASES}
    for row in records:
        phase = assignments[row["id"]]["phase"]
        if phase not in PHASES:
            raise ValueError("Unknown partition")
        value = row["endpoints"].get(endpoint)
        if value is not None:
            if _number(value) is None:
                raise ValueError("A prepared endpoint is not finite")
            outputs[phase].append({"id": row["id"], "sequence": row["sequence"], "target": value})
    selected = {}
    for phase, limit in zip(PHASES, limits, strict=True):
        selected[phase] = sorted(
            outputs[phase],
            key=lambda row: hashlib.sha256(f"{PARTITION_VERSION}:selection:{row['id']}".encode()).hexdigest(),
        )[:limit]
    return selected, {phase: len(rows) for phase, rows in outputs.items()}


def write_task(
    output, *, identity, stage, paper, protein, endpoint, unit, records, assignments, references, minimum_skill=0.0
):
    """Emit a task only after the existing admission contract accepts it."""
    folder = Path(output) / identity
    folder.mkdir()
    data, eligible_counts = partition_rows(records, endpoint, assignments)
    assets = {}
    for phase, rows in data.items():
        path = folder / f"{phase}.json"
        _write(path, rows)
        assets[phase] = bound(path)
    definition = {
        "schema": 1,
        "id": identity,
        "stage": stage,
        "plugin_id": "generated_sequence.v1",
        "scope": {
            "protein": protein,
            "assay": identity,
            "target": endpoint,
            "unit": unit,
            "minimum_sequence_length": 20,
            "maximum_sequence_length": 2048,
        },
        "paper": PAPERS[paper],
        "references": references,
        "data": assets,
        "minimum_skill": minimum_skill,
    }
    load_task(definition)
    task_path = folder / "task.json"
    _write(task_path, definition)
    return {
        "task": bound(task_path),
        "id": identity,
        "stage": stage,
        "rows": {phase: len(rows) for phase, rows in data.items()},
        "eligible_rows": eligible_counts,
        "paper": PAPERS[paper]["doi"],
    }


def audit_catalog(entries):
    """Enforce exact sequence separation globally, including between endpoints."""
    seen = {}
    unique_ids, by_stage, papers = set(), collections.Counter(), set()
    for entry in entries:
        task = json.loads(Path(entry["task"]["path"]).read_text())
        if entry["task"] != bound(entry["task"]["path"]):
            raise ValueError("Catalog task changed")
        if task["id"] in unique_ids:
            raise ValueError("Repeated catalog identity")
        unique_ids.add(task["id"])
        by_stage[task["stage"]] += 1
        papers.add(task["paper"]["doi"])
        for phase, rows in load_task(task).items():
            for row in rows:
                previous = seen.setdefault(row["sequence"], phase)
                if previous != phase:
                    raise ValueError("Cross-task partition leakage by exact sequence")
    return {
        "tasks": len(entries),
        "papers": sorted(papers),
        "stages": dict(by_stage),
        "unique_sequences": len(seen),
        "global_exact_sequence_separation": True,
        "homology_independence_established": False,
        "new_autonomous_models": 0,
        "physical_experiments_reproduced": False,
    }
