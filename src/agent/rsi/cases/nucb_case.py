"""Frozen full-sequence NucB classification from per-generation measurements.

Uses only labels available in the declared training generations. The consolidated
landscape selects the latest observed label, so it is deliberately not used for
historical training. This is a new modelling task, not author CNN retraining.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from agent.rsi.engine.generated_classifier import AA, validate_rows
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import sha256_file


def reference_sequence(source):
    tree = ast.parse((Path(source) / "nuclease_design/constants.py").read_text())
    values = [
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FULL_REFERENCE_SEQ" for t in n.targets)
    ]
    if len(values) != 1 or not isinstance(values[0], str) or set(values[0]) - AA:
        raise ValueError("Invalid author reference sequence")
    return values[0]


def reconstruct_sequence(reference, mutations):
    seq = list(reference)
    seen = set()
    for old, position, new in mutations:
        if type(position) is not int or not 1 <= position <= len(reference) or position in seen:
            raise ValueError("Invalid or repeated mutation position")
        if reference[position - 1] != old or not isinstance(new, str) or len(new) != 1 or new not in AA:
            raise ValueError("Mutation inconsistent with author reference")
        seq[position - 1] = new
        seen.add(position)
    return "".join(seq)


def temporal_partition(generations):
    if set(generations) != {"g1", "g2", "g3", "g4"}:
        raise ValueError("All four frozen generations required")
    train = {}
    for generation in ("g1", "g2"):
        for row in generations[generation]:
            train[row["sequence"]] = row
    val = {r["sequence"]: r for r in generations["g3"] if r["sequence"] not in train}
    future = {r["sequence"]: r for r in generations["g4"] if r["sequence"] not in train and r["sequence"] not in val}
    return tuple([group[k] for k in sorted(group)] for group in (train, val, future))


def prepare_nucb_case(assets, rebuilt):
    assets, rebuilt = Path(assets), Path(rebuilt)
    manifest = json.loads((assets / "download_manifest.json").read_text())
    for row in manifest["files"]:
        if sha256_file(assets / "source" / row["path"]) != row["sha256"]:
            raise ValueError("Author source changed")
    report = json.loads((rebuilt / "report.json").read_text())
    # Original run failed at a later aggregation stage. Verify every used stage;
    # do not relabel that original overall run as successful.
    stages = {s["generation"]: s for s in report["stages"]}
    reference = reference_sequence(assets / "source")
    generations = {}
    provenance = {}
    for generation in ("g1", "g2", "g3", "g4"):
        name = "processed_data/" + generation + ".csv"
        path = rebuilt / "data" / name
        stage = stages[generation]
        if (
            stage["status"] != "replayed"
            or not stage["matches_author_tables"]
            or sha256_file(path) != stage["output_hashes"][name]
        ):
            raise ValueError("Unverified generation table")
        rows = []
        seen = set()
        with path.open() as stream:
            for raw in csv.DictReader(stream):
                sequence = reconstruct_sequence(reference, json.loads(raw["mutations"]))
                if sequence in seen:
                    raise ValueError("Duplicate within-generation sequence")
                seen.add(sequence)
                label = raw["activity_greater_than_wt"]
                if label not in {"True", "False"}:
                    raise ValueError("Unknown published binary label")
                rows.append(
                    {
                        "id": digest({"protein": "NucB", "sequence": sequence}),
                        "sequence": sequence,
                        "target": int(label == "True"),
                    }
                )
        if len(rows) != stage["rows"]:
            raise ValueError("Generation row count changed")
        generations[generation] = rows
        provenance[generation] = {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
    train, val, future = temporal_partition(generations)
    scope = {
        "protein": "NucB",
        "assay": "published generation-specific NucB screening at pH 7, author FDR calls",
        "target": "activity_greater_than_wt",
        "unit": "class probability",
        "sequence_length": len(reference),
        "class_labels": ["not_above_wildtype", "above_wildtype"],
    }
    for group in (train, val, future):
        validate_rows(group, scope, labels=True)
    source = {
        "repository": "google-deepmind/nuclease_design",
        "commit": manifest["commit"],
        "author_source_manifest_sha256": sha256_file(assets / "download_manifest.json"),
        "reconstructed_parent_report_sha256": sha256_file(rebuilt / "report.json"),
        "generation_files": provenance,
        "reference_sequence_sha256": digest(reference),
    }
    split = {
        "training": "all G1 and G2 sequences; G2 wins repeats within training",
        "acceptance": "G3 sequences absent from training",
        "future": "G4 sequences absent from all G1/G2/G3",
        "representation": "full-length canonical amino-acid sequence",
        "label": "generation-specific author activity_greater_than_wt FDR boolean, not enrichment or historical CNN classes",
        "history": "public retrospective data already examined by investigators; not a blinded or prospective study",
        "limits": "generation shift includes design selection and assay/gating differences; not a pure distribution-shift causal estimate",
        "counts": {"training": len(train), "acceptance": len(val), "future": len(future)},
        "deduplication": "exact sequence disjoint; near-neighbours and common lineage remain",
        "reference_sequence": reference,
    }
    return scope, train, val, future, source, split
