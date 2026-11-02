"""Backbone-level VHH evaluation with independent C-alpha RMSD replay.

This is a separate publication-oriented audit. It does not rewrite legacy
sequence-level filters or infer binding labels for computational designs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .rfantibody_audit import AA3
from agent.rsi.engine.storage import sha256_file

PAPER_DOI = "10.1038/s41586-025-09721-5"
SUPPLEMENT_SHA256 = "f5d4fc4f3a65a2dc3ab696e165bb86bf150b2437cdc1a7046cfbecae8ccab05c"
REFERENCE_PERCENT = {
    "tcdb": {"1.0": 41.0, "0.1": 33.5, "0.0": 27.5},
    "il7r": {"1.0": 34.5, "0.1": 27.0, "0.0": 12.5},
    "ha": {"1.0": 11.5, "0.1": 7.5, "0.0": 3.5},
    "rbd": {"1.0": 28.0, "0.1": 23.5, "0.0": 11.0},
    "rsv_site3": {"1.0": 25.0, "0.1": 3.0, "0.0": 1.0},
}
OPTIONAL_LIGHT = {"framework_aligned_L1_rmsd", "framework_aligned_L2_rmsd", "framework_aligned_L3_rmsd"}


def read_metadata(path):
    """Read only sequences, C-alpha coordinates and scores, without retaining PDB text."""
    records = {}
    current = None
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("QV_TAG "):
                parts = line.split()
                if len(parts) != 2 or not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[1]) or parts[1] in records:
                    raise ValueError("Invalid or duplicate design identity")
                current = parts[1]
                records[current] = {"residues": {}, "ca": {}, "scores": {}}
            elif line.startswith(("QV_SCORE ", "ATOM ")):
                if current is None:
                    raise ValueError("Structure or scores precede design identity")
                record = records[current]
                if line.startswith("QV_SCORE "):
                    _, tag, fields = line.split(maxsplit=2)
                    if tag != current or record["scores"]:
                        raise ValueError("Ambiguous score identity")
                    for field in fields.strip().split("|"):
                        key, raw = field.split("=", 1)
                        if key in record["scores"]:
                            raise ValueError("Duplicate score name")
                        value = float(raw)
                        if not math.isfinite(value) and key not in OPTIONAL_LIGHT:
                            raise ValueError("Nonfinite applicable score")
                        record["scores"][key] = value if math.isfinite(value) else None
                else:
                    key = (line[21], line[22:27])
                    residue = AA3.get(line[17:20])
                    if residue is None or key[0] not in {"H", "T"}:
                        raise ValueError("Only canonical heavy/target residues are admitted")
                    if key in record["residues"] and record["residues"][key] != residue:
                        raise ValueError("Conflicting residue identity")
                    record["residues"][key] = residue
                    xyz = [float(line[start : start + 8]) for start in (30, 38, 46)]
                    if not all(math.isfinite(v) for v in xyz):
                        raise ValueError("Nonfinite coordinates")
                    if line[12:16].strip() == "CA":
                        if key in record["ca"]:
                            raise ValueError("Duplicate C-alpha coordinate")
                        record["ca"][key] = xyz
    if not records:
        raise ValueError("Empty quiver")
    for record in records.values():
        if set(record["ca"]) != set(record["residues"]) or {c for c, _ in record["ca"]} != {"H", "T"}:
            raise ValueError("Incomplete heavy/target C-alpha coordinates")
        record["sequences"] = {
            chain: "".join(aa for (c, _), aa in record["residues"].items() if c == chain) for chain in ["H", "T"]
        }
        record["coordinates"] = {
            chain: np.array([record["ca"][key] for key in record["residues"] if key[0] == chain])
            for chain in ["H", "T"]
        }
        del record["residues"], record["ca"]
    return records


def target_aligned_ca_rmsd(reference, prediction):
    """Independent NumPy Kabsch fit on target CA, then heavy-chain CA RMSD."""
    if reference["sequences"] != prediction["sequences"]:
        raise ValueError("Sequence changed during refinement")
    fixed, moving = reference["coordinates"]["T"], prediction["coordinates"]["T"]
    if fixed.shape != moving.shape or len(fixed) < 3:
        raise ValueError("Incompatible target alignment")
    fcentre, mcentre = fixed.mean(axis=0), moving.mean(axis=0)
    f, m = fixed - fcentre, moving - mcentre
    if np.linalg.matrix_rank(f, tol=1e-8) < 2 or np.linalg.matrix_rank(m, tol=1e-8) < 2:
        raise ValueError("Degenerate target alignment")
    u, _, vh = np.linalg.svd(m.T @ f)
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] *= -1
    rotation = u @ vh
    heavy = (prediction["coordinates"]["H"] - mcentre) @ rotation + fcentre
    return float(np.sqrt(np.mean(np.sum((heavy - reference["coordinates"]["H"]) ** 2, axis=1))))


def summarize_quivers(
    backbone_path,
    sequence_path,
    refined_path,
    *,
    target,
    backbones,
    sequences_per_backbone,
    hotspot_proportion,
    source_commit,
):
    if target not in REFERENCE_PERCENT or type(backbones) is not int or not 1 <= backbones <= 1000:
        raise ValueError("Unknown target or invalid backbone allocation")
    if type(sequences_per_backbone) is not int or not 1 <= sequences_per_backbone <= 20:
        raise ValueError("Invalid sequence allocation")
    if type(hotspot_proportion) not in (int, float) or hotspot_proportion not in (0, 0.1, 1):
        raise ValueError("Select one of the three published hotspot proportions")
    if not isinstance(source_commit, str) or not re.fullmatch("[0-9a-f]{40}", source_commit):
        raise ValueError("Explicit declared source commit required")
    paths = [Path(p) for p in [backbone_path, sequence_path, refined_path]]
    hashes = {str(p): sha256_file(p) for p in paths}
    parents, sequences, refined = [read_metadata(p) for p in paths]
    expected = backbones * sequences_per_backbone
    if len(parents) != backbones or len(sequences) != expected or set(refined) != {tag + "_best" for tag in sequences}:
        raise ValueError("Incomplete backbone/sequence/refinement allocation")
    groups = {name: [] for name in parents}
    indices = {name: set() for name in parents}
    rows = []
    for tag, original in sequences.items():
        match = re.fullmatch(r"(.+)_dldesign_([0-9]+)", tag)
        if match is None or match[1] not in parents or int(match[2]) in indices[match[1]]:
            raise ValueError("Unbound or duplicated sequence-to-backbone identity")
        parent, index = match[1], int(match[2])
        indices[parent].add(index)
        if original["sequences"]["T"] != parents[parent]["sequences"]["T"] or len(original["sequences"]["H"]) != len(
            parents[parent]["sequences"]["H"]
        ):
            raise ValueError("Sequence design differs from its parent target or length")
        result = refined[tag + "_best"]
        scores = result["scores"]
        required = ["interaction_pae", "pae", "target_aligned_antibody_rmsd", "pred_lddt"]
        if any(type(scores.get(k)) not in (float, int) or scores[k] < 0 for k in required):
            raise ValueError("Missing or invalid structural scores")
        native = target_aligned_ca_rmsd(original, result)
        delta = abs(native - scores["target_aligned_antibody_rmsd"])
        if delta > 0.02:
            raise ValueError("Reported RMSD disagrees with independent coordinate replay")
        passes = scores["interaction_pae"] < 10 and scores["target_aligned_antibody_rmsd"] < 2
        row = {
            "id": tag + "_best",
            "backbone_id": parent,
            "sequence_index": index,
            "interaction_pae": scores["interaction_pae"],
            "global_pae": scores["pae"],
            "reported_target_aligned_ca_rmsd": scores["target_aligned_antibody_rmsd"],
            "independent_target_aligned_ca_rmsd": native,
            "rmsd_absolute_difference": delta,
            "interface_rule_pass": passes,
            "legacy_global_rule_pass": scores["pae"] < 10 and scores["target_aligned_antibody_rmsd"] < 2,
            "near_rounded_threshold": abs(scores["interaction_pae"] - 10) <= 0.005000001
            or abs(scores["target_aligned_antibody_rmsd"] - 2) <= 0.005000001,
        }
        rows.append(row)
        groups[parent].append(row)
    if any(ids != set(range(sequences_per_backbone)) for ids in indices.values()):
        raise ValueError("Incomplete per-backbone sequence allocation")
    grouped = []
    for parent, candidates in groups.items():
        selected = min(candidates, key=lambda x: (x["interaction_pae"], x["reported_target_aligned_ca_rmsd"], x["id"]))
        grouped.append(
            {
                "backbone_id": parent,
                "selected_by_minimum_interface_pae": selected["id"],
                "selected_pass": selected["interface_rule_pass"],
                "any_sequence_pass": any(v["interface_rule_pass"] for v in candidates),
                "passing_sequences": sum(v["interface_rule_pass"] for v in candidates),
            }
        )
    matched = backbones == 200 and sequences_per_backbone == 8
    if {str(p): sha256_file(p) for p in paths} != hashes:
        raise ValueError("Source quivers changed during audit")
    return {
        "schema": 1,
        "status": "passed",
        "scope": "independent_structure_reanalysis",
        "inputs": hashes,
        "target": target,
        "declared_source_commit": source_commit,
        "allocation": {
            "backbones": backbones,
            "sequences_per_backbone": sequences_per_backbone,
            "total_sequences": expected,
            "hotspot_proportion": float(hotspot_proportion),
        },
        "paper_reference": {
            "doi": PAPER_DOI,
            "supplementary_pdf_sha256": SUPPLEMENT_SHA256,
            "table": "Supplementary Methods Table 6",
            "pdf_page": 69,
            "backbones": 200,
            "sequences_per_backbone": 8,
            "reported_percent": REFERENCE_PERCENT[target][str(float(hotspot_proportion))],
        },
        "matching_published_scale": matched,
        "exact_paper_replication_established": False,
        "selection_rule": "Minimum interaction_pae, then RMSD, then ID; any-passing-sequence result is a separate sensitivity analysis because the paper does not fully specify tie/threshold selection.",
        "sequence_passes": sum(row["interface_rule_pass"] for row in rows),
        "legacy_global_sequence_passes": sum(row["legacy_global_rule_pass"] for row in rows),
        "selected_backbone_passes": sum(row["selected_pass"] for row in grouped),
        "any_passing_sequence_backbones": sum(row["any_sequence_pass"] for row in grouped),
        "selected_backbone_pass_percent": 100 * sum(row["selected_pass"] for row in grouped) / backbones,
        "any_passing_sequence_backbone_percent": 100 * sum(row["any_sequence_pass"] for row in grouped) / backbones,
        "independent_rmsd_count": len(rows),
        "max_rmsd_absolute_difference": max(row["rmsd_absolute_difference"] for row in rows),
        "near_rounded_threshold_sequences": sum(row["near_rounded_threshold"] for row in rows),
        "score_decimal_places_in_current_author_quiver": 2,
        "coordinate_replay_tolerance_angstrom": 0.02,
        "experimental_binding_verified": False,
        "rows": rows,
        "backbone_records": grouped,
        "limitations": [
            "Structure confidence and backbone success are not experimental binding.",
            "Public implementation version, stochastic sampling and full published target coverage require separate verification.",
            "Repeated computational sequences from one backbone are not independent experimental replicates.",
            "Rounded QV scores cannot resolve candidates exactly at the thresholds.",
        ],
    }
