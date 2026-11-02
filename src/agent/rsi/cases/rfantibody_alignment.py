"""Compare VHH alignment metrics without treating structure scores as binding."""

from __future__ import annotations

import math
from statistics import mean


def compare_alignment_rows(rows, legacy_rmsds):
    """Compare both metrics over exactly the same structures and backbone groups."""
    if not rows:
        raise ValueError("No structures supplied")
    identities = [row["id"] for row in rows]
    if len(set(identities)) != len(identities) or set(identities) != set(legacy_rmsds):
        raise ValueError("Alignment identities must match exactly without duplicates")
    compared = []
    groups = {}
    for row in rows:
        old = legacy_rmsds[row["id"]]
        new = row["independent_target_aligned_ca_rmsd"]
        pae = row["interaction_pae"]
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in [old, new, pae]):
            raise ValueError("Finite nonnegative metrics required")
        current_pass = pae < 10 and new < 2
        legacy_pass = pae < 10 and old < 2
        item = {
            "id": row["id"],
            "backbone_id": row["backbone_id"],
            "interaction_pae": pae,
            "current_kabsch_ca_rmsd": new,
            "legacy_usalign_ca_rmsd": old,
            "absolute_difference": abs(old - new),
            "current_interface_rule_pass": current_pass,
            "legacy_alignment_interface_rule_pass": legacy_pass,
            "rmsd_threshold_changed": (new < 2) != (old < 2),
            "combined_filter_changed": current_pass != legacy_pass,
        }
        compared.append(item)
        groups.setdefault(item["backbone_id"], []).append(item)
    grouped = []
    for parent, candidates in sorted(groups.items()):
        current = min(candidates, key=lambda r: (r["interaction_pae"], r["current_kabsch_ca_rmsd"], r["id"]))
        legacy = min(candidates, key=lambda r: (r["interaction_pae"], r["legacy_usalign_ca_rmsd"], r["id"]))
        grouped.append(
            {
                "backbone_id": parent,
                "current_selected_id": current["id"],
                "legacy_alignment_selected_id": legacy["id"],
                "current_selected_pass": current["current_interface_rule_pass"],
                "legacy_alignment_selected_pass": legacy["legacy_alignment_interface_rule_pass"],
                "current_any_pass": any(r["current_interface_rule_pass"] for r in candidates),
                "legacy_alignment_any_pass": any(r["legacy_alignment_interface_rule_pass"] for r in candidates),
            }
        )
    return {
        "scope": "alignment_only_sensitivity_on_fixed_structures_and_current_interface_pae",
        "structures": len(compared),
        "backbones": len(grouped),
        "maximum_rmsd_difference": max(r["absolute_difference"] for r in compared),
        "mean_rmsd_difference": mean(r["absolute_difference"] for r in compared),
        "rmsd_threshold_changes": sum(r["rmsd_threshold_changed"] for r in compared),
        "combined_filter_changes": sum(r["combined_filter_changed"] for r in compared),
        "current_sequence_passes": sum(r["current_interface_rule_pass"] for r in compared),
        "legacy_alignment_sequence_passes": sum(r["legacy_alignment_interface_rule_pass"] for r in compared),
        "current_selected_backbone_passes": sum(r["current_selected_pass"] for r in grouped),
        "legacy_alignment_selected_backbone_passes": sum(r["legacy_alignment_selected_pass"] for r in grouped),
        "current_any_passing_backbones": sum(r["current_any_pass"] for r in grouped),
        "legacy_alignment_any_passing_backbones": sum(r["legacy_alignment_any_pass"] for r in grouped),
        "selected_design_changes": sum(r["current_selected_id"] != r["legacy_alignment_selected_id"] for r in grouped),
        "rows": compared,
        "backbone_records": grouped,
        "limitations": [
            "Both alignments use the same current-version PAE; legacy PAE cannot be recovered from scalar summaries.",
            "This does not replay old neural-network inference or establish the unpublished experimental implementation.",
            "Structural filter passes do not establish experimental binding.",
        ],
    }
