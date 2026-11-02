"""Sequence-preserving VHH structure audit and explicit computational filters."""

from __future__ import annotations

import math
import re

from .case_workflow import audit_quiver
from .storage import atomic_json, sha256_file

AA3 = dict(
    zip(
        "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
        "ACDEFGHIKLMNPQRSTVWY",
    )
)


def read_quiver(path):
    records = {}
    current = None
    for line in path.read_text().splitlines():
        if line.startswith("QV_TAG "):
            current = line.split()[1]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", current) or current in records:
                raise ValueError("unsafe or duplicate design id")
            records[current] = {"pdb": [], "scores": {}, "residues": {}}
        elif current is not None:
            record = records[current]
            if line.startswith("QV_SCORE "):
                _, tag, values = line.split(maxsplit=2)
                if tag != current:
                    raise ValueError("score identity differs")
                for item in values.split("|"):
                    key, val = item.split("=", 1)
                    number = float(val)
                    if not math.isfinite(number) and key not in {
                        "framework_aligned_L1_rmsd",
                        "framework_aligned_L2_rmsd",
                        "framework_aligned_L3_rmsd",
                    }:
                        raise ValueError("nonfinite applicable score")
                    record["scores"][key] = number if math.isfinite(number) else None
            else:
                record["pdb"].append(line)
                if line.startswith("ATOM "):
                    key = (line[21], line[22:27])
                    aa = AA3.get(line[17:20])
                    if aa is None:
                        raise ValueError("unknown amino acid")
                    if key in record["residues"] and record["residues"][key] != aa:
                        raise ValueError("conflicting residue identity")
                    record["residues"][key] = aa
    for record in records.values():
        record["sequences"] = {
            chain: "".join(aa for (c, _), aa in record["residues"].items() if c == chain)
            for chain in sorted({c for c, _ in record["residues"]})
        }
        if set(record["sequences"]) != {"H", "T"}:
            raise ValueError("VHH audit requires heavy and target chains, no light chain")
    return records


def export_audit(sequence_quiver, refined_quiver, output, expected_count):
    audit_quiver(sequence_quiver, expected_count)
    audit_quiver(refined_quiver, expected_count)
    before, after = read_quiver(sequence_quiver), read_quiver(refined_quiver)
    if {x + "_best" for x in before} != set(after):
        raise ValueError("sequence-to-refinement identities differ")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    fasta = []
    for tag, record in after.items():
        if record["sequences"] != before[tag.removesuffix("_best")]["sequences"]:
            raise ValueError("sequence changed during refinement")
        scores = record["scores"]
        required = ["pae", "interaction_pae", "pred_lddt", "target_aligned_antibody_rmsd"]
        if any(scores.get(k) is None for k in required):
            raise ValueError("missing quality scores")
        passes = scores["pae"] < 10 and scores["target_aligned_antibody_rmsd"] < 2
        interface_passes = passes and scores["interaction_pae"] < 10
        rows.append(
            {
                "id": tag,
                "scores": scores,
                "sequence_lengths": {k: len(v) for k, v in record["sequences"].items()},
                "author_minimum_mapping_pass": passes,
                "additional_interface_filter_pass": interface_passes,
            }
        )
        (output / (tag + ".pdb")).write_text("\n".join(record["pdb"]) + "\n")
        fasta.extend([">" + tag, record["sequences"]["H"]])
    (output / "nanobodies.fasta").write_text("\n".join(fasta) + "\n")
    report = {
        "status": "completed",
        "inputs": {str(p): sha256_file(p) for p in [sequence_quiver, refined_quiver]},
        "designs": len(rows),
        "sequence_preservation_verified": True,
        "filter_mapping": "README pAE < 10 mapped to RF2 global pae; RMSD < 2 A mapped to target_aligned_antibody_rmsd. Interface pAE < 10 also reported.",
        "filter_timing": "post-run descriptive audit of fixed batch; no statistical efficacy inference",
        "light_chain_metrics": "not applicable for VHH; null, not zero",
        "experimental_binding_claim": False,
        "author_minimum_passes": sum(x["author_minimum_mapping_pass"] for x in rows),
        "additional_interface_passes": sum(x["additional_interface_filter_pass"] for x in rows),
        "records": rows,
    }
    atomic_json(output / "audit.json", report)
    return report
