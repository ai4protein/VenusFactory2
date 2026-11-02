"""All-chain PHL7 C-alpha structural comparison with two alignment solvers.

No outlier removal or chain-pair selection. This is structure analysis, not
PROSS design, new MD, diffraction refinement or biological replication.
"""
from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
from Bio.SeqUtils import seq1
from scipy.spatial.transform import Rotation

EXPECTED_PDBS = {"7NEI": "WT", "9QNM": "R2M2", "9QT8": "R2M2-P155G", "9QV8": "R2M2-A186S",
                 "9QVA": "R2M2-P155G-A186S", "9QYA": "R4M6", "9QYB": "R4M10", "9QYC": "R4M12"}


def independent_rmsd(reference, mobile):
    reference, mobile = np.asarray(reference, dtype=np.float64), np.asarray(mobile, dtype=np.float64)
    if reference.shape != mobile.shape or reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 3:
        raise ValueError("invalid alignment shape")
    if not np.isfinite(reference).all() or not np.isfinite(mobile).all():
        raise ValueError("nonfinite coordinates")
    a, b = reference - reference.mean(0), mobile - mobile.mean(0)
    rotation, _ = Rotation.align_vectors(a, b)
    return float(np.sqrt(np.mean(np.sum((a - rotation.apply(b)) ** 2, axis=1))))


def chains(path):
    structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    if len(structure) != 1:
        raise ValueError("expected one crystallographic model")
    result = {}
    for chain in structure[0]:
        residues = {}
        for residue in chain:
            if residue.id[0] != " " or "CA" not in residue:
                continue
            atom = residue["CA"]
            if atom.is_disordered():
                atom = sorted(atom.disordered_get_list(), key=lambda a: (-(a.get_occupancy() or 0), a.get_altloc()))[0]
            residues[(residue.id[1], residue.id[2])] = (seq1(residue.resname), atom)
        if residues:
            result[chain.id] = residues
    return result


def run(assets, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((assets / "manifest.json").read_text())
    if len(manifest) != 8 or {x["id"] for x in manifest} != set(EXPECTED_PDBS):
        raise ValueError("incomplete published PDB set")
    for row in manifest:
        p = assets / (row["id"] + ".cif")
        if hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("PDB source changed")
    source = {code: chains(assets / (code + ".cif")) for code in EXPECTED_PDBS}
    rows = []
    for code, mobile_chains in source.items():
        if code == "7NEI":
            continue
        for ref_chain, ref in source["7NEI"].items():
            for mob_chain, mobile in mobile_chains.items():
                common = sorted(set(ref) & set(mobile))
                ref_atoms, mob_atoms = [ref[k][1] for k in common], [mobile[k][1] for k in common]
                fit = Superimposer()
                fit.set_atoms(ref_atoms, mob_atoms)
                check = independent_rmsd([a.coord for a in ref_atoms], [a.coord for a in mob_atoms])
                if abs(float(fit.rms) - check) > 1e-8:
                    raise ValueError("independent alignment disagreement")
                rows.append({"pdb": code, "variant": EXPECTED_PDBS[code], "reference_chain": ref_chain,
                             "mobile_chain": mob_chain, "paired_ca": len(common), "residue_numbers": [k[0] for k in common],
                             "rmsd_angstrom": float(fit.rms), "independent_rmsd_angstrom": check,
                             "mutation_differences": [f"{ref[k][0]}{k[0]}{mobile[k][0]}" for k in common if ref[k][0] != mobile[k][0]],
                             "reference_unpaired": len(set(ref) - set(mobile)), "mobile_unpaired": len(set(mobile) - set(ref))})
    report = {"status": "completed", "paper_doi": "10.1038/s41467-026-70868-4", "sources": manifest,
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "method": "all protein chains; auth residue number + insertion code; highest occupancy CA (alphabetic altloc tie break); least-squares proper rotation; no outlier rejection",
              "published_comparison": {"variant": "R2M2", "ca_atoms": 258, "reported_rmsd_angstrom": 0.60,
                                       "chain_pair_and_outlier_policy": "not specified in main-text result"},
              "alignments": rows, "alignment_count": len(rows), "new_model_or_md_generated": False,
              "full_paper_reproduced": False}
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "analysis.py")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.assets, args.output)
    print(json.dumps({"alignments": result["alignment_count"], "R2M2": [x for x in result["alignments"] if x["variant"] == "R2M2"]}, indent=2))
