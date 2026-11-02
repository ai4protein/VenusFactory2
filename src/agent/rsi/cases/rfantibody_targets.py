"""Explicit structure numbering and provenance for VHH reconstruction inputs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .rfantibody_audit import AA3
from agent.rsi.engine.storage import sha256_file

TARGETS = ("ha", "rsv_site3", "tcdb", "il7r", "rbd")


def pdb_residues(path):
    """Read the first model, canonical ATOM records, blank/A alternate locations."""
    records = {}
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  ") or line[16] not in {" ", "A"}:
                continue
            residue = line[17:20]
            if residue not in AA3:
                raise ValueError("Noncanonical ATOM residue requires an explicit policy")
            key = (line[21], int(line[22:26]), line[26].strip())
            atom = line[12:16].strip()
            xyz = tuple(float(line[k : k + 8]) for k in (30, 38, 46))
            if not all(math.isfinite(x) for x in xyz):
                raise ValueError("Nonfinite atom coordinate")
            record = records.setdefault(key, {"aa": AA3[residue], "atoms": {}, "lines": {}})
            if record["aa"] != AA3[residue]:
                raise ValueError("Conflicting residue identity")
            if atom in record["atoms"]:
                raise ValueError("Ambiguous blank/A alternate atom")
            record["atoms"][atom] = xyz
            record["lines"][atom] = line
    if not records or any("CA" not in r["atoms"] for r in records.values()):
        raise ValueError("Empty structure or residue without C-alpha")
    return records


def exact_ca_subset(reference, candidate, tolerance=0.01):
    """Require a unique residue-preserving, unchanged-coordinate subset mapping."""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.1:
        raise ValueError("Invalid coordinate tolerance")
    keys = list(reference)
    coordinates = np.asarray([reference[k]["atoms"]["CA"] for k in keys])
    result = []
    used = set()
    for key, record in candidate.items():
        distances = np.linalg.norm(coordinates - np.asarray(record["atoms"]["CA"]), axis=1)
        indices = np.flatnonzero(distances <= tolerance)
        if len(indices) != 1:
            raise ValueError("Missing or ambiguous coordinate correspondence")
        index = int(indices[0])
        mapped = keys[index]
        if mapped in used or reference[mapped]["aa"] != record["aa"]:
            raise ValueError("Repeated correspondence or residue identity mismatch")
        used.add(mapped)
        result.append(
            {
                "candidate": list(key),
                "reference": list(mapped),
                "aa": record["aa"],
                "ca_distance": float(distances[index]),
            }
        )
    return result


def resolve_cif_hotspots(cif_path, chain, residue_numbers, *, chain_namespace):
    """Resolve author residue numbers under an explicitly declared chain namespace."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    if chain_namespace not in {"auth", "label"}:
        raise ValueError("Declare author or label chain namespace")
    if not residue_numbers or len(set(residue_numbers)) != len(residue_numbers):
        raise ValueError("Distinct hotspot residues required")
    data = MMCIF2Dict(str(cif_path))
    rows = []
    for number in residue_numbers:
        if type(number) is not int:
            raise ValueError("Integer author residue number required")
        matches = []
        for i, atom in enumerate(data["_atom_site.label_atom_id"]):
            if atom != "CA" or data["_atom_site.group_PDB"][i] != "ATOM":
                continue
            if data.get("_atom_site.pdbx_PDB_model_num", ["1"] * len(data["_atom_site.label_atom_id"]))[i] != "1":
                continue
            if data["_atom_site.label_alt_id"][i] not in {".", "A"}:
                continue
            if data[f"_atom_site.{chain_namespace}_asym_id"][i] != chain or data["_atom_site.auth_seq_id"][i] != str(
                number
            ):
                continue
            if data["_atom_site.pdbx_PDB_ins_code"][i] not in {"?", "."}:
                raise ValueError("Insertion-coded hotspot requires an explicit identifier")
            matches.append(
                {
                    "declared_chain": chain,
                    "chain_namespace": chain_namespace,
                    "auth_chain": data["_atom_site.auth_asym_id"][i],
                    "label_chain": data["_atom_site.label_asym_id"][i],
                    "auth_residue_number": number,
                    "label_residue_number": int(data["_atom_site.label_seq_id"][i]),
                    "aa": AA3[data["_atom_site.label_comp_id"][i]],
                    "ca": [float(data[f"_atom_site.Cartn_{axis}"][i]) for axis in "xyz"],
                }
            )
        if len(matches) != 1:
            raise ValueError("Hotspot must resolve to exactly one observed residue")
        rows.extend(matches)
    return rows


def write_observed_chain(source, destination, source_chain, output_chain):
    """Preserve an entire observed chain's coordinates, residue numbers and gaps."""
    if len(output_chain) != 1 or not output_chain.isalpha() or not output_chain.isupper():
        raise ValueError("Single uppercase output chain required")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Refusing to overwrite an admitted input")
    records = pdb_residues(source)
    selected = {key: row for key, row in records.items() if key[0] == source_chain}
    if not selected or any(not {"N", "CA", "C"}.issubset(r["atoms"]) for r in selected.values()):
        raise ValueError("Observed chain must have complete backbone N/CA/C atoms")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        for row in selected.values():
            for line in row["lines"].values():
                stream.write(line[:16] + " " + line[17:21] + output_chain + line[22:])
        stream.write("TER\nEND\n")
    rebuilt = pdb_residues(destination)
    mapping = exact_ca_subset(records, rebuilt, tolerance=0.0)
    return {
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "source_chain": source_chain,
        "output_chain": output_chain,
        "observed_residues": len(selected),
        "policy": "Whole deposited observed target chain; no additional geometric crop, sequence repair, sidechain construction or relaxation. Preserve observed residue gaps.",
        "coordinate_mapping": mapping,
        "publication_crop_equivalence_established": False,
    }


def validate_target_input(root, record):
    root = Path(root).resolve()
    declared = root / record["path"]
    path = declared.resolve()
    path.relative_to((root / "research_runs/rsi").resolve())
    if declared.is_symlink() or sha256_file(path) != record["sha256"]:
        raise ValueError("Admitted target changed")
    residues = pdb_residues(path)
    if len(residues) != record["residues"]:
        raise ValueError("Target residue allocation changed")
    seen = set()
    for hotspot in record["hotspots"]:
        key = (hotspot["chain"], hotspot["number"], "")
        if key in seen or key not in residues or residues[key]["aa"] != hotspot["aa"]:
            raise ValueError("Hotspot identity is missing, duplicated or changed")
        seen.add(key)
    if not seen:
        raise ValueError("Explicit target hotspots required")
    return {"residues": len(residues), "hotspots": len(seen), "sha256": sha256_file(path)}
