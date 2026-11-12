"""Verify SKEMPI's cleaned coordinates and original-to-cleaned residue maps.

No tar members are extracted. Sequences represent observed structure residues,
not complete UniProt proteins; missing/modified residues are never invented.
"""
from __future__ import annotations

import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

from .skempi_source import SOURCE, read_regular, require, sha256

ARCHIVE = "SKEMPI2_PDBs.tgz"
URL = SOURCE + "database/download/" + ARCHIVE
THREE = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
ONE = "ARNDCQEGHILKMFPSTWYV"
AMINO = dict(zip(THREE, ONE))


def read_structures(manifest_path, expected_sha256):
    path = Path(manifest_path).absolute()
    raw = read_regular(path, 1024**2)
    require(sha256(raw) == expected_sha256, "Structure manifest checksum changed")
    manifest = json.loads(raw)
    require(manifest.get("schema") == "skempi_structure_snapshot.v1"
            and manifest.get("license") == "CC-BY-4.0", "Unknown structure snapshot")
    urls = {ARCHIVE: URL, "terms.html": SOURCE + "info/terms", "faq.html": SOURCE + "info/faq_and_help"}
    files = manifest.get("files", [])
    require(len(files) == len(urls) and {f.get("file") for f in files} == set(urls), "Incomplete structure/license snapshot")
    contents = {}
    for item in files:
        name = item["file"]
        require(item.get("url") == urls[name] and item.get("final_url") == urls[name], "Unexpected structure/document URL")
        body = read_regular(path.parent / name, 64 * 1024**2 if name == ARCHIVE else 1024**2)
        require(item.get("bytes") == len(body) and item.get("sha256") == sha256(body), "Structure/document asset changed")
        contents[name] = body
    members = {}
    with tarfile.open(fileobj=io.BytesIO(contents[ARCHIVE]), mode="r:gz") as archive:
        total = 0
        for index, entry in enumerate(archive):
            require(index < 3000 and entry.size <= 8 * 1024**2, "Archive member limits exceeded")
            total += entry.size
            require(total <= 256 * 1024**2, "Expanded archive exceeds limit")
            path_parts = Path(entry.name).parts
            require(not entry.name.startswith("/") and ".." not in path_parts and not entry.issym()
                    and not entry.islnk() and (entry.isfile() or entry.isdir()), "Unsafe archive member")
            if entry.isdir() or Path(entry.name).name.startswith("._"):
                continue
            require(re.fullmatch(r"PDBs/[0-9A-Za-z]{4}\.(pdb|mapping)", entry.name), "Unexpected archive asset")
            require(entry.name not in members, "Duplicate archive member")
            members[entry.name] = archive.extractfile(entry).read()
    return {"manifest_sha256": expected_sha256, "archive_sha256": sha256(contents[ARCHIVE]),
            "archive_bytes": len(contents[ARCHIVE]), "url": URL}, members


def parse_pair(pdb, mapping):
    residues = {}
    for line in pdb.decode("ascii").splitlines():
        if line.startswith(("MODEL ", "ENDMDL")):
            raise ValueError("Multiple-model semantics require explicit review")
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        require(len(line) >= 54, "Truncated coordinate record")
        chain, number, insertion = line[21], int(line[22:26]), line[26].strip()
        key = (chain, number, insertion)
        name = line[17:20].strip()
        records = residues.setdefault(key, {})
        record = line[:6].strip()
        records.setdefault(record, set()).add(name)
        require(len(records.get("ATOM", set())) <= 1, "Ambiguous polymer residue identity")
    original, cleaned = {}, {}
    order = {}
    for line in mapping.decode("ascii").splitlines():
        match = re.fullmatch(r"([A-Z0-9]{3}) ([A-Za-z0-9])\s*(-?\d+)([A-Za-z]?)\s+(\d+)\s*", line)
        require(match is not None, "Unrecognized original-to-cleaned mapping line")
        name, chain, old, insertion, new = match.groups()
        old_key, new_key = (chain, int(old), insertion), (chain, int(new), "")
        require(old_key not in original and new_key not in cleaned, "Non-bijective residue mapping")
        records = residues.get(new_key, {})
        # Cleaned SKEMPI files retain waters/ions whose HETATM numbers can
        # overlap the renumbered polymer. They are not amino-acid residues.
        identities = records.get("ATOM", records.get("HETATM", set()))
        require(identities == {name}, "Mapping disagrees with coordinate residue")
        original[old_key] = {"cleaned": new_key, "amino_acid": AMINO.get(name), "residue_name": name}
        cleaned[new_key] = original[old_key]
        order.setdefault(chain, []).append(new_key)
    require(original, "Empty residue mapping")
    chains = {}
    for chain, keys in order.items():
        require([k[1] for k in keys] == list(range(1, len(keys) + 1)), "Cleaned residues are not contiguous from one")
        unknown = [cleaned[key]["residue_name"] for key in keys if cleaned[key]["amino_acid"] is None]
        chains[chain] = {"observed_sequence": "".join(cleaned[key]["amino_acid"] or "X" for key in keys),
                         "nonstandard_residues": unknown, "source": "cleaned_structure_observed_residues"}
    return {"original": original, "cleaned": cleaned, "chains": chains}


def attach_mapping(observations, members):
    cache, failures, counts = {}, {}, Counter()
    for row in observations:
        pdb_id = row["pdb_id"]
        if pdb_id not in cache and pdb_id not in failures:
            try:
                pdb_name, map_name = f"PDBs/{pdb_id}.pdb", f"PDBs/{pdb_id}.mapping"
                parsed = parse_pair(members[pdb_name], members[map_name])
                parsed["hashes"] = {"pdb_sha256": sha256(members[pdb_name]), "mapping_sha256": sha256(members[map_name])}
                cache[pdb_id] = parsed
            except (KeyError, ValueError) as exc:
                failures[pdb_id] = type(exc).__name__ + ": " + str(exc)[:180]
        issues, links = [], []
        if pdb_id in failures:
            issues.append("structure_unverified: " + failures[pdb_id])
        else:
            parsed = cache[pdb_id]
            required = list("".join(row["partner_chains"]))
            for chain in required:
                if chain not in parsed["chains"]:
                    issues.append("partner_chain_missing:" + chain)
                elif parsed["chains"][chain]["nonstandard_residues"]:
                    issues.append("partner_nonstandard_residues:" + chain)
            for original, cleaned in zip(row["mutations_pdb"], row["mutations_cleaned"]):
                key = (original["chain"], original["residue_number"], original["insertion_code"])
                desired = (cleaned["chain"], cleaned["residue_number"], cleaned["insertion_code"])
                actual = parsed["original"].get(key)
                if original["repeated_site"] or original["unchanged"]:
                    issues.append("ambiguous_repeated_or_unchanged_mutation:" + original["raw"])
                if actual is None:
                    issues.append("original_position_unmapped:" + original["raw"])
                elif actual["cleaned"] != desired:
                    issues.append("source_cleaned_number_mismatch:" + original["raw"])
                elif actual["amino_acid"] != original["wild_type"]:
                    issues.append("wild_type_mismatch:" + original["raw"])
                else:
                    links.append({"source_mutation": original["raw"], "cleaned_mutation": cleaned["raw"],
                                  "sequence_index": cleaned["residue_number"] - 1})
            row["structure_hashes"] = parsed["hashes"]
            row["partner_sequences"] = {chain: parsed["chains"][chain] for chain in required if chain in parsed["chains"]}
        row["sequence_mapping_status"] = "verified_observed_residues" if not issues else "requires_review"
        row["sequence_mapping_issues"] = issues
        row["mutation_mapping"] = links
        row["training_eligible"] = False
        counts[row["sequence_mapping_status"]] += 1
    return {"rows": dict(counts), "coordinate_sets": len(cache), "failed_coordinate_sets": failures,
            "complete_protein_sequences": False,
            "limitations": "Only observed coordinates are mapped. Nonstandard partner residues and source inconsistencies are excluded from automatic sequence use."}
