"""Lossless SKEMPI2 source staging; no training admission or row-wise split.

Only public, hash-bound source files are read. Structure/sequence correspondence
and a reviewed biological split remain required after this conversion.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "skempi_source.v1"
SOURCE = "https://life.bsc.es/pid/skempi2/"
FILE = "skempi_v2.csv"
URL = SOURCE + "database/download/" + FILE
LICENSE = SOURCE + "info/terms"
PAPER = "https://doi.org/10.1093/bioinformatics/bty635"
TYPES = {"Pr/PI", "AB/AG", "TCR/pMHC"}
AA = "ACDEFGHIKLMNPQRSTVWY"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
COLUMNS = [
    "#Pdb", "Mutation(s)_PDB", "Mutation(s)_cleaned", "iMutation_Location(s)",
    "Hold_out_type", "Hold_out_proteins", "Affinity_mut (M)", "Affinity_mut_parsed",
    "Affinity_wt (M)", "Affinity_wt_parsed", "Reference", "Protein 1", "Protein 2",
    "Temperature", "kon_mut (M^(-1)s^(-1))", "kon_mut_parsed",
    "kon_wt (M^(-1)s^(-1))", "kon_wt_parsed", "koff_mut (s^(-1))",
    "koff_mut_parsed", "koff_wt (s^(-1))", "koff_wt_parsed",
    "dH_mut (kcal mol^(-1))", "dH_wt (kcal mol^(-1))",
    "dS_mut (cal mol^(-1) K^(-1))", "dS_wt (cal mol^(-1) K^(-1))",
    "Notes", "Method", "SKEMPI version",
]
ENDPOINTS = {
    "kd": ("Affinity_{} (M)", "M", "Affinity_{}_parsed", True),
    "kon": ("kon_{} (M^(-1)s^(-1))", "M^-1 s^-1", "kon_{}_parsed", True),
    "koff": ("koff_{} (s^(-1))", "s^-1", "koff_{}_parsed", True),
    "enthalpy": ("dH_{} (kcal mol^(-1))", "kcal mol^-1", None, False),
    "entropy": ("dS_{} (cal mol^(-1) K^(-1))", "cal mol^-1 K^-1", None, False),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path, limit):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Source paths cannot use symlinks")
    require(path.is_file() and 0 < path.stat().st_size <= limit, "Missing or oversized source file")
    return path.read_bytes()


def read_snapshot(manifest_path, expected_sha256):
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", expected_sha256),
            "Supply the frozen source manifest SHA-256")
    path = Path(manifest_path).absolute()
    raw = read_regular(path, 1024**2)
    require(sha256(raw) == expected_sha256, "Source manifest checksum changed")
    manifest = json.loads(raw)
    require(manifest.get("schema") == "property_source_snapshot.v1", "Unknown source manifest")
    entries = [item for item in manifest.get("files", []) if item.get("source") == "skempi2"]
    require(len(entries) == 1, "Snapshot requires exactly one SKEMPI2 table")
    item = entries[0]
    require(item.get("file") == FILE and item.get("url") == URL and item.get("final_url", URL) == URL,
            "Unexpected SKEMPI2 source path or URL")
    require(item.get("status") == "downloaded_not_admitted", "Unexpected snapshot state")
    content = read_regular(path.parent / FILE, 16 * 1024**2)
    require(item.get("bytes") == len(content) and item.get("sha256") == sha256(content),
            "Source table checksum or size changed")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), delimiter=";")
    require(reader.fieldnames == COLUMNS, "Unexpected, reordered or duplicate SKEMPI2 columns")
    rows = []
    for line, row in enumerate(reader, 2):
        require(line <= 20001 and None not in row and all(v is not None for v in row.values()),
                "Malformed or oversized SKEMPI2 table")
        rows.append(row)
    require(rows and type(item.get("rows")) is int and item["rows"] == len(rows), "Source row count changed")
    return {"manifest_sha256": expected_sha256, "sha256": sha256(content), "bytes": len(content),
            "rows": len(rows), "url": URL, "file": FILE}, rows


def mutation_list(raw):
    result, sites = [], set()
    for token in raw.split(","):
        match = re.fullmatch(r"([" + AA + r"])([A-Za-z0-9])(-?\d+)([A-Za-z]?)([" + AA + r"])", token)
        require(match is not None, "Unrecognized source mutation: " + token)
        wt, chain, number, insertion, mutant = match.groups()
        site = (chain, int(number), insertion)
        result.append({"raw": token, "wild_type": wt, "chain": chain, "residue_number": int(number),
                       "insertion_code": insertion, "mutant": mutant,
                       "repeated_site": site in sites, "unchanged": wt == mutant})
        sites.add(site)
    return result


def measurement(raw, unit, *, positive, parsed=None):
    result = {"raw": raw, "unit": unit, "kind": "missing", "value": None}
    if parsed is not None:
        result["source_parsed"] = parsed
    if not raw:
        return result
    if raw in {"n.b", "n.b."}:
        result["kind"] = "no_detectable_binding"
        return result
    if raw == "unf":
        result["kind"] = "unfolded"
        return result
    match = re.fullmatch(r"([<>~]?)(" + NUMBER + ")", raw)
    if match is None:
        result["kind"] = "unparsed"
        return result
    qualifier, number = match.groups()
    value = float(number)
    require(math.isfinite(value), "Non-finite source measurement")
    result.update(kind={"": "exact", "<": "upper_bound", ">": "lower_bound", "~": "approximate"}[qualifier],
                  value=value)
    if positive and value <= 0:
        result["kind"] = "invalid_nonpositive"
    if parsed:
        compatible = re.fullmatch(NUMBER, parsed) is not None
        result["parsed_disagrees"] = not compatible or not math.isclose(value, float(parsed), rel_tol=1e-12, abs_tol=0)
    return result


def temperature(raw):
    match = re.fullmatch(r"(" + NUMBER + r")(\(assumed\))?", raw)
    if not match:
        return {"raw": raw, "kelvin": None, "kind": "missing" if not raw else "unparsed"}
    value = float(match.group(1))
    require(math.isfinite(value) and value > 0, "Invalid Kelvin temperature")
    return {"raw": raw, "kelvin": value, "kind": "assumed" if match.group(2) else "measured"}


def normalize(row, line):
    parts = row["#Pdb"].split("_")
    require(len(parts) == 3 and re.fullmatch(r"[0-9A-Za-z]{4}", parts[0])
            and all(re.fullmatch(r"[A-Za-z0-9]+", p) for p in parts[1:]), "Invalid complex/chain identifier")
    require(set(parts[1]).isdisjoint(parts[2]), "Partner chains overlap")
    original, cleaned = mutation_list(row["Mutation(s)_PDB"]), mutation_list(row["Mutation(s)_cleaned"])
    require(len(original) == len(cleaned), "Original/cleaned mutation counts differ")
    require(all((a["wild_type"], a["chain"], a["mutant"]) == (b["wild_type"], b["chain"], b["mutant"])
                for a, b in zip(original, cleaned)), "Original/cleaned mutation identities disagree")
    require(all(m["chain"] in parts[1] + parts[2] for m in original), "Mutation chain absent from complex")
    endpoints = {name: {variant: measurement(row[column.format(variant)], unit, positive=positive,
                                            parsed=row[parsed.format(variant)] if parsed else None)
                        for variant in ("wt", "mut")}
                 for name, (column, unit, parsed, positive) in ENDPOINTS.items()}
    temp = temperature(row["Temperature"])
    problems = []
    for variant in ("wt", "mut"):
        kd = endpoints["kd"][variant]
        if kd["kind"] != "exact":
            problems.append("kd_" + variant + "_" + kd["kind"])
        if kd.get("parsed_disagrees"):
            problems.append("kd_" + variant + "_parsed_disagrees")
    if temp["kind"] != "measured":
        problems.append("temperature_" + temp["kind"])
    ddg = None if problems else (8.314 / 4184) * temp["kelvin"] * (
        math.log(endpoints["kd"]["mut"]["value"]) - math.log(endpoints["kd"]["wt"]["value"]))
    return {"id": f"skempi2:line:{line}", "source_line": line, "raw": dict(row),
            "complex_id": row["#Pdb"], "pdb_id": parts[0], "partner_chains": parts[1:],
            "mutations_pdb": original, "mutations_cleaned": cleaned,
            "sequence_mapping_status": "not_verified", "measurements": endpoints, "temperature": temp,
            "derived_ddg": {"value": ddg, "unit": "kcal mol^-1", "definition": "G_mut - G_wt",
                            "formula": "R*T*(ln(Kd_mut)-ln(Kd_wt)); R=8.314/4184",
                            "excluded_reasons": problems},
            "duplicate_fingerprint": sha256(canonical(row).encode()),
            "mutation_group": sha256(canonical([row["#Pdb"], sorted(m["raw"] for m in original)]).encode())}


def exclusion_groups(observations):
    """Conservative connected components, not a claim of sequence-cluster independence."""
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        aa, bb = find(a), find(b)
        parent[max(aa, bb)] = min(aa, bb)
    known = {row["complex_id"] for row in observations}
    for row in observations:
        raw = row["raw"]
        reference = raw["Reference"].strip()
        require(reference, "Missing original-study reference")
        source = "complex:" + row["complex_id"]
        # Same study and all source-defined complex/type exclusions stay together.
        union(source, "study:" + reference)
        if row.get("sequence_mapping_status") == "verified_observed_residues":
            for chain in row["partner_sequences"].values():
                union(source, "sequence:" + sha256(chain["observed_sequence"].encode()))
        for cluster in row.get("sequence_clusters", []):
            union(source, "homology:" + cluster)
        for token in filter(None, raw["Hold_out_type"].split(",")):
            require(token in TYPES, "Unknown hold-out type")
            union(source, "type:" + token)
        for token in filter(None, raw["Hold_out_proteins"].split(",")):
            require(token in known or token in TYPES, "Unknown hold-out protein/type")
            union(source, ("type:" if token in TYPES else "complex:") + token)
    components = defaultdict(list)
    for key in sorted(parent):
        components[find(key)].append(key)
    groups = {root: sha256(canonical(keys).encode()) for root, keys in components.items()}
    counts = Counter()
    for row in observations:
        row["exclusion_group"] = groups[find("complex:" + row["complex_id"])]
        counts[row["exclusion_group"]] += 1
    return {"count": len(counts), "sizes": sorted(counts.values(), reverse=True),
            "criteria": ["same_complex", "source_holdout_proteins", "source_holdout_type", "same_original_study", "identical_observed_partner_sequence",
                         *(["observed_partner_alignment_cluster"] if any(row.get("sequence_clusters") for row in observations) else [])],
            "sequence_similarity_verified": any(row.get("sequence_clusters") for row in observations), "split_assigned": False}


def export_source(manifest_path, expected_sha256, output, *, structure_manifest=None, structure_sha256=None):
    asset, rows = read_snapshot(manifest_path, expected_sha256)
    observations = [normalize(row, line) for line, row in enumerate(rows, 2)]
    structures = None
    require(bool(structure_manifest) == bool(structure_sha256), "Supply both structure manifest and checksum")
    if structure_manifest:
        from .skempi_structures import attach_mapping, read_structures
        structure_asset, members = read_structures(structure_manifest, structure_sha256)
        structures = {"source": structure_asset, "mapping": attach_mapping(observations, members)}
    groups = exclusion_groups(observations)
    exact = Counter(row["duplicate_fingerprint"] for row in observations)
    repeated = Counter(row["mutation_group"] for row in observations)
    counts = Counter(reason for row in observations for reason in row["derived_ddg"]["excluded_reasons"])
    summary = {
        "observations": len(observations), "complexes": len({r["complex_id"] for r in observations}),
        "studies": len({r["raw"]["Reference"].strip() for r in observations}),
        "exact_duplicate_excess": sum(n - 1 for n in exact.values()),
        "repeated_complex_mutation_excess": sum(n - 1 for n in repeated.values()),
        "exact_ddg_conversions": sum(r["derived_ddg"]["value"] is not None for r in observations),
        "ddg_exclusions": dict(sorted(counts.items())), "exclusion_groups": groups,
        "admitted_tasks": 0, "new_models": 0, "external_model_requests": 0,
    }
    report = {"schema": SCHEMA, "status": "normalized_not_admitted", "source": asset,
              "attribution": {"database": "SKEMPI 2.0", "authors": "Jankauskaite, Jimenez-Garcia, Dapkunas, Fernandez-Recio and Moal",
                              "paper": PAPER, "license": "CC-BY-4.0", "license_reference": LICENSE,
                              "modifications": "Lossless typed normalization, exact-measurement ddG derivation, conservative exclusion graph"},
              "summary": summary,
              "admission_blockers": ["Verify source PDB/cleaned residue mapping, WT identities and both partner sequences.",
                                     "Review homologous protein/complex grouping in addition to source holdouts and studies.",
                                     "Freeze a viable biological split; do not break connected groups to meet sample quotas.",
                                     "Review repeated measurements, assay conditions, source notes and each endpoint before selecting targets."],
              "structures": structures, "observations_file": "observations.jsonl"}
    output = Path(output).absolute()
    require(not output.exists() and not any(p.is_symlink() for p in output.parents), "Output must be a new non-symlink directory")
    output.mkdir(parents=True)
    body = "".join(canonical(row) + "\n" for row in observations).encode()
    (output / "observations.jsonl").write_bytes(body)
    report["observations_sha256"] = sha256(body)
    (output / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    return report
