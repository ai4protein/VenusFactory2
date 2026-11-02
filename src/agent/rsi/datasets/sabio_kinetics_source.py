"""Import source-bound plant enzyme kinetics without treating rows as models.

Raw and normalized units are checked independently. Conditions, publications
and substrate associations stay attached to each measurement. This stage does
not choose model partitions, aggregate labels or grant task admission.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "sabio_plant_kinetics.v1"
ENDPOINT = "https://sabiork.h-its.org/api/ft/proxy-select"
ORGANISMS = {
    "Arabidopsis thaliana": {3702},
    "Zea mays": {4577},
    "Oryza sativa": {4530, 39947, 4538},
    "Solanum lycopersicum": {4081},
    "Triticum aestivum": {4565},
    "Hordeum vulgare": {4513},
    "Spinacia oleracea": {3562},
}
# A declared panel of primary-metabolism enzymes, not all enzymes of each taxon.
EC_NUMBERS = {
    "1.1.1.22",
    "1.1.1.25",
    "1.1.1.37",
    "1.1.1.40",
    "1.1.1.82",
    "1.1.1.86",
    "1.2.1.8",
    "1.2.1.12",
    "1.2.1.13",
    "1.3.1.105",
    "1.18.1.2",
    "2.2.1.1",
    "2.5.1.47",
    "2.6.1.2",
    "2.7.1.1",
    "2.7.1.2",
    "2.7.1.11",
    "2.7.1.40",
    "2.7.7.9",
    "2.7.9.1",
    "3.2.1.20",
    "3.2.1.21",
    "3.2.1.25",
    "3.2.1.26",
    "3.2.1.74",
    "3.2.1.78",
    "3.2.1.182",
    "4.2.1.1",
    "5.1.3.1",
    "5.1.3.2",
    "5.3.1.6",
    "5.4.2.12",
    "6.3.2.2",
    "6.3.2.3",
    "6.3.4.4",
}
CANONICAL_UNITS = {"kcat": "s^(-1)", "Km": "M", "kcat/Km": "M^(-1)*s^(-1)"}
ACCESSION = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[1-9][0-9]*)?")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def unit_factor(parameter, unit):
    """Multipliers to s^-1, M, or M^-1 s^-1; unknown units never pass through."""
    require(isinstance(unit, str), "missing_unit")
    unit = unit.replace("\u03bc", "\u00b5")
    time = {"ms^(-1)": 1000.0, "s^(-1)": 1.0, "min^(-1)": 1 / 60, "h^(-1)": 1 / 3600}
    concentration = {"M": 1.0, "mM": 1e-3, "\u00b5M": 1e-6, "nM": 1e-9, "pM": 1e-12}
    if parameter == "kcat" and unit in time:
        return time[unit]
    if parameter == "Km" and unit in concentration:
        return concentration[unit]
    if parameter == "kcat/Km":
        for c, factor in concentration.items():
            for t, t_factor in time.items():
                if unit == f"{c}^(-1)*{t}":
                    return t_factor / factor
    raise ValueError("unsupported_unit")


def parameter_value(parameter):
    name = parameter.get("parameter_type", {}).get("name")
    require(name in CANONICAL_UNITS, "unsupported_parameter")
    require(parameter.get("role") == "Constant", "nonconstant_parameter")
    value = parameter.get("start_value")
    require(finite(value) and value > 0, "nonpositive_or_nonfinite_value")
    require(parameter.get("end_value", value) == value, "interval_parameter")
    unit = parameter.get("unit", {})
    factor = unit_factor(name, unit.get("name"))
    result = value * factor
    require(finite(result) and result > 0, "invalid_converted_value")
    normalized = parameter.get("n_start_value")
    require(unit.get("n_name") == CANONICAL_UNITS[name], "normalized_unit_mismatch")
    require(
        finite(normalized) and math.isclose(result, normalized, rel_tol=1e-8, abs_tol=0.0), "normalized_value_mismatch"
    )
    sd = parameter.get("standard_deviation")
    if sd is not None:
        require(finite(sd) and sd >= 0, "invalid_standard_deviation")
        normalized_sd = parameter.get("n_standard_deviation")
        require(
            finite(normalized_sd) and math.isclose(sd * factor, normalized_sd, rel_tol=1e-8, abs_tol=0.0),
            "normalized_sd_mismatch",
        )
    return {
        "property": name,
        "value": result,
        "unit": CANONICAL_UNITS[name],
        "log10_value": math.log10(result),
        "standard_deviation": None if sd is None else sd * factor,
        "raw_value": value,
        "raw_unit": unit["name"],
    }


def entry_identity(entry):
    general = entry.get("general", {})
    organism = general.get("organism", {})
    name, taxon = organism.get("name"), organism.get("ncbi_taxonomy_id")
    require(name in ORGANISMS and type(taxon) is int and taxon in ORGANISMS[name], "outside_plant_panel")
    enzyme = entry.get("enzyme_description", {})
    require(enzyme.get("ec_number") in EC_NUMBERS, "outside_enzyme_panel")
    require(enzyme.get("wildtype") == "wildtype", "not_wildtype")
    require(general.get("experiment_type") == "in_vitro", "not_in_vitro")
    raw_proteins = {p.get("uniprot_id", "") for p in enzyme.get("proteins", [])}
    proteins = {p for p in raw_proteins if isinstance(p, str) and ACCESSION.fullmatch(p)}
    require(len(proteins) == 1, "ambiguous_or_missing_protein")
    protein = next(iter(proteins))
    # Legacy oligomer expressions may repeat the same accession. Other identities
    # or expressions are retained in the raw source and excluded here.
    for raw in raw_proteins - proteins:
        require(
            isinstance(raw, str) and re.fullmatch(r"\(" + re.escape(protein) + r"\)\*(?:n|[1-9][0-9]*)", raw),
            "unresolved_protein_expression",
        )
    pub = entry.get("publication", {})
    pubmed = str(pub.get("pubmed_id", ""))
    require(re.fullmatch(r"[1-9][0-9]*", pubmed) is not None, "missing_publication")
    require(general.get("publication_id") == pub.get("publication_id"), "publication_mismatch")
    require(type(entry.get("id")) is int and entry["id"] > 0, "invalid_entry_id")
    return {
        "protein": protein,
        "organism": name,
        "taxonomy_id": taxon,
        "ec_number": enzyme["ec_number"],
        "publication_group": "pubmed:" + pubmed,
        "publication": pub,
        "strain": general.get("strain", {}),
        "enzyme_description": enzyme,
    }


def associated_substrate(entry, parameter, kind):
    key = parameter.get("species", {}).get("species_key")
    if not key:
        require(kind == "kcat", "missing_associated_substrate")
        return None
    parts = key.split(" | ")
    require(len(parts) == 3 and parts[2] == "Substrate", "non_substrate_association")
    matches = [
        s.get("compound", {})
        for s in entry.get("reaction", {}).get("species", [])
        if s.get("role") == "Substrate" and s.get("compound", {}).get("name") == parts[1]
    ]
    unique = {canonical(c): c for c in matches}
    require(len(unique) == 1, "ambiguous_substrate")
    compound = next(iter(unique.values()))
    require(type(compound.get("id")) is int and compound["id"] > 0, "missing_substrate_id")
    return compound


def extract_entry(entry, source):
    excluded = []
    try:
        identity = entry_identity(entry)
    except ValueError as error:
        return [], [{"entry_id": entry.get("id"), "parameter_index": None, "reason": str(error)}]
    result = []
    for index, parameter in enumerate(entry.get("kineticlaw", {}).get("parameter", [])):
        try:
            values = parameter_value(parameter)
            substrate = associated_substrate(entry, parameter, values["property"])
            observation = {
                "id": f"sabio_{entry['id']}_p{index}",
                **identity,
                **values,
                "associated_substrate": substrate,
                "reaction": entry.get("reaction", {}),
                "conditions": entry.get("experimental_conditions", {}),
                "parameter_comment": parameter.get("comment", ""),
                "sources": [{**source, "entry_id": entry["id"], "parameter_index": index}],
            }
            result.append(observation)
        except ValueError as error:
            excluded.append({"entry_id": entry["id"], "parameter_index": index, "reason": str(error)})
    return result, excluded


def read_entries(manifest_path, expected_sha256):
    manifest_path = Path(manifest_path)
    require(not manifest_path.is_symlink() and manifest_path.is_file(), "missing_source_manifest")
    require(sha256(manifest_path) == expected_sha256, "source_manifest_changed")
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("schema") == "sabio_source_pages.v1" and manifest.get("complete") is True,
        "incomplete_source_manifest",
    )
    require(manifest.get("endpoint") == ENDPOINT, "unexpected_source_endpoint")
    require(
        set(manifest.get("organisms", [])) == set(ORGANISMS)
        and set(manifest.get("ec_numbers", [])) == EC_NUMBERS
        and manifest.get("enzyme_type") == "wildtype",
        "source_panel_mismatch",
    )
    require(
        type(manifest.get("total_entries")) is int and 0 < manifest["total_entries"] <= 2000, "invalid_source_count"
    )
    seen, entries, offset = set(), [], 0
    require(isinstance(manifest.get("pages"), list) and 1 <= len(manifest["pages"]) <= 20, "invalid_source_pages")
    for page in manifest["pages"]:
        path = Path(page["path"])
        require(path.is_absolute() and not any(p.is_symlink() for p in [path, *path.parents]), "invalid_source_path")
        require(path.is_relative_to(manifest_path.resolve().parent / "source"), "source_outside_snapshot")
        require(path.is_file() and sha256(path) == page["sha256"], "source_page_changed")
        obj = json.loads(path.read_text())
        response = obj["response"]
        require(
            response["numFound"] == manifest["total_entries"] and response["start"] == offset == page["offset"],
            "source_page_boundary_mismatch",
        )
        require(page["parameters"]["q"] == manifest["query"], "source_query_mismatch")
        require(
            len(response["docs"]) == page["entries"] == min(100, manifest["total_entries"] - offset),
            "source_page_count_mismatch",
        )
        for doc in response["docs"]:
            entry_id = doc["EntryID"]
            require(type(entry_id) is int and entry_id not in seen, "duplicate_source_entry")
            seen.add(entry_id)
            encoded = doc["Json"]
            if isinstance(encoded, list):
                require(len(encoded) == 1, "ambiguous_source_record")
                encoded = encoded[0]
            entry = json.loads(encoded) if isinstance(encoded, str) else encoded
            require(isinstance(entry, dict) and entry.get("id") == entry_id, "source_identity_mismatch")
            entries.append((entry, {"path": str(path), "sha256": page["sha256"]}))
        offset += len(response["docs"])
    require(len(seen) == manifest["total_entries"], "missing_source_page")
    return manifest, entries


def build_observations(entries):
    observations, excluded, seen = [], [], {}
    duplicates = 0
    for entry, source in entries:
        rows, skipped = extract_entry(entry, source)
        excluded.extend(skipped)
        for row in rows:
            # Repeated entries with the same reported measurement are one datum;
            # every original source coordinate remains available.
            content = {k: v for k, v in row.items() if k not in {"id", "sources", "raw_value", "raw_unit"}}
            key = hashlib.sha256(canonical(content).encode()).hexdigest()
            if key in seen:
                seen[key]["sources"].extend(row["sources"])
                duplicates += 1
            else:
                row["measurement_key"] = key
                seen[key] = row
                observations.append(row)
    return observations, excluded, duplicates


def export_source(manifest_path, expected_sha256, output):
    manifest, entries = read_entries(manifest_path, expected_sha256)
    observations, excluded, duplicates = build_observations(entries)
    cohorts = defaultdict(list)
    for row in observations:
        cohorts[(row["organism"], row["protein"], row["property"])].append(row)
    inventory = []
    for (organism, protein, property_name), rows in sorted(cohorts.items()):
        inventory.append(
            {
                "organism": organism,
                "protein": protein,
                "property": property_name,
                "observations": len(rows),
                "publication_groups": len({r["publication_group"] for r in rows}),
                "with_ph": sum(bool(r["conditions"].get("envvar_ph")) for r in rows),
                "with_temperature": sum(bool(r["conditions"].get("envvar_temperature")) for r in rows),
                "ready_for_acquisition": False,
            }
        )
    summary = {
        "source_entries": len(entries),
        "retained_observations": len(observations),
        "exact_duplicate_observations_collapsed": duplicates,
        "properties": dict(Counter(r["property"] for r in observations)),
        "unique_proteins": len({r["protein"] for r in observations}),
        "publication_groups": len({r["publication_group"] for r in observations}),
        "protein_property_cohorts": len(cohorts),
        "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
        "new_models": 0,
        "external_model_calls": 0,
    }
    report = {
        "schema": SCHEMA,
        "source_manifest": {"path": str(Path(manifest_path).resolve()), "sha256": expected_sha256},
        "source_query": manifest["query"],
        "summary": summary,
        "policy": {
            "unit_conversion": "raw units independently converted and checked against normalized source values",
            "label_aggregation": "none; exact repeated measurements collapsed with all source coordinates",
            "conditions": "retained as reported; no imputation or aggregate-label/condition join",
            "independence": "publication identity retained; partitions and substrate/protein identity grouping not yet frozen",
            "task_admission": False,
            "next_required": [
                "Choose a scientific task and sufficient independent groups",
                "Freeze predictors and treatment of missing conditions",
                "Bind sequence or substrate features when required",
                "Prepare training/validation/final partitions and domain controls",
            ],
        },
        "inventory": inventory,
    }
    output = Path(output)
    require(not output.exists(), "output_already_exists")
    output.mkdir(parents=True)
    for filename, value in {
        "observations.json": observations,
        "excluded.json": excluded,
        "manifest.json": report,
    }.items():
        (output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
    return report
