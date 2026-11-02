"""Cross-library registered scope inventory; metadata is not renewed qualification."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from agent.harness.contracts import digest

from agent.rsi.datasets.research_project_catalog import scope_key, text
from .scientist_model_library import ScientistModelLibrary


def canonical_scope(key):
    scope = key["scope"]
    candidate = {
        "protein_scope": {"kind": "protein", "identifier": scope["protein"]},
        "property": scope["target"],
        "scenario": scope.get("context", {"assay": scope["assay"]}),
    }
    return scope_key(candidate, {"scope": scope, "plugin_id": key["plugin_id"]})


def summarize_catalogs(catalogs, *, target_scopes=10000):
    if type(target_scopes) is not int or target_scopes <= 0:
        raise ValueError("Positive scope target required")
    names, scopes, programs = set(), {}, set()
    records = []
    for item in catalogs:
        name, catalog = item["library"], item["catalog"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Library identities must be nonempty and distinct")
        names.add(name)
        seen = set()
        for row in catalog["entries"]:
            entry = row["entry_id"]
            if entry in seen:
                raise ValueError("Repeated entry within a library")
            seen.add(entry)
            key = row["key"]
            scope_id = canonical_scope(key)
            programs.add(key["program_sha256"])
            scope = key["scope"]
            record = scopes.setdefault(
                scope_id,
                {
                    "scope_id": scope_id,
                    "protein": scope["protein"],
                    "property": scope["target"],
                    "unit": scope["unit"],
                    "scenario": scope.get("context", {"assay": scope["assay"]}),
                    "registrations": [],
                },
            )
            record["registrations"].append(
                {
                    "library": name,
                    "entry_id": entry,
                    "program_sha256": key["program_sha256"],
                    "training_sha256": key["training_sha256"],
                    "plugin_id": key["plugin_id"],
                }
            )
        if len(seen) != catalog["indexed_entries"]:
            raise ValueError("Catalog entry count disagrees with entries")
        records.append({"library": name, "catalog_sha256": digest(catalog), "indexed_entries": len(seen)})
    ordered = [scopes[k] for k in sorted(scopes)]
    registrations = sum(len(x["registrations"]) for x in ordered)
    return {
        "schema": "model_scope_inventory.v1",
        "status": "registered_metadata_inventory",
        "libraries": records,
        "registered_entries": registrations,
        "registered_scopes": len(scopes),
        "repeated_scope_registrations": registrations - len(scopes),
        "scopes_in_multiple_libraries": sum(len({r["library"] for r in s["registrations"]}) > 1 for s in ordered),
        "distinct_source_programs": len(programs),
        "property_counts": dict(sorted(Counter(text(s["property"]) for s in ordered).items())),
        "scopes": ordered,
        "target_scopes": target_scopes,
        "minimum_additional_scopes_if_all_registered_entries_remain_qualified": max(0, target_scopes - len(scopes)),
        "currently_reverified_scopes": None,
        "completion_claim": False,
        "counting_rule": "Normalized declared protein, property, unit and observed scenario; source versions, training partitions, interfaces and repeated registrations do not add coverage.",
        "limitations": [
            "Declared biological aliases and equivalent measurement units are not automatically reconciled.",
            "Registration objects are inspected; native scientific evidence and numerical predictions are not replayed by this inventory.",
        ],
        "new_autonomous_models": 0,
        "external_model_calls": 0,
    }


def inventory_libraries(roots, *, target_scopes=10000):
    if not isinstance(roots, list) or not roots:
        raise ValueError("Declare the library roots explicitly")
    libraries = []
    for root in roots:
        path = Path(root).absolute()
        if not path.is_dir():
            raise ValueError("Declared model library does not exist")
        library = ScientistModelLibrary(path)
        libraries.append((library, library.catalog(verify=False)))
    report = summarize_catalogs(
        [{"library": str(lib.root), "catalog": catalog} for lib, catalog in libraries], target_scopes=target_scopes
    )
    for library, before in libraries:
        if library.catalog(verify=False) != before:
            raise ValueError("Model catalog changed during inventory; retry with a new snapshot")
    report["catalog_metadata_stable_during_inventory"] = True
    return report
