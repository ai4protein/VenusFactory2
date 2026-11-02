"""Versioned residue reference data and non-executing literal consistency checks."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math

REFERENCE_DATA = {
    "canonical_order": "ACDEFGHIKLMNPQRSTVWY",
    "plugin_id": "protein_residue_references.v1",
    "references": {
        "free_amino_acid.average_mass.v1": {
            "interpretation": "Free amino acids, not "
            "polymer residue masses or "
            "the molecular mass of an "
            "assembled peptide. A "
            "sequence mean is a linear "
            "combination of amino-acid "
            "frequencies.",
            "name": "Average free amino-acid molecular mass",
            "source": {
                "official_release_bytes_match": True,
                "package": "biopython",
                "sha256": "897fb42f1bd6e3f837be6918ae6c54e77e61013b284cbabd4e7dfc01723e1a91",
                "url": "https://raw.githubusercontent.com/biopython/biopython/biopython-187/Bio/Data/IUPACData.py",
                "version": "1.87",
            },
            "source_symbol": "protein_weights",
            "unit": "Da",
            "values": {
                "A": 89.0932,
                "C": 121.1582,
                "D": 133.1027,
                "E": 147.1293,
                "F": 165.1891,
                "G": 75.0666,
                "H": 155.1546,
                "I": 131.1729,
                "K": 146.1876,
                "L": 131.1729,
                "M": 149.2113,
                "N": 132.1179,
                "P": 115.1305,
                "Q": 146.1445,
                "R": 174.201,
                "S": 105.0926,
                "T": 119.1192,
                "V": 117.1463,
                "W": 204.2252,
                "Y": 181.1885,
            },
        },
        "kyte_doolittle.hydropathy.v1": {
            "interpretation": "A per-residue empirical "
            "hydropathy scale; its sequence "
            "mean is a linear combination "
            "of amino-acid frequencies.",
            "name": "Kyte-Doolittle residue hydropathy",
            "source": {
                "official_release_bytes_match": True,
                "package": "biopython",
                "sha256": "488a4c336556c5e6aa41e9208cd4a9390d7a29d3d83e2c8be1f0cb518c718947",
                "url": "https://raw.githubusercontent.com/biopython/biopython/biopython-187/Bio/SeqUtils/ProtParamData.py",
                "version": "1.87",
            },
            "source_symbol": "kd",
            "unit": "dimensionless hydropathy scale",
            "values": {
                "A": 1.8,
                "C": 2.5,
                "D": -3.5,
                "E": -3.5,
                "F": 2.8,
                "G": -0.4,
                "H": -3.2,
                "I": 4.5,
                "K": -3.9,
                "L": 3.8,
                "M": 1.9,
                "N": -3.5,
                "P": -1.6,
                "Q": -3.5,
                "R": -4.5,
                "S": -0.8,
                "T": -0.7,
                "V": 4.2,
                "W": -0.9,
                "Y": -1.3,
            },
        },
    },
    "schema": 1,
    "scope": "Reference literal parity only. This does not validate feature computation, source "
    "usage, biological mechanism, independent information, predictive benefit or model "
    "novelty.",
    "source_release": "biopython-187",
}
DECLARATION = "PROTEIN_FEATURE_REFERENCES"
PROPERTY_NAMES = {"HYDRO", "HYDROPATHY", "HYDROPHOBICITY", "MW", "MASS", "AA_MASS", "MOLECULAR_WEIGHT"}
LIMITATION = (
    "Only declared literal values, amino-acid order and units are checked. "
    "Feature computation, use of these constants, biological mechanism, predictive benefit, "
    "independent information and model novelty are not established."
)


def reference_catalog():
    payload = copy.deepcopy(REFERENCE_DATA)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return {
        **payload,
        "catalog_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "declaration_name": DECLARATION,
        "declaration_example": {
            "MASS": {
                "reference_id": "free_amino_acid.average_mass.v1",
                "residue_order": payload["canonical_order"],
                "unit": "Da",
            }
        },
        "declaration_rule": (
            "For a named reference property, declare its module-level literal array or mapping "
            "in PROTEIN_FEATURE_REFERENCES. Copy the table values with the stated residue order and unit. "
            "A missing declaration does not certify a property; a contradictory declaration is rejected."
        ),
        "limitations": LIMITATION,
    }


def _top_assignments(tree):
    values = {}
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for target in targets:
            if isinstance(target, ast.Name):
                values.setdefault(target.id, []).append((node.value, len(targets) == 1))
    return values


def _literal(node):
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Attribute)
            or node.func.attr not in {"array", "asarray"}
            or not isinstance(node.func.value, ast.Name)
            or len(node.args) != 1
            or any(k.arg != "dtype" for k in node.keywords)
        ):
            raise ValueError("Reference arrays require a literal list, mapping or array/asarray literal wrapper")
        node = node.args[0]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise ValueError("Reference initialization is not a supported literal") from exc


def audit_source_references(source):
    if not isinstance(source, str) or not 1 <= len(source.encode()) <= 60000:
        raise ValueError("Model source must contain 1..60000 bytes")
    tree = ast.parse(source)
    assignments = _top_assignments(tree)
    catalog = reference_catalog()
    report = {
        "schema": 1,
        "plugin_id": catalog["plugin_id"],
        "catalog_sha256": catalog["catalog_sha256"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "status": "not_declared",
        "declared_literals_verified": False,
        "references": [],
        "declaration_errors": [],
        "unclaimed_property_named_assignments": sorted(set(assignments) & PROPERTY_NAMES),
        "biological_interpretation_verified": False,
        "limitations": LIMITATION,
    }
    if DECLARATION not in assignments:
        return report
    try:
        entries = assignments[DECLARATION]
        if len(entries) != 1 or not entries[0][1] or not isinstance(entries[0][0], ast.Dict):
            raise ValueError("Reference declarations must be one module-level literal mapping")
        keys = [ast.literal_eval(k) for k in entries[0][0].keys]
        if any(type(k) is not str for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("Reference declaration keys must be unique symbol names")
        declarations = _literal(entries[0][0])
        if not isinstance(declarations, dict) or len(declarations) > 32:
            raise ValueError("At most 32 explicit reference symbols can be declared")
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        report["status"] = "invalid_declaration"
        report["declaration_errors"] = [str(exc)]
        return report
    report["unclaimed_property_named_assignments"] = sorted((set(assignments) & PROPERTY_NAMES) - set(declarations))
    for symbol, spec in declarations.items():
        row = {"symbol": symbol, "status": "unverifiable"}
        try:
            if (
                not symbol.isidentifier()
                or not isinstance(spec, dict)
                or set(spec) != {"reference_id", "residue_order", "unit"}
            ):
                raise ValueError("Each reference declaration needs reference_id, residue_order and unit")
            reference_id = spec["reference_id"]
            if reference_id not in catalog["references"]:
                raise ValueError("Unknown versioned residue reference")
            reference = catalog["references"][reference_id]
            order = spec["residue_order"]
            if type(order) is not str or len(order) != 20 or set(order) != set(catalog["canonical_order"]):
                raise ValueError("Declared residue order must contain each canonical amino acid once")
            if spec["unit"] != reference["unit"]:
                raise ValueError("Declared unit differs from the reference definition")
            entries = assignments.get(symbol, [])
            if len(entries) != 1 or not entries[0][1]:
                raise ValueError("Reference symbol must have one unambiguous module-level initialization")
            values = _literal(entries[0][0])
            if isinstance(values, dict):
                if set(values) != set(order):
                    raise ValueError("Reference mapping must cover all twenty residues")
                ordered = [values[a] for a in order]
            elif isinstance(values, (list, tuple)) and len(values) == 20:
                ordered = list(values)
            else:
                raise ValueError("Reference vector must contain exactly twenty numeric values")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in ordered):
                raise ValueError("Reference entries must be finite real numbers")
            mismatches = [
                {"residue": a, "observed": v, "expected": reference["values"][a]}
                for a, v in zip(order, ordered, strict=True)
                if not math.isclose(v, reference["values"][a], rel_tol=0, abs_tol=1e-6)
            ]
            row.update(
                reference_id=reference_id,
                residue_order=order,
                unit=spec["unit"],
                source=reference["source"],
                status="mismatch" if mismatches else "matched",
                mismatches=mismatches,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            row["error"] = str(exc)
        report["references"].append(row)
    if declarations:
        report["declared_literals_verified"] = all(x["status"] == "matched" for x in report["references"])
        report["status"] = (
            "matched_declared_literals" if report["declared_literals_verified"] else "invalid_declaration"
        )
    return report


def require_reference_parity(source):
    report = audit_source_references(source)
    if report["status"] == "invalid_declaration":
        failures = report["declaration_errors"] + [
            f"{x['symbol']}: {x.get('error', x['status'])}" for x in report["references"] if x["status"] != "matched"
        ]
        raise ValueError("Protein reference declarations failed: " + "; ".join(failures))
    return report
