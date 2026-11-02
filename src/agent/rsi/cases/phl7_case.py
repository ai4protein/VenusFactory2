"""Pinned PHL7 OPES analysis replay, separate from new MD or full-paper reproduction.

Public source: doi:10.5281/zenodo.18650810 (CC BY 4.0, Kuenze and Zlobin).
Archived scripts are inspected as data. No pickle objects or author shell scripts
are executed. Only the published PLUMED reweighting inputs enter the driver.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from agent.rsi.engine.storage import sha256_file

ARCHIVE_SHA256 = "3dd9cb3d0bfcfa53d02ae528bdd940d3fc815e5570ba071f3f439daaabae9f91"
ARCHIVE_NAME = "MD_simulations_PHL7.zip"
PREFIX = "MD_simulations_PHL7/"
VARIANTS = ("WT", "R4M6", "R4M9", "R4M10")
ANALYSES = {"93_open_closed": "fes-fh.dat", "Met_rotation": "fes.dat"}
# Declared before any PLUMED replay, in printed source-file units.
TOLERANCES = {"coordinates": 1e-6, "free_energy_kj_mol": 5e-5, "derivatives": 1e-3}


def read_table(text):
    fields = None
    settings = {}
    for line in text.splitlines():
        if line.startswith("#! FIELDS "):
            if fields is not None:
                raise ValueError("duplicate PLUMED field header")
            fields = line.split()[2:]
        elif line.startswith("#! SET "):
            _, _, key, value = line.split(maxsplit=3)
            if key in settings:
                raise ValueError("duplicate PLUMED metadata")
            settings[key] = value.strip()
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("missing or duplicate PLUMED fields")
    data = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
    if not len(data) or data.shape[1] != len(fields):
        raise ValueError("PLUMED column contract mismatch")
    return fields, settings, data


def _numeric_series(text, required):
    fields, settings, data = read_table(text)
    if fields != required or not np.isfinite(data).all() or not (np.diff(data[:, 0]) > 0).all():
        raise ValueError("invalid PLUMED time series")
    return fields, settings, data


def compare_fes(reference_text, actual_text):
    rf, rs, reference = read_table(reference_text)
    af, ass, actual = read_table(actual_text)
    if rf != af or rs != ass or reference.shape != actual.shape or reference.shape[1] != 5:
        raise ValueError("free-energy grid schema changed")
    if not np.isfinite(reference[:, :2]).all() or not np.isfinite(actual[:, :2]).all():
        raise ValueError("nonfinite grid coordinates")
    if np.isnan(reference).any() or np.isnan(actual).any() or np.isneginf(reference).any() or np.isneginf(actual).any():
        raise ValueError("invalid grid values")
    coordinate_error = float(np.abs(reference[:, :2] - actual[:, :2]).max())
    if coordinate_error > TOLERANCES["coordinates"]:
        raise ValueError("grid coordinates changed")
    finite_reference = np.isfinite(reference[:, 2:])
    finite_actual = np.isfinite(actual[:, 2:])
    finite_equal = bool(np.array_equal(finite_reference, finite_actual))
    common = finite_reference & finite_actual
    errors = []
    for column in range(3):
        mask = common[:, column]
        errors.append(float(np.abs(reference[mask, column + 2] - actual[mask, column + 2]).max()) if mask.any() else None)
    if errors[0] is None:
        raise ValueError("no comparable finite free energies")
    energy_mask = common[:, 0]
    delta = actual[energy_mask, 2] - reference[energy_mask, 2]
    offset = float(np.mean(delta))
    return {
        "grid_rows": len(reference),
        "coordinate_max_abs_error": coordinate_error,
        "finite_support_identical": finite_equal,
        "reference_finite_energies": int(finite_reference[:, 0].sum()),
        "actual_finite_energies": int(finite_actual[:, 0].sum()),
        "free_energy_max_abs_error": errors[0],
        "derivative_max_abs_errors": errors[1:],
        "mean_energy_offset": offset,
        "offset_removed_max_abs_error": float(np.abs(delta - offset).max()),
        "matches_reference": finite_equal and errors[0] <= TOLERANCES["free_energy_kj_mol"]
        and all(x is not None and x <= TOLERANCES["derivatives"] for x in errors[1:]),
        "offset_removed_is_diagnostic_only": True,
    }


def allocation_ids():
    return [f"{analysis}/{variant}/repl_{rep}" for analysis in ANALYSES for variant in VARIANTS for rep in (1, 2, 3)]


def admit(archive):
    archive = Path(archive)
    if archive.is_symlink() or sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("PHL7 public archive identity changed")
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = set()
        members = []
        for info in z.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(info.external_attr >> 16) or info.filename in names:
                raise ValueError("unsafe or duplicate archive entry")
            names.add(info.filename)
            if not info.is_dir():
                members.append({"name": info.filename, "bytes": info.file_size})
        for identity in allocation_ids():
            analysis, variant, replica = identity.split("/")
            parent = PREFIX + "OPES/" + analysis + "/" + variant + "/"
            loc = parent + replica + "/"
            variables = ["D1", "D2"] if analysis == "93_open_closed" else ["T1", "T2"]
            fields = ["time", *variables, "opes.bias", "opes.rct"]
            _, _, cv = _numeric_series(z.read(loc + "CV").decode(), fields)
            _, _, trimmed = _numeric_series(z.read(loc + "CV_").decode(), fields)
            if cv.shape != (50001, 5) or trimmed.shape != (45001, 5) or not np.array_equal(trimmed, cv[5000:]):
                raise ValueError("OPES input does not preserve the published 10-100 ns frame subset")
            if (cv[0, 0], cv[-1, 0]) != (0.0, 100000.0) or not np.all(np.diff(cv[:, 0]) == 2.0):
                raise ValueError("OPES time axis changed")
            inputs = {"CV_": loc + "CV_", "reweight.dat": parent + "reweight.dat"}
            if analysis == "93_open_closed":
                _, _, dd = _numeric_series(z.read(loc + "DD").decode(), ["time", "D_FH"])
                _, _, ddt = _numeric_series(z.read(loc + "DD_").decode(), ["time", "D_FH"])
                if dd.shape != (50001, 2) or not np.array_equal(ddt, dd[5000:]) or not np.array_equal(ddt[:, 0] * 2, trimmed[:, 0]):
                    raise ValueError("OPES driver-frame to simulation-time mapping changed")
                inputs["DD_"] = loc + "DD_"
            script = z.read(parent + "plumed.dat").decode()
            mode = re.findall(r"^(OPES_METAD(?:_EXPLORE)?) ", script, flags=re.MULTILINE)
            if len(mode) != 1:
                raise ValueError("ambiguous published OPES action")
            reference = loc + ANALYSES[analysis]
            grid = compare_fes(z.read(reference).decode(), z.read(reference).decode())
            rows.append({"id": identity, "analysis": analysis, "variant": variant, "replica": replica,
                         "input_members": inputs, "reference_member": reference,
                         "input_sha256": {k: hashlib.sha256(z.read(v)).hexdigest() for k, v in inputs.items()},
                         "reference_sha256": hashlib.sha256(z.read(reference)).hexdigest(), "retained_frames": len(trimmed),
                         "time_ps": [float(trimmed[0, 0]), float(trimmed[-1, 0])], "published_action": mode[0],
                         "grid_rows": grid["grid_rows"],
                         "time_mapping": "paired_by_row; DD driver index times 2 equals CV physical ps" if analysis == "93_open_closed" else "CV time in ps"})
    return {
        "schema_version": 1, "case": "phl7.opes_analysis.v1", "archive_sha256": ARCHIVE_SHA256,
        "source_doi": "10.5281/zenodo.18650810", "source_license": "CC-BY-4.0",
        "paper_doi": "10.1038/s41467-026-70868-4", "members": members, "allocations": rows,
        "comparison_tolerances": TOLERANCES,
        "method_discrepancies": [
            "Main Methods discard 100 ns of HREX; Supplementary Methods specify analysis of 200-500 ns.",
            "Supplementary Methods describe all OPES runs as explore; archived plumed.dat actions vary by system. Exact actions are recorded per allocation.",
            "Open/closed reweighting ignores timestamps and pairs CV/DD rows; DD driver timestamps are frame indices, with two ps per frame in CV.",
        ],
        "missing_for_full_computational_reproduction": [
            "Full unbiased-MD, HREX and OPES trajectories (no XTC/TRR/DCD/TPR in the released archive).",
            "All equilibrated systems and their run topologies; HREX preparation releases the WT example, not all variants.",
            "Complete PROSS/ENDURE design inputs, environment and intermediate decisions.",
            "Gromacs 2023 with Plumed 2.9 integration and the specified QM/MM setup for force-field calibration.",
            "Resolution or separately reported replay of main-text/supplementary/script protocol discrepancies.",
        ],
        "new_molecular_dynamics_performed": False, "full_paper_reproduced": False,
        "restricted_evidence_evaluation": False,
    }


def canonical_phl7(report):
    if report.get("execution_origin") != "new_analysis_of_author_outputs" or report.get("archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("PHL7 provenance mismatch")
    if report.get("new_molecular_dynamics_performed") is not False or report.get("full_paper_reproduced") is not False:
        raise ValueError("unsupported PHl7 reproduction claim")
    rows = report.get("allocations", [])
    if len(rows) != 24 or sorted(x["id"] for x in rows) != sorted(allocation_ids()):
        raise ValueError("incomplete or duplicate PHL7 analysis allocation")
    if any(x.get("status") != "completed" or x.get("comparison") is None for x in rows):
        raise ValueError("incomplete PHL7 computations")
    for row in rows:
        c = row["comparison"]
        errors = [c.get("free_energy_max_abs_error"), *c.get("derivative_max_abs_errors", [])]
        if len(errors) != 3 or any(type(x) not in (float, int) or not np.isfinite(x) or x < 0 for x in errors):
            raise ValueError("invalid PHL7 numerical error")
        expected = (c.get("finite_support_identical") is True
                    and errors[0] <= TOLERANCES["free_energy_kj_mol"]
                    and all(x <= TOLERANCES["derivatives"] for x in errors[1:]))
        if c.get("matches_reference") is not expected:
            raise ValueError("PHL7 agreement inconsistent with numerical evidence")
    return {"computational_scope": "phl7_published_opes_outputs_to_free_energy_grids",
            "execution_origin": report["execution_origin"], "completed_analyses": 24,
            "reference_agreement_count": sum(x["comparison"]["matches_reference"] for x in rows),
            "analysis_discrepancies": [x["id"] for x in rows if not x["comparison"]["matches_reference"]],
            "new_molecular_dynamics_performed": False, "full_paper_reproduced": False,
            "new_algorithm_or_agent_superiority_established": False,
            "published_method_discrepancies": report["method_discrepancies"]}


def verify_artifacts(report, output):
    output = Path(output).resolve()
    if sha256_file(output / "protocol.json") != report.get("protocol_sha256"):
        raise ValueError("PHL7 protocol changed")
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol.get("archive_sha256") != ARCHIVE_SHA256 or protocol.get("comparison_tolerances") != TOLERANCES:
        raise ValueError("PHL7 admission changed")
    expected = {x["id"]: x for x in protocol["allocations"]}
    if set(expected) != set(allocation_ids()) or len(protocol["allocations"]) != 24:
        raise ValueError("PHL7 protocol allocation changed")
    canonical_phl7(report)
    if report.get("method_discrepancies") != protocol.get("method_discrepancies"):
        raise ValueError("PHL7 discrepancy projection changed")
    for row in report["allocations"]:
        pinned = expected[row["id"]]
        bindings = {**pinned["input_sha256"], "reference.dat": pinned["reference_sha256"]}
        if any(row["artifact_hashes"].get(k) != v for k, v in bindings.items()):
            raise ValueError("PHL7 input no longer matches the admitted archive")
        folder = output / "allocations" / row["id"]
        for name, sha in row["artifact_hashes"].items():
            path = (folder / name).resolve()
            path.relative_to(folder.resolve())
            if sha256_file(path) != sha:
                raise ValueError("PHL7 artifact changed")
        required = {*expected[row["id"]]["input_members"], "reference.dat", ANALYSES[expected[row["id"]]["analysis"]], "driver.log"}
        if set(row["artifact_hashes"]) != required:
            raise ValueError("PHL7 input/output evidence missing")
        comparison = compare_fes((folder / "reference.dat").read_text(), (folder / ANALYSES[expected[row["id"]]["analysis"]]).read_text())
        if comparison != row["comparison"]:
            raise ValueError("PHL7 comparison changed")
    return canonical_phl7(report)
