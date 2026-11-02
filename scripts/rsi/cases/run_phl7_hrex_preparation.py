"""Reconstruct the published PHL7 WT eight-replica HREX preparation.

PLUMED scales the archived standalone topology. Independent numerical checks
verify the scaling and molecular counts. This is input preparation, not MD or
an energy replay, and does not reconstruct the missing mutant systems.
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
import math
import os
import re
import subprocess
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ARCHIVE_SHA256 = "3dd9cb3d0bfcfa53d02ae528bdd940d3fc815e5570ba071f3f439daaabae9f91"
PREFIX = "MD_simulations_PHL7/HREX_preparation/"
# PLUMED 2.9.0 mixes six-significant-digit print and twelve-digit conversion.
RTOL = 6e-6
ATOL = 1e-10


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_topology(text):
    blocks = defaultdict(list)
    molecule = "GLOBAL"
    section = None
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            raise ValueError("Standalone topology contains preprocessor directive")
        if line.startswith("["):
            match = re.fullmatch(r"\[\s*([A-Za-z0-9_]+)\s*\]", line)
            if not match:
                raise ValueError("Malformed topology section")
            section = match.group(1)
            if section == "moleculetype":
                molecule = "PENDING"
            if section in {"system", "molecules"}:
                molecule = "GLOBAL"
            continue
        if section is None:
            if line.startswith("*"):
                blocks[("GLOBAL", "preamble")].append(line.split())
                continue
            raise ValueError("Data outside topology section")
        fields = line.split()
        if section == "moleculetype":
            molecule = fields[0]
        blocks[(molecule, section)].append(fields)
    return dict(blocks)


def molecular_atoms(blocks):
    entries = blocks[("GLOBAL", "molecules")]
    return sum(len(blocks[(row[0], "atoms")]) * int(row[1]) for row in entries)


def verify_scaling(reference, actual, scale):
    if reference.keys() != actual.keys():
        raise ValueError("Topology blocks changed")
    combination = int(reference[("GLOBAL", "defaults")][0][1])
    if combination not in {1, 2}:
        raise ValueError("Unvalidated combination rule")
    hot = {
        key[0]: {row[0] for row in rows if row[1].endswith("_")} for key, rows in reference.items() if key[1] == "atoms"
    }
    checked = 0
    max_relative = 0.0
    counts = defaultdict(int)
    for key, rows in reference.items():
        other = actual[key]
        section = key[1]
        if len(rows) != len(other):
            raise ValueError("Topology row count changed")
        for row, observed in zip(rows, other, strict=True):
            if len(row) != len(observed):
                raise ValueError("Topology field count changed")
            factors = {}
            if section == "atoms" and row[1].endswith("_"):
                factors[6] = math.sqrt(scale)
            elif section == "atomtypes" and row[0].endswith("_"):
                factors[len(row) - 1] = scale
                if combination == 1:
                    factors[len(row) - 2] = scale
            elif section in {"pairtypes", "nonbond_params"}:
                factor = scale ** (sum(x.endswith("_") for x in row[:2]) / 2)
                factors[4] = factor
                if combination == 1:
                    factors[3] = factor
            elif section == "pairs" and len(row) > 3:
                q = [math.sqrt(scale) if x in hot[key[0]] else 1.0 for x in row[:2]]
                if row[2] == "1":
                    factors[4] = q[0] * q[1]
                    if combination == 1:
                        factors[3] = q[0] * q[1]
                elif row[2] == "2":
                    factors.update({4: q[0], 5: q[1], 7: q[0] * q[1]})
                    if combination == 1:
                        factors[6] = q[0] * q[1]
                else:
                    raise ValueError("Unvalidated pair function")
            elif section == "dihedrals":
                kind = int(row[4])
                factor = scale ** (sum(x in hot[key[0]] for x in [row[0], row[3]]) / 2)
                if kind in {1, 2, 4, 9}:
                    factors[6] = factor
                elif kind in {3, 5}:
                    factors.update(dict.fromkeys(range(5, len(row)), factor))
                else:
                    raise ValueError("Unvalidated dihedral function")
            for i, (value, got) in enumerate(zip(row, observed, strict=True)):
                if i not in factors:
                    if value != got:
                        raise ValueError(f"Unscaled field changed: {key} column {i}")
                else:
                    expected = float(value) * factors[i]
                    measured = float(got)
                    if not math.isfinite(measured) or not math.isclose(expected, measured, rel_tol=RTOL, abs_tol=ATOL):
                        raise ValueError(f"Scaled parameter differs: {key} column {i}: {measured} vs {expected}")
                    if expected:
                        max_relative = max(max_relative, abs(measured - expected) / abs(expected))
                    checked += 1
                    counts[section] += 1
    return {
        "numeric_parameters_checked": checked,
        "parameter_counts_by_section": dict(counts),
        "max_relative_scaling_error": max_relative,
        "unchanged_fields_identical": True,
        "molecular_atoms": molecular_atoms(actual),
    }


def main(args):
    if sha(args.archive) != ARCHIVE_SHA256:
        raise ValueError("Archive identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    args._owns_output = True
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    report = {
        "status": "running",
        "new_molecular_dynamics_performed": False,
        "energy_replay_performed": False,
        "full_paper_reproduced": False,
        "replicas": [],
    }
    save(args.output / "report.json", report)
    inputs = {}
    with zipfile.ZipFile(args.archive) as archive:
        for member, local in [
            ("hrex.bash", "author_hrex.bash"),
            ("hrex.mdp", "hrex.mdp"),
            ("WT/hrex.top", "hrex.top"),
            ("WT/equilibrated.gro", "equilibrated.gro"),
        ]:
            data = archive.read(PREFIX + member)
            path = args.output / local
            path.write_bytes(data)
            inputs[member] = sha(path)
    script = (args.output / "author_hrex.bash").read_text()
    if not all(s in script for s in ["nrep=8", "tmin=300", "tmax=600", "partial_tempering"]):
        raise ValueError("Published schedule changed")
    prefix = args.plumed.resolve().parents[1]
    env = {**os.environ, "PATH": str(prefix / "bin") + ":/usr/bin:/bin"}
    awk = str(prefix / "bin/gawk")
    # Preserve the exact public awk output precision used by the author's shell recipe.
    temperatures = subprocess.check_output(
        [
            awk,
            "-v",
            "n=8",
            "-v",
            "tmin=300",
            "-v",
            "tmax=600",
            'BEGIN{for(i=0;i<n;i++){t=tmin*exp(i*log(tmax/tmin)/(n-1));printf(t);if(i<n-1)printf(",");}}',
        ],
        env=env,
        text=True,
    ).strip()
    scales = [
        subprocess.check_output(
            [awk, f'BEGIN{{FS=",";}}{{print $1/${i + 1};}}'], input=temperatures, env=env, text=True
        ).strip()
        for i in range(8)
    ]
    gro = (args.output / "equilibrated.gro").read_text().splitlines()
    atom_count = int(gro[1])
    source = parse_topology((args.output / "hrex.top").read_text())
    if len(gro) != atom_count + 3 or molecular_atoms(source) != atom_count:
        raise ValueError("Coordinate/topology atom count differs")
    for line in gro[2:-1]:
        if not all(math.isfinite(float(line[i : i + 8])) for i in range(20, 68, 8)):
            raise ValueError("Invalid coordinate/velocity")
    mdp = {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in (args.output / "hrex.mdp").read_text().splitlines()
        if "=" in line
    }
    protocol = {
        "source_doi": "10.5281/zenodo.18650810",
        "paper_doi": "10.1038/s41467-026-70868-4",
        "archive_sha256": ARCHIVE_SHA256,
        "inputs": inputs,
        "temperatures_kelvin": temperatures,
        "scales": scales,
        "plumed_binary_sha256": sha(args.plumed),
        "plumed_scaling_script_sha256": sha(prefix / "lib/plumed/scripts/partial_tempering.sh"),
        "gawk_sha256": sha(Path(awk)),
        "relative_tolerance": RTOL,
        "absolute_tolerance": ATOL,
        "coordinate_atoms": atom_count,
        "requested_production_ns": int(mdp["nsteps"]) * float(mdp["dt"]) / 1000,
        "implementation_sha256": sha(Path(__file__)),
        "scope": "Published WT HREX topology preparation only; physical bath remains 300 K. No dynamics, energies, exchange acceptance or missing variant reconstruction.",
    }
    save(args.output / "protocol.json", protocol)
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    reference = None
    for i, scale in enumerate(scales):
        output = args.output / f"topol_{i}.top"
        err = args.output / f"topol_{i}.stderr.log"
        start = time.monotonic()
        with (args.output / "hrex.top").open("rb") as inp, output.open("wb") as dst, err.open("wb") as errors:
            process = subprocess.run(
                [str(args.plumed), "partial_tempering", scale],
                stdin=inp,
                stdout=dst,
                stderr=errors,
                env=env,
                cwd=args.output,
                timeout=120,
            )
        if process.returncode or "ERROR:" in err.read_text():
            raise ValueError("PLUMED topology scaling failed")
        actual = parse_topology(output.read_text())
        if reference is None:
            reference = actual
            for key in [k for k in source if k[1] in {"atoms", "bonds", "angles", "constraints", "molecules"}]:
                a, b = source[key], reference[key]
                if len(a) != len(b):
                    raise ValueError("Unscaled source connectivity differs")
                for x, y in zip(a, b, strict=True):
                    if len(x) != len(y):
                        raise ValueError("Unscaled source field count differs")
                    for j, (v, w) in enumerate(zip(x, y, strict=True)):
                        if key[1] == "atoms" and j == 6:
                            if not math.isclose(float(v), float(w), rel_tol=RTOL, abs_tol=ATOL):
                                raise ValueError("Unscaled source charge differs")
                        elif v != w:
                            raise ValueError("Unscaled source connectivity or identity differs")
        checks = verify_scaling(reference, actual, float(scale))
        if checks["molecular_atoms"] != atom_count:
            raise ValueError("Replica atom count differs")
        report["replicas"].append(
            {
                "index": i,
                "scale": float(scale),
                "returncode": process.returncode,
                "wall_seconds": time.monotonic() - start,
                "topology_sha256": sha(output),
                "checks": checks,
            }
        )
        save(args.output / "report.json", report)
    report.update(
        status="completed",
        replicas_prepared=8,
        coordinate_atoms=atom_count,
        protocol_sha256=sha(args.output / "protocol.json"),
        wall_seconds=time.monotonic() - started,
        missing_next=[
            "GROMACS 2023.1 with PLUMED 2.9 integration and MPI",
            "Simulation index groups and validated grompp inputs",
            "lambda=1 energy equivalence and scaling energy checks before production",
            "Eight 500 ns replica trajectories and exchange analysis",
            "Equivalent initial systems for all required mutant variants",
        ],
    )
    save(args.output / "report.json", report)
    print(
        json.dumps(
            {"status": report["status"], "replicas": 8, "atoms": atom_count, "new_molecular_dynamics_performed": False}
        ),
        flush=True,
    )


def execute(args):
    args._owns_output = False
    try:
        main(args)
    except Exception as exc:
        report_path = args.output / "report.json"
        if args._owns_output and report_path.exists():
            failure = json.loads(report_path.read_text())
            failure.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            save(report_path, failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--plumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
