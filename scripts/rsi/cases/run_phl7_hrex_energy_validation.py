"""Validate public PHL7 WT HREX inputs and fixed-frame energies, without dynamics."""

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
from pathlib import Path

from scripts.rsi.cases.run_phl7_hrex_preparation import parse_topology

RTOL = 2e-5
ATOL = 0.02  # kJ/mol, fixed before energy evaluation; published topology has rounded coefficients.
INPUT_SHA = {
    "hrex.top": "a2b2b391d92afc829c4cf4434a94d94672ddc1b445a092242399d3ad0f482021",
    "equilibrated.gro": "98406919cb011028908bf7945307b1a1bbe126583c6d2ab3f50980597ca7945b",
    "hrex.mdp": "1f4cd4edc66dcd75e5d3ef23ce7c71f8b4037eb97fb15df691db0276a1df1099",
}
SCALED_TERMS = {
    "Proper Dih.",
    "Per. Imp. Dih.",
    "LJ-14",
    "Coulomb-14",
    "LJ (SR)",
    "Disper. corr.",
    "Coulomb (SR)",
    "Coul. recip.",
}
UNCHANGED_TERMS = {"Bond", "Harmonic Pot.", "Angle"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def mark_atoms(text, mode):
    """Change only atom-type suffixes in [atoms], preserving all parameters."""
    if mode not in {"unmark", "all_hot"}:
        raise ValueError("Unknown marking mode")
    section = None
    output = []
    changed = 0
    for line in text.splitlines():
        clean = line.split(";", 1)[0].strip()
        match = re.fullmatch(r"\[\s*(\w+)\s*\]", clean)
        if match:
            section = match.group(1)
        if section == "atoms" and clean and not clean.startswith("["):
            spans = list(re.finditer(r"\S+", line.split(";", 1)[0]))
            if len(spans) < 7:
                raise ValueError("Malformed atom row")
            old = spans[1].group()
            base = old.removesuffix("_")
            new = base + ("_" if mode == "all_hot" else "")
            line = line[: spans[1].start()] + new + line[spans[1].end() :]
            changed += old != new
        output.append(line)
    return "\n".join(output) + "\n", changed


def build_index(topology, coordinates):
    """Require the complete molecule/atom order to match, including PET and solvent."""
    parsed = parse_topology(topology)
    molecules = [v for k, v in parsed.items() if k[1] == "molecules"]
    if len(molecules) != 1:
        raise ValueError("Missing or duplicated molecule list")
    lines = coordinates.splitlines()
    if len(lines) != int(lines[1]) + 3:
        raise ValueError("Invalid GRO length")
    atoms = lines[2:-1]
    groups = {"Protein_PET": [], "Water_and_Ions": []}
    cursor = 0
    for name, count in molecules[0]:
        if name not in {"Protein_chain_A", "SOL", "NA", "CL"}:
            raise ValueError("Undeclared molecule")
        template = parsed[(name, "atoms")]
        for _ in range(int(count)):
            for row in template:
                if cursor >= len(atoms):
                    raise ValueError("Topology exceeds coordinates")
                actual = atoms[cursor]
                if (actual[5:10].strip(), actual[10:15].strip()) != (row[3], row[4]):
                    raise ValueError(f"Coordinate identity differs at atom {cursor + 1}")
                group = "Protein_PET" if name == "Protein_chain_A" else "Water_and_Ions"
                groups[group].append(cursor + 1)
                cursor += 1
    if cursor != len(atoms) or not all(groups.values()):
        raise ValueError("Index partition incomplete")
    output = []
    for name, indices in groups.items():
        output.append(f"[ {name} ]")
        output.extend(" ".join(map(str, indices[i : i + 15])) for i in range(0, len(indices), 15))
    return "\n".join(output) + "\n", {k: len(v) for k, v in groups.items()}


def restore_scaled_charge_precision(scaled, source, scale):
    """Diagnostic only: preserve original PLUMED output and replace its rounded atom charges."""
    original = parse_topology(source)
    names = iter(row[0] for key, rows in original.items() if key[1] == "moleculetype" for row in rows)
    molecule, section, atom_index = None, None, 0
    result = []
    for line in scaled.splitlines():
        clean = line.split(";", 1)[0].strip()
        match = re.fullmatch(r"\[\s*(\w+)\s*\]", clean)
        if match:
            section = match.group(1)
            if section == "moleculetype":
                molecule = next(names)
            if section == "atoms":
                atom_index = 0
        elif section == "atoms" and clean:
            row = original[(molecule, "atoms")][atom_index]
            spans = list(re.finditer(r"\S+", line.split(";", 1)[0]))
            if spans[0].group() != row[0] or spans[4].group() != row[4]:
                raise ValueError("Charge repair atom order differs")
            value = float(row[6]) * math.sqrt(scale)
            line = line[: spans[6].start()] + format(value, ".16g") + line[spans[6].end() :]
            atom_index += 1
        result.append(line)
    return "\n".join(result) + "\n"


def topology_charge(text):
    parsed = parse_topology(text)
    molecules = [v for k, v in parsed.items() if k[1] == "molecules"]
    if len(molecules) != 1:
        raise ValueError("Ambiguous molecular charge")
    return math.fsum(
        int(count) * math.fsum(float(row[6]) for row in parsed[(name, "atoms")]) for name, count in molecules[0]
    )


def explicit_pair_coulomb(topology, coordinates):
    """Analytic GROMACS pairs_nb type-1 term; require this published zero-LJ case."""
    parsed = parse_topology(topology)
    pairs = [(key, row) for key, rows in parsed.items() if key[1] == "pairs_nb" for row in rows]
    if len(pairs) != 1 or pairs[0][0][0] != "Protein_chain_A":
        raise ValueError("Expected one explicit protein-PET pair")
    row = pairs[0][1]
    if len(row) != 7 or row[2] != "1" or float(row[5]) != 0 or float(row[6]) != 0:
        raise ValueError("Unvalidated explicit pair function or nonzero LJ")
    molecules = next(rows for key, rows in parsed.items() if key[1] == "molecules")
    if molecules[0] != ["Protein_chain_A", "1"]:
        raise ValueError("Explicit pair coordinate offset changed")
    lines = coordinates.splitlines()
    box = [float(v) for v in lines[-1].split()]
    if len(box) != 3 or any(v <= 0 for v in box):
        raise ValueError("Only the published orthorhombic box is validated")
    positions = [[float(lines[int(a) + 1][i : i + 8]) for i in [20, 28, 36]] for a in row[:2]]
    delta = [(a - b) - length * round((a - b) / length) for a, b, length in zip(*positions, box, strict=True)]
    distance = math.sqrt(sum(v * v for v in delta))
    if distance <= 0:
        raise ValueError("Zero explicit pair distance")
    factor = 138.93545764438198  # GROMACS 2023.1 ONE_4PI_EPS0, kJ mol^-1 nm e^-2.
    value = factor * float(row[3]) * float(row[4]) / distance
    atoms = parsed[("Protein_chain_A", "atoms")]
    identities = [atoms[int(a) - 1] for a in row[:2]]
    return {
        "row": row,
        "atom_identities": identities,
        "both_outside_published_hot_region": all(not a[1].endswith("_") for a in identities),
        "distance_nm": distance,
        "energy_kj_mol": value,
        "coulomb_constant": factor,
        "source": "Published pairs_nb is retained by PLUMED; it is not part of the hot atom-charge scaling.",
    }


def energies(path):
    legends, rows = {}, []
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'@ s(\d+) legend "(.*)"', line.strip())
        if match:
            legends[int(match.group(1))] = match.group(2)
        elif line.strip() and not line.lstrip().startswith(("#", "@")):
            rows.append([float(x) for x in line.split()])
    if len(rows) != 1 or sorted(legends) != list(range(len(legends))) or len(rows[0]) != len(legends) + 1:
        raise ValueError("Expected exactly one complete energy frame")
    if not all(math.isfinite(x) for x in rows[0]):
        raise ValueError("Non-finite energy")
    result = {name: rows[0][i + 1] for i, name in legends.items()}
    if len(result) != len(legends):
        raise ValueError("Duplicated energy labels")
    return result


def check_energies(reference, unit_scale, all_hot_half, *, untempered_pair_coulomb=0.0):
    required = SCALED_TERMS | UNCHANGED_TERMS | {"Potential"}
    if (
        not required <= reference.keys()
        or reference.keys() != unit_scale.keys()
        or reference.keys() != all_hot_half.keys()
    ):
        raise ValueError("Energy components missing or changed")
    checks = []
    for term in sorted(required):
        if not all(math.isfinite(x[term]) for x in [reference, unit_scale, all_hot_half]):
            raise ValueError("Non-finite energy")
        expected = reference[term]
        got = unit_scale[term]
        if not math.isclose(got, expected, rel_tol=RTOL, abs_tol=ATOL):
            raise ValueError(f"Unit-scale energy mismatch: {term}")
        checks.append(
            {
                "check": "unit_scale",
                "term": term,
                "expected": expected,
                "observed": got,
                "absolute_error": abs(got - expected),
            }
        )
        if term == "Potential":
            expected = sum(reference[t] * (0.5 if t in SCALED_TERMS else 1) for t in SCALED_TERMS | UNCHANGED_TERMS)
        else:
            expected *= 0.5 if term in SCALED_TERMS else 1
        if term in {"Potential", "Coulomb (SR)"}:
            expected += 0.5 * untempered_pair_coulomb
        got = all_hot_half[term]
        if not math.isclose(got, expected, rel_tol=RTOL, abs_tol=ATOL):
            raise ValueError(f"All-hot half-scale energy mismatch: {term}: {got} vs {expected}")
        checks.append(
            {
                "check": "all_hot_half",
                "term": term,
                "expected": expected,
                "observed": got,
                "absolute_error": abs(got - expected),
            }
        )
    if not math.isclose(
        sum(reference[t] for t in SCALED_TERMS | UNCHANGED_TERMS), reference["Potential"], rel_tol=RTOL, abs_tol=ATOL
    ):
        raise ValueError("Potential energy decomposition incomplete")
    return checks


def main(args):
    source = args.preparation.resolve()
    report_before = sha(source / "report.json")
    original = json.loads((source / "report.json").read_text())
    if original["status"] != "completed" or original["replicas_prepared"] != 8:
        raise ValueError("Preparation incomplete")
    expected = dict(INPUT_SHA)
    expected.update({f"topol_{row['index']}.top": row["topology_sha256"] for row in original["replicas"]})
    for name, digest in expected.items():
        if sha(source / name) != digest:
            raise ValueError(f"Prepared input changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    args._owns_output = True
    start = time.monotonic()
    report = {
        "status": "running",
        "new_molecular_dynamics_performed": False,
        "full_paper_reproduced": False,
        "cases": [],
    }
    save(args.output / "report.json", report)
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    helper = Path(__file__).with_name("run_phl7_hrex_preparation.py")
    (args.output / helper.name).write_bytes(helper.read_bytes())
    env = {**os.environ, "OMP_NUM_THREADS": "2", "GMX_MAXBACKUP": "-1", "OPENBLAS_NUM_THREADS": "1"}
    commands = []
    waivers = []

    def command(label, argv, stdin=None, timeout=180, allow_charge_warning=None):
        began = time.monotonic()
        p = subprocess.run(argv, cwd=args.output, input=stdin, capture_output=True, text=True, env=env, timeout=timeout)
        (args.output / f"{label}.stdout.log").write_text(p.stdout)
        (args.output / f"{label}.stderr.log").write_text(p.stderr)
        commands.append(
            {"label": label, "argv": argv, "returncode": p.returncode, "wall_seconds": time.monotonic() - began}
        )
        save(args.output / "commands.json", commands)
        if p.returncode:
            warning = re.search(r"System has non-zero total charge: ([+-]?[0-9.]+)", p.stderr)
            if (
                allow_charge_warning is not None
                and warning is not None
                and p.stderr.count("WARNING ") == 1
                and "You are using Ewald electrostatics in a system with net charge." in p.stderr
                and "Too many warnings (1)." in p.stderr
                and math.isclose(float(warning.group(1)), allow_charge_warning, abs_tol=5.1e-7, rel_tol=0)
            ):
                receipt = {
                    "label": label,
                    "computed_total_charge": allow_charge_warning,
                    "reported_total_charge": float(warning.group(1)),
                    "maxwarn": 1,
                    "scope": "Fixed-frame diagnostic only. Not approval for production MD; fractional-charge PME warning retained.",
                }
                waivers.append(receipt)
                save(args.output / "fixed_frame_warning_waivers.json", waivers)
                return command(label + "_fixed_frame_only", argv + ["-maxwarn", "1"], stdin, timeout)
            raise RuntimeError(f"Command failed: {label}; inspect preserved logs")
        return p.stdout + p.stderr

    version = command("gromacs_version", [str(args.gmx), "--version"])
    if "GROMACS version:    2023.1" not in version or "Precision:          double" not in version:
        raise ValueError("Expected double-precision GROMACS 2023.1")
    executable = re.search(r"^Executable:\s+(.*)$", version, re.M).group(1).strip()
    protocol = {
        "scope": "Fixed-frame CPU validation; not dynamics, replica exchange or full-paper reproduction.",
        "prepared_inputs": expected,
        "preparation_report_sha256": report_before,
        "implementation_sha256": sha(Path(__file__)),
        "parser_sha256": sha(helper),
        "gromacs_launcher_sha256": sha(args.gmx),
        "gromacs_executable": executable,
        "gromacs_executable_sha256": sha(Path(executable)),
        "relative_energy_tolerance": RTOL,
        "absolute_energy_tolerance_kj_mol": ATOL,
        "energy_precision": "double GROMACS / 12-decimal xvg output",
        "threads": 2,
        "gpu": False,
        "plumed_binary_sha256": sha(args.plumed),
        "basis": "PLUMED 2.9 partial_tempering official unit-scale and all-hot half-scale tests",
        "all_hot_charge_precision": args.all_hot_charge_precision,
        "explicit_pair_policy": args.explicit_pair_policy,
        "all_hot_precision_scope": "Optional diagnostic changes synthetic all-hot control only; published partial-hot replica topologies remain byte-identical.",
        "basis_url": "https://www.plumed.org/doc-v2.9/user-doc/html/partial_tempering.html",
        "production_mdp_changed": False,
        "warning_policy": "Only the exact one non-zero-total-charge PME warning can be retained for a fixed-frame rerun; charge must independently match the topology. Never blanket-ignore warnings.",
        "frame": "Published equilibrated.gro only; no sampling claim",
    }
    save(args.output / "protocol.json", protocol)
    top = (source / "hrex.top").read_text()
    gro = (source / "equilibrated.gro").read_text()
    special_pair = explicit_pair_coulomb(top, gro)
    save(args.output / "explicit_pair.json", special_pair)
    index, counts = build_index(top, gro)
    (args.output / "index.ndx").write_text(index)
    save(
        args.output / "index.json",
        {
            "atom_counts": counts,
            "rule": "Exact expanded topology/GRO atom-name and residue-name match; protein+PET is Protein_chain_A, solvent/ions are SOL/NA/CL.",
        },
    )
    for name in ["equilibrated.gro", "hrex.mdp"]:
        (args.output / name).write_bytes((source / name).read_bytes())
    unmarked, unmarked_count = mark_atoms(top, "unmark")
    (args.output / "unscaled.top").write_text(unmarked)
    all_hot, all_marked_count = mark_atoms(unmarked, "all_hot")
    (args.output / "all_hot_input.top").write_text(all_hot)
    plumed_env_path = str(args.plumed.resolve().parents[1] / "bin")
    env["PATH"] = plumed_env_path + ":/usr/bin:/bin"
    with (args.output / "all_hot_input.top").open() as f:
        command("plumed_all_hot_half", [str(args.plumed), "partial_tempering", "0.5"], f.read())
    if "ERROR:" in (args.output / "plumed_all_hot_half.stderr.log").read_text():
        raise ValueError("PLUMED scaling reported an error")
    (args.output / "all_hot_half.top").write_bytes((args.output / "plumed_all_hot_half.stdout.log").read_bytes())
    if args.all_hot_charge_precision == "full":
        baseline = (args.output / "all_hot_half.top").read_text()
        (args.output / "all_hot_half.plumed_original.top").write_text(baseline)
        repaired = restore_scaled_charge_precision(baseline, unmarked, 0.5)
        (args.output / "all_hot_half.top").write_text(repaired)
    for i in range(8):
        (args.output / f"topol_{i}.top").write_bytes((source / f"topol_{i}.top").read_bytes())
    specs = [("unscaled", "unscaled.top"), ("all_hot_half", "all_hot_half.top")]
    specs += [(f"replica_{i}", f"topol_{i}.top") for i in range(8)]
    collected = {}
    for label, topology in specs:
        tpr = f"{label}.tpr"
        command(
            label + "_grompp",
            [
                str(args.gmx),
                "grompp",
                "-f",
                "hrex.mdp",
                "-c",
                "equilibrated.gro",
                "-p",
                topology,
                "-n",
                "index.ndx",
                "-o",
                tpr,
                "-po",
                f"{label}.mdout.mdp",
                "-pp",
                f"{label}.processed.top",
            ],
            allow_charge_warning=topology_charge((args.output / topology).read_text())
            if label not in {"unscaled", "replica_0"}
            else None,
        )
        command(
            label + "_rerun",
            [
                str(args.gmx),
                "mdrun",
                "-s",
                tpr,
                "-rerun",
                "equilibrated.gro",
                "-deffnm",
                label,
                "-ntmpi",
                "1",
                "-ntomp",
                "2",
                "-nb",
                "cpu",
                "-pme",
                "cpu",
                "-pin",
                "off",
                "-reprod",
            ],
        )
        command(
            label + "_energy",
            [str(args.gmx), "energy", "-f", label + ".edr", "-o", label + ".xvg", "-dp"],
            "\n".join(term.replace(" ", "-") for term in sorted(SCALED_TERMS | UNCHANGED_TERMS | {"Potential"}))
            + "\n0\n",
        )
        collected[label] = energies(args.output / (label + ".xvg"))
        if set(collected[label]) != SCALED_TERMS | UNCHANGED_TERMS | {"Potential"}:
            raise ValueError("Incomplete energy selection")
        report["cases"].append(
            {
                "label": label,
                "topology_sha256": sha(args.output / topology),
                "tpr_sha256": sha(args.output / tpr),
                "energy_sha256": sha(args.output / (label + ".edr")),
                "components": collected[label],
            }
        )
        save(args.output / "report.json", report)
    pair_energy = special_pair["energy_kj_mol"] if args.explicit_pair_policy == "published_untempered" else 0.0
    checks = check_energies(
        collected["unscaled"], collected["replica_0"], collected["all_hot_half"], untempered_pair_coulomb=pair_energy
    )
    for name, digest in expected.items():
        if sha(source / name) != digest:
            raise ValueError("Historical input changed during validation")
    if sha(source / "report.json") != report_before:
        raise ValueError("Historical report changed")
    report.update(
        status="completed",
        energy_replay_performed=True,
        generic_uniform_half_scaling_validated=args.explicit_pair_policy == "uniform_half",
        explicit_pair_aware_scaling_validated=args.explicit_pair_policy == "published_untempered",
        explicit_pair=special_pair,
        all_hot_diagnostic_charge_precision=args.all_hot_charge_precision,
        fixed_frames=1,
        prepared_replica_inputs=8,
        energy_comparisons=checks,
        atom_counts=counts,
        source_hot_atoms_unmarked=unmarked_count,
        all_hot_atoms_marked=all_marked_count,
        wall_seconds=time.monotonic() - start,
        protocol_sha256=sha(args.output / "protocol.json"),
        historical_inputs_unchanged=True,
        fixed_frame_warning_waivers=waivers,
        production_inputs_approved=False,
        limits=[
            "One fixed frame only; not trajectory sampling or production stability",
            "No MPI or PLUMED-coupled replica exchange executed",
            "Missing variant initial systems remain unresolved",
        ],
    )
    save(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "energy_cases": len(specs),
                "comparisons": len(checks),
                "new_molecular_dynamics_performed": False,
            }
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
            report = json.loads(report_path.read_text())
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            save(report_path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--gmx", type=Path, required=True)
    parser.add_argument("--plumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-hot-charge-precision", choices=["published", "full"], default="published")
    parser.add_argument(
        "--explicit-pair-policy", choices=["uniform_half", "published_untempered"], default="uniform_half"
    )
    execute(parser.parse_args())
