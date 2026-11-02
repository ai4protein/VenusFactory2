"""Bounded eight-replica CPU integration check of the public PHL7 WT HREX setup.

Four ps per replica checks execution, finite observables and exchange scheduling;
it cannot establish equilibrium, mixing, production stability or paper efficacy.
"""

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
import time
from pathlib import Path

from command_lifecycle import MPI_TERMINATION_GRACE, install_termination_handlers, run_in_group
from scripts.rsi.cases.run_phl7_hrex_energy_validation import build_index, topology_charge

STEPS = 2000
EXCHANGE = 100
THREADS = 1
REPLICAS = 8
SEED = 43001


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def allocation_plan(affinity, quota_limits):
    affinity = sorted(set(affinity))
    if not affinity or any(not math.isfinite(x) or x <= 0 for x in quota_limits):
        raise ValueError("Invalid CPU allocation evidence")
    capacity = min([len(affinity), *quota_limits])
    requested = REPLICAS * THREADS
    if requested > math.floor(capacity):
        raise ValueError("Eight MPI ranks exceed the current CPU allocation")
    return {
        "affinity_logical_cpus": affinity,
        "finite_cgroup_quota_cpus": quota_limits,
        "effective_logical_cpu_budget": math.floor(capacity),
        "requested_threads": requested,
        "mpi_slot_policy": "use allocated hardware threads, without oversubscription",
    }


def cpu_allocation():
    limits = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, group = line.split(":", 2)
        if "cpu" in controllers.split(","):
            root, names = Path("/sys/fs/cgroup/cpu"), ("cpu.cfs_quota_us", "cpu.cfs_period_us")
        elif not controllers:
            root, names = Path("/sys/fs/cgroup"), ("cpu.max",)
        else:
            continue
        folder = root / group.lstrip("/")
        if ".." in folder.parts:
            raise ValueError("Unexpected cgroup path")
        while True:
            if all((folder / name).is_file() for name in names):
                values = (
                    (folder / names[0]).read_text().split()
                    if len(names) == 1
                    else [(folder / name).read_text().strip() for name in names]
                )
                quota, period = values
                if quota not in ("max", "-1"):
                    limits.append(int(quota) / int(period))
            if folder == root:
                break
            folder = folder.parent
    return allocation_plan(os.sched_getaffinity(0), limits)


def short_mdp(source, replica):
    updates = {
        "nsteps": str(STEPS),
        "nstxout": "0",
        "nstlog": "100",
        "nstenergy": "100",
        "nstxout-compressed": "100",
        "ld-seed": str(SEED + replica),
        "gen-seed": str(SEED + replica),
    }
    output, seen = [], set()
    for line in source.splitlines():
        if "=" in line and not line.lstrip().startswith(";"):
            key = line.split("=", 1)[0].strip().replace("_", "-")
            if key in updates:
                line = key + " = " + updates[key]
                seen.add(key)
        output.append(line)
    output.extend(key + " = " + value for key, value in updates.items() if key not in seen)
    return "\n".join(output) + "\n", updates


def parse_series(path):
    legends, rows = {}, []
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'@ s(\d+) legend "(.*)"', line.strip())
        if match:
            legends[int(match.group(1))] = match.group(2)
        elif line.strip() and not line.lstrip().startswith(("#", "@")):
            rows.append([float(x) for x in line.split()])
    if set(legends.values()) != {"Potential", "Temperature", "Pressure", "Density"} or sorted(legends) != list(
        range(4)
    ):
        raise ValueError("Required observables missing")
    if len(rows) != STEPS // 100 + 1 or any(len(row) != 5 or not all(math.isfinite(x) for x in row) for row in rows):
        raise ValueError("Missing or nonfinite observable frames")
    if any(not math.isclose(row[0], i * 0.2, abs_tol=1e-6) for i, row in enumerate(rows)):
        raise ValueError("Simulation duration or output interval differs")
    result = {name: [row[i + 1] for row in rows] for i, name in legends.items()}
    if not all(150 < x < 450 for x in result["Temperature"]):
        raise ValueError("Short-run temperature envelope exceeded")
    if not all(800 < x < 1200 for x in result["Density"]):
        raise ValueError("Short-run density envelope exceeded")
    return {
        name: {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
        for name, values in result.items()
    }


def exchange_steps(log):
    steps = [int(x) for x in re.findall(r"Replica exchange at step\s+(\d+)", log)]
    if steps != list(range(EXCHANGE, STEPS, EXCHANGE)):
        raise ValueError("Replica exchange schedule incomplete")
    if re.search(r"LINCS WARNING|constraint error|Fatal error:", log, flags=re.I):
        raise ValueError("Integration reported an error")
    return steps


def execute(args):
    args._owns_output = False
    try:
        main(args)
    except Exception as exc:
        report = args.output / "report.json"
        if args._owns_output and report.exists():
            value = json.loads(report.read_text())
            value.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            save(report, value)
        raise


def main(args):
    setup = json.loads((args.build / "build_result.json").read_text())
    patch = json.loads((args.build / "patch_audit.json").read_text())
    validation = json.loads((args.validation / "report.json").read_text())
    if (
        setup["status"] != "completed"
        or patch["status"] != "passed"
        or validation["status"] != "completed"
        or not validation["explicit_pair_aware_scaling_validated"]
    ):
        raise ValueError("MPI build and fixed-frame energy validation must pass first")
    prefix = Path(json.loads((args.build / "install_request.json").read_text())["prefix"])
    gmx = Path(setup.get("gmx_binary", prefix / "gromacs-2023.1-plumed/bin/gmx_mpi"))
    backend = getattr(args, "backend", "cpu")
    if backend not in {"cpu", "cuda"} or (backend == "cuda" and setup.get("gpu_backend") != "CUDA"):
        raise ValueError("Selected compute backend is not supported by the verified build")
    for name, value in setup["runtime_sha256"].items():
        if sha(Path(name)) != value:
            raise ValueError("Compiled MPI runtime changed")
    source = args.preparation
    prepared = json.loads((source / "report.json").read_text())
    for row in prepared["replicas"]:
        if sha(source / f"topol_{row['index']}.top") != row["topology_sha256"]:
            raise ValueError("Published prepared topology changed")
    protocol_source = json.loads((args.validation / "protocol.json").read_text())
    for name in ["hrex.mdp", "hrex.top", "equilibrated.gro"]:
        if sha(source / name) != protocol_source["prepared_inputs"][name]:
            raise ValueError("Published source input changed")
    allocation = cpu_allocation()
    args.output.mkdir(parents=True, exist_ok=False)
    args._owns_output = True
    started = time.monotonic()
    report = {
        "status": "running",
        "new_molecular_dynamics_performed": False,
        "full_paper_reproduced": False,
        "replicas": [],
        "warnings": [],
    }
    save(args.output / "report.json", report)
    files = [
        Path(__file__),
        Path(__file__).with_name("run_phl7_hrex_energy_validation.py"),
        Path(__file__).with_name("run_phl7_hrex_preparation.py"),
        Path(__file__).with_name("command_lifecycle.py"),
    ]
    for p in files:
        (args.output / p.name).write_bytes(p.read_bytes())
    env = {
        **os.environ,
        "PATH": str(prefix / "bin") + ":/usr/bin:/bin",
        "LD_LIBRARY_PATH": ":".join(setup.get("runtime_library_dirs", [str(prefix / "lib")])),
        "PLUMED_KERNEL": str(prefix / "lib/libplumedKernel.so"),
        "OMP_NUM_THREADS": str(THREADS),
        "OPENBLAS_NUM_THREADS": "1",
    }
    commands = []

    def command(label, cmd, *, cwd=args.output, stdin=None, timeout=180, accept_charge=None):
        t = time.monotonic()
        log = args.output / (label + ".log")
        with log.open("w") as stream:
            code = run_in_group(
                cmd, cwd=cwd, env=env, log=stream, stdin=stdin, timeout=timeout, termination_grace=MPI_TERMINATION_GRACE
            )
        commands.append({"label": label, "command": cmd, "returncode": code, "wall_seconds": time.monotonic() - t})
        save(args.output / "commands.json", commands)
        if code:
            text = log.read_text()
            found = re.search(r"System has non-zero total charge: ([+-]?[0-9.]+)", text)
            if (
                accept_charge is not None
                and found
                and text.count("WARNING ") == 1
                and "Too many warnings (1)." in text
                and "You are using Ewald electrostatics in a system with net charge." in text
                and math.isclose(float(found.group(1)), accept_charge, rel_tol=0, abs_tol=5.1e-7)
            ):
                report["warnings"].append(
                    {
                        "case": label,
                        "net_charge": accept_charge,
                        "scope": "Published partial-hot charges retained for bounded 4 ps integration only; PME background convention unchanged. No production-stability approval.",
                    }
                )
                save(args.output / "report.json", report)
                return command(label + "_bounded_charge_warning", cmd + ["-maxwarn", "1"], cwd=cwd, timeout=timeout)
            raise RuntimeError("Command failed: " + label)

    command("gromacs_version", [str(gmx), "--version"])
    command("plumed_version", [str(prefix / "bin/plumed"), "info", "--version"])
    index, atom_counts = build_index((source / "hrex.top").read_text(), (source / "equilibrated.gro").read_text())
    if sum(atom_counts.values()) != 40557:
        raise ValueError("Unexpected system size")
    derived = {}
    for i in range(REPLICAS):
        rep = args.output / f"replica_{i}"
        rep.mkdir()
        (rep / "index.ndx").write_text(index)
        for src, dst in [("equilibrated.gro", "start.gro"), (f"topol_{i}.top", "topol.top")]:
            (rep / dst).write_bytes((source / src).read_bytes())
        mdp, updates = short_mdp((source / "hrex.mdp").read_text(), i)
        (rep / "smoke.mdp").write_text(mdp)
        (rep / "plumed.dat").write_text(
            "# No additional bias; GROMACS/PLUMED handles the published Hamiltonian exchanges.\n"
        )
        derived[str(i)] = {
            "mdp_updates": updates,
            "mdp_sha256": sha(rep / "smoke.mdp"),
            "topology_sha256": sha(rep / "topol.top"),
        }
        command(
            f"replica_{i}_grompp",
            [
                str(gmx),
                "grompp",
                "-f",
                "smoke.mdp",
                "-c",
                "start.gro",
                "-p",
                "topol.top",
                "-n",
                "index.ndx",
                "-o",
                "topol.tpr",
                "-po",
                "mdout.mdp",
            ],
            cwd=rep,
            accept_charge=topology_charge((rep / "topol.top").read_text()) if i else None,
        )
    protocol = {
        "scope": "Bounded HREX integration only; not equilibration, mixing, production stability or paper reproduction",
        "compute_backend": backend,
        "steps": STEPS,
        "dt_ps": 0.002,
        "replicas": REPLICAS,
        "ps_per_replica": 4,
        "exchange_interval_steps": EXCHANGE,
        "exchange_interval_source": "Engineering diagnostic; publication launch interval not established from inspected archive",
        "exchange_seed": SEED,
        "ranks": REPLICAS,
        "threads_per_rank": THREADS,
        "cpu_allocation": allocation,
        "runtime": setup["runtime_sha256"],
        "build_report_sha256": sha(args.build / "build_result.json"),
        "patch_audit_sha256": sha(args.build / "patch_audit.json"),
        "energy_validation_report_sha256": sha(args.validation / "report.json"),
        "implementation": {p.name: sha(p) for p in files},
        "derived_inputs": derived,
        "temperature_envelope_kelvin": [150, 450],
        "density_envelope_kg_m3": [800, 1200],
        "starting_state": "Published coordinates/velocities; thermostat/barostat checkpoint unavailable",
    }
    save(args.output / "protocol.json", protocol)
    cmd = [
        str(prefix / "bin/mpiexec"),
        "--use-hwthread-cpus",
        "-np",
        str(REPLICAS),
        "--bind-to",
        "none",
        str(gmx),
        "mdrun",
        "-multidir",
        *[f"replica_{i}" for i in range(REPLICAS)],
        "-s",
        "topol.tpr",
        "-deffnm",
        "md",
        "-plumed",
        "plumed.dat",
        "-hrex",
        "-replex",
        str(EXCHANGE),
        "-reseed",
        str(SEED),
        "-ntomp",
        str(THREADS),
        "-nb",
        "gpu" if backend == "cuda" else "cpu",
        "-pme",
        "cpu",
        "-pin",
        "off",
        "-reprod",
    ]
    report["new_molecular_dynamics_requested"] = True
    report["new_molecular_dynamics_performed"] = None
    save(args.output / "report.json", report)
    command("mpi_hrex", cmd, timeout=900)
    report["new_molecular_dynamics_performed"] = True
    for i in range(REPLICAS):
        rep = args.output / f"replica_{i}"
        command(
            f"replica_{i}_energy",
            [str(gmx), "energy", "-f", "md.edr", "-o", "observables.xvg", "-dp"],
            cwd=rep,
            stdin="Potential\nTemperature\nPressure\nDensity\n0\n",
        )
        stats = parse_series(rep / "observables.xvg")
        exchanges = exchange_steps((rep / "md.log").read_text())
        if backend == "cuda" and not re.search(
            r"PP tasks will do .* interactions on the GPU", (rep / "md.log").read_text()
        ):
            raise ValueError("Requested CUDA offload lacks native GPU execution evidence")
        final = (rep / "md.gro").read_text().splitlines()
        if (
            int(final[1]) != 40557
            or len(final) != 40560
            or not all(math.isfinite(float(line[j : j + 8])) for line in final[2:-1] for j in range(20, 68, 8))
        ):
            raise ValueError("Invalid final coordinates or velocities")
        report["replicas"].append(
            {
                "index": i,
                "observable_summary": stats,
                "exchange_steps": exchanges,
                "artifacts": {
                    name: sha(rep / name)
                    for name in ["topol.tpr", "md.gro", "md.edr", "md.log", "md.cpt", "observables.xvg"]
                },
            }
        )
        save(args.output / "report.json", report)
    text = (args.output / "mpi_hrex.log").read_text()
    if "PLUMED" not in text or "2.9.0" not in text:
        raise ValueError("PLUMED runtime evidence missing")
    report.update(
        status="completed",
        ps_per_replica=4,
        replicas_completed=8,
        exchange_attempts_per_replica=19,
        wall_seconds=time.monotonic() - started,
        protocol_sha256=sha(args.output / "protocol.json"),
        limitations=[
            "Four ps only: no equilibrium, mixing or production-stability claim",
            "Known PME fractional-charge warnings retained",
            "Original thermostat/barostat checkpoint and full mutant initial systems unavailable",
            "Author production exchange interval has not been established",
        ],
    )
    save(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "replicas": 8, "ps_per_replica": 4}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    install_termination_handlers()
    execute(parser.parse_args())
