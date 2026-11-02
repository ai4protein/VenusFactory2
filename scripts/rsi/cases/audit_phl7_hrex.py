
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
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description="Independent native readback of bounded PHL7 HREX integration")
for name in ["run", "preparation", "validation", "build", "output"]:
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
root, prep, validation, build, out = [
    getattr(args, k).resolve() for k in ["run", "preparation", "validation", "build", "output"]
]
assert all(not out.is_relative_to(p) for p in [root, prep, validation, build]), (
    "Audit output must be outside source evidence"
)
out.mkdir(parents=True, exist_ok=False)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def mdp(p):
    result = {}
    for line in p.read_text().splitlines():
        line = line.split(";")[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip().replace("_", "-")
            assert key not in result
            result[key] = value.strip()
    return result


def series(p):
    legends = {int(i): name for i, name in re.findall(r'^@ s(\d+) legend "([^\"]+)"', p.read_text(), re.M)}
    assert sorted(legends) == [0, 1, 2, 3] and set(legends.values()) == {
        "Potential",
        "Temperature",
        "Pressure",
        "Density",
    }
    rows = [
        [float(v) for v in line.split()]
        for line in p.read_text().splitlines()
        if line.strip() and not line.startswith(("#", "@"))
    ]
    assert len(rows) == 21 and all(len(row) == 5 and all(math.isfinite(v) for v in row) for row in rows)
    assert all(abs(row[0] - i * 0.2) < 1e-6 for i, row in enumerate(rows))
    summary = {}
    for i, name in legends.items():
        values = [row[i + 1] for row in rows]
        summary[name] = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    return rows, summary


before = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
t = time.monotonic()
try:
    report = read(root / "report.json")
    protocol = read(root / "protocol.json")
    runtime = read(build / "build_result.json")
    assert (
        report["status"] == "completed"
        and report["new_molecular_dynamics_performed"] is True
        and report["full_paper_reproduced"] is False
    )
    assert report["protocol_sha256"] == sha(root / "protocol.json")
    assert (
        protocol["steps"] == 2000
        and protocol["dt_ps"] == 0.002
        and protocol["replicas"] == 8
        and protocol["ps_per_replica"] == 4
    )
    assert protocol["exchange_interval_steps"] == 100 and protocol["exchange_seed"] == 43001
    assert protocol["ranks"] == 8 and protocol["threads_per_rank"] == 1
    assert (
        protocol["cpu_allocation"]["requested_threads"]
        == 8
        <= protocol["cpu_allocation"]["effective_logical_cpu_budget"]
    )
    assert runtime["status"] == "completed" and runtime["runtime_sha256"] == protocol["runtime"]
    assert all(sha(Path(name)) == value for name, value in protocol["runtime"].items())
    assert protocol["build_report_sha256"] == sha(build / "build_result.json") and protocol[
        "patch_audit_sha256"
    ] == sha(build / "patch_audit.json")
    assert protocol["energy_validation_report_sha256"] == sha(validation / "report.json")
    assert all(sha(root / name) == value for name, value in protocol["implementation"].items())
    assert len(report["replicas"]) == 8 and [x["index"] for x in report["replicas"]] == list(range(8))
    assert len(report["warnings"]) == 7 and [x["case"] for x in report["warnings"]] == [
        f"replica_{i}_grompp" for i in range(1, 8)
    ]
    prefix = Path(read(build / "install_request.json")["prefix"])
    gmx = Path(runtime.get("gmx_binary", prefix / "gromacs-2023.1-plumed/bin/gmx_mpi"))
    env = {
        **os.environ,
        "PATH": str(prefix / "bin") + ":/usr/bin:/bin",
        "LD_LIBRARY_PATH": ":".join(runtime.get("runtime_library_dirs", [str(prefix / "lib")])),
        "PLUMED_KERNEL": str(prefix / "lib/libplumedKernel.so"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    calls = []

    def native(label, cmd, stdin=None):
        with (out / (label + ".log")).open("w") as log:
            completed = subprocess.run(
                [str(gmx), *cmd],
                cwd=out,
                env=env,
                input=stdin,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
        assert completed.returncode == 0, label
        calls.append({"label": label, "argv": [str(gmx), *cmd], "returncode": completed.returncode})

    rows = []
    exchange_reference = None
    for i, item in enumerate(report["replicas"]):
        rep = root / f"replica_{i}"
        derived = protocol["derived_inputs"][str(i)]
        assert all(sha(rep / name) == value for name, value in item["artifacts"].items())
        assert sha(rep / "topol.top") == sha(prep / f"topol_{i}.top") == derived["topology_sha256"]
        assert (
            sha(rep / "start.gro") == sha(prep / "equilibrated.gro") and sha(rep / "smoke.mdp") == derived["mdp_sha256"]
        )
        expected = mdp(prep / "hrex.mdp")
        expected.update(
            {
                "nsteps": "2000",
                "nstxout": "0",
                "nstlog": "100",
                "nstenergy": "100",
                "nstxout-compressed": "100",
                "ld-seed": str(43001 + i),
                "gen-seed": str(43001 + i),
            }
        )
        assert mdp(rep / "smoke.mdp") == expected
        log = (rep / "md.log").read_text()
        assert not re.search(r"LINCS WARNING|constraint error|Fatal error:", log, re.I)
        assert "Finished mdrun" in log and "PLUMED" in log and "2.9.0" in log
        if protocol.get("compute_backend") == "cuda":
            assert re.search(r"PP tasks will do .* interactions on the GPU", log)
        events = [int(x) for x in re.findall(r"Replica exchange at step\s+(\d+)", log)]
        assert events == list(range(100, 2000, 100)) == item["exchange_steps"]
        attempts = re.search(r"Repl\s+(\d+) attempts, (\d+) odd, (\d+) even", log)
        assert tuple(map(int, attempts.groups())) == (19, 10, 9)
        accepted = [0] * 7
        event_lines = re.findall(r"^Repl ex\s+(.+)$", log, re.M)
        assert len(event_lines) == 19
        for j, line in enumerate(event_lines):
            tokens = line.split()
            assert [int(x) for x in tokens if x != "x"] == list(range(8))
            for k, value in enumerate(tokens):
                if value == "x":
                    left, right = int(tokens[k - 1]), int(tokens[k + 1])
                    assert right == left + 1 and left % 2 == j % 2
                    accepted[left] += 1
        summary = re.search(r"Repl  number of exchanges:\s*\nRepl[^\n]+\nRepl([^\n]+)", log)
        assert list(map(int, summary.group(1).split())) == accepted
        if exchange_reference is None:
            exchange_reference = event_lines
        assert event_lines == exchange_reference
        native(
            f"replica_{i}_energy",
            ["energy", "-f", str(rep / "md.edr"), "-o", str(out / f"replica_{i}_energy.xvg"), "-dp"],
            "Potential\nTemperature\nPressure\nDensity\n0\n",
        )
        values, stats = series(out / f"replica_{i}_energy.xvg")
        saved, saved_stats = series(rep / "observables.xvg")
        assert values == saved and stats == saved_stats == item["observable_summary"]
        assert 150 < stats["Temperature"]["min"] <= stats["Temperature"]["max"] < 450
        assert 800 < stats["Density"]["min"] <= stats["Density"]["max"] < 1200
        native(f"replica_{i}_trajectory", ["check", "-f", str(rep / "md.xtc")])
        check = (out / f"replica_{i}_trajectory.log").read_text()
        assert re.search(r"Last frame\s+20 time\s+4\.0+", check), check[-1200:]
        final = (rep / "md.gro").read_text().splitlines()
        assert len(final) == 40560 and int(final[1]) == 40557
        assert all(math.isfinite(float(line[j : j + 8])) for line in final[2:-1] for j in range(20, 68, 8))
        rows.append(
            {
                "replica": i,
                "frames": 21,
                "ps": 4,
                "schedule_events": 19,
                "edge_attempts": [10, 9, 10, 9, 10, 9, 10],
                "edge_accepted": accepted,
                "temperature_K": stats["Temperature"],
                "density_kg_m3": stats["Density"],
                "edr_sha256": sha(rep / "md.edr"),
                "xtc_sha256": sha(rep / "md.xtc"),
            }
        )
    after = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
    assert before == after
    result = {
        "status": "passed",
        "external_model_calls": 0,
        "new_md_performed_by_audit": False,
        "historical_files_unchanged": True,
        "observed_original_new_md": True,
        "replicas": rows,
        "global_exchange_schedule_events": 19,
        "global_pair_attempts": 67,
        "global_accepted_swaps": sum(rows[0]["edge_accepted"]),
        "native_commands": calls,
        "source_run": str(root),
        "source_report_sha256": sha(root / "report.json"),
        "auditor_sha256": sha(Path(__file__)),
        "wall_seconds": time.monotonic() - t,
        "scope": "Independent terminal artifact binding, native EDR/trajectory readback and exchange recount only. Bounded 4 ps test; executor provenance is assessed separately. No convergence, thermodynamic correctness or production-stability claim.",
    }
except Exception as exc:
    result = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "wall_seconds": time.monotonic() - t,
    }
    traceback.print_exc()
save(out / "report.json", result)
print(
    json.dumps(
        {k: result.get(k) for k in ["status", "error", "wall_seconds", "global_pair_attempts", "global_accepted_swaps"]}
    ),
    flush=True,
)

raise SystemExit(0 if result["status"] == "passed" else 1)
