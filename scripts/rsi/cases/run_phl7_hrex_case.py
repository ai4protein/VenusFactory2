"""New PHL7 WT preparation, independent energy checks and bounded HREX integration."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
import os
import sys
import time
from pathlib import Path

from command_lifecycle import install_termination_handlers, run_in_group

from agent.rsi.cases.phl7_case import ARCHIVE_NAME, ARCHIVE_SHA256
from agent.rsi.cases.phl7_hrex import PLUGIN_ID, runtime_configuration, verify_artifacts
from agent.rsi.engine.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]


def run(output, backend):
    output = output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    if backend not in {"cpu", "cuda"}:
        raise ValueError("Unknown HREX backend")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running",
        "execution_origin": "new_computation",
        "compute_backend": backend,
        "archive_sha256": ARCHIVE_SHA256,
        "new_molecular_dynamics_performed": False,
        "full_paper_reproduced": False,
        "stages": [],
    }
    atomic_json(output / "report.json", report)
    try:
        runtime, paths = runtime_configuration(ROOT, backend)
        build = paths["build"]
        atomic_json(output / "runtime_manifest.json", runtime)
        files = [
            Path(__file__),
            Path(__file__).parent.parent / "command_lifecycle.py",
            *[
                Path(__file__).with_name(name)
                for name in [
                    "run_phl7_hrex_preparation.py",
                    "run_phl7_hrex_energy_validation.py",
                    "run_phl7_hrex_smoke.py",
                    "audit_phl7_hrex.py",
                ]
            ],
            ROOT / "src/agent/rsi/cases/phl7_hrex.py",
        ]
        (output / "implementation").mkdir()
        for path in files:
            (output / "implementation" / path.name).write_bytes(path.read_bytes())
        protocol = {
            "plugin_id": PLUGIN_ID,
            "compute_backend": backend,
            "archive_sha256": ARCHIVE_SHA256,
            "implementation": {path.name: sha256_file(path) for path in files},
            "build_report_sha256": sha256_file(build / "build_result.json"),
            "runtime_manifest_sha256": sha256_file(output / "runtime_manifest.json"),
            "scope": "One public WT system, eight replicas, four ps; no full-paper or efficacy claim",
        }
        atomic_json(output / "protocol.json", protocol)
        report["protocol_sha256"] = sha256_file(output / "protocol.json")
        archive = ROOT / "research_runs/rsi/case_assets/PHL7/zenodo_18650810" / ARCHIVE_NAME
        plumed, gmx = paths["plumed"], paths["gmx_double"]
        preparation, validation, dynamics, audit = [
            output / key for key in ["preparation", "energy_validation", "dynamics", "native_audit"]
        ]
        specifications = [
            (
                "preparation",
                "run_phl7_hrex_preparation.py",
                ["--archive", str(archive), "--plumed", str(plumed), "--output", str(preparation)],
                300,
            ),
            (
                "energy_validation",
                "run_phl7_hrex_energy_validation.py",
                [
                    "--preparation",
                    str(preparation),
                    "--gmx",
                    str(gmx),
                    "--plumed",
                    str(plumed),
                    "--output",
                    str(validation),
                    "--all-hot-charge-precision",
                    "full",
                    "--explicit-pair-policy",
                    "published_untempered",
                ],
                600,
            ),
            (
                "dynamics",
                "run_phl7_hrex_smoke.py",
                [
                    "--preparation",
                    str(preparation),
                    "--validation",
                    str(validation),
                    "--build",
                    str(build),
                    "--output",
                    str(dynamics),
                    "--backend",
                    backend,
                ],
                1200,
            ),
            (
                "native_audit",
                "audit_phl7_hrex.py",
                [
                    "--run",
                    str(dynamics),
                    "--preparation",
                    str(preparation),
                    "--validation",
                    str(validation),
                    "--build",
                    str(build),
                    "--output",
                    str(audit),
                ],
                300,
            ),
        ]
        for name, script, arguments, timeout in specifications:
            row = {"id": name, "status": "running"}
            report["stages"].append(row)
            if name == "dynamics":
                # A killed child may have integrated steps without a final report.
                report["new_molecular_dynamics_performed"] = None
            atomic_json(output / "report.json", report)
            command = [sys.executable, "-u", str(output / "implementation" / script), *arguments]
            t = time.monotonic()
            with (output / (name + ".log")).open("w") as stream:
                code = run_in_group(command, cwd=ROOT, env=os.environ.copy(), log=stream, timeout=timeout)
            row.update(returncode=code, wall_seconds=time.monotonic() - t)
            if code:
                raise RuntimeError("PHL7 stage failed: " + name)
            path = output / name / "report.json"
            body = json.loads(path.read_text())
            if body["status"] != ("passed" if name == "native_audit" else "completed"):
                raise ValueError("PHL7 stage evidence is incomplete: " + name)
            row.update(status="completed", report_sha256=sha256_file(path))
            if name == "dynamics":
                report["new_molecular_dynamics_performed"] = body["new_molecular_dynamics_performed"]
            atomic_json(output / "report.json", report)
        native = json.loads((audit / "report.json").read_text())
        if any(sha256_file(path) != protocol["implementation"][path.name] for path in files):
            raise ValueError("Pipeline source changed during execution")
        report.update(
            status="completed",
            summary={
                "replicas": 8,
                "ps_per_replica": 4,
                "frames_per_replica": 21,
                "exchange_schedule_events": native["global_exchange_schedule_events"],
                "pair_attempts": native["global_pair_attempts"],
                "accepted_swaps": native["global_accepted_swaps"],
                "energy_identities": 24,
                "atoms_per_replica": 40557,
            },
        )
        report["evidence"] = verify_artifacts(report, output)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if report["stages"] and report["stages"][-1]["status"] == "running":
            report["stages"][-1]["status"] = "failed"
        dynamics_report = output / "dynamics/report.json"
        if report["new_molecular_dynamics_performed"] is None and dynamics_report.is_file():
            try:
                partial = json.loads(dynamics_report.read_text()).get("new_molecular_dynamics_performed")
                if partial is None or type(partial) is bool:
                    report["new_molecular_dynamics_performed"] = partial
            except (OSError, ValueError):
                pass
    report["wall_seconds"] = time.monotonic() - started
    atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    install_termination_handlers()
    result = run(args.output, args.backend)
    print(json.dumps({k: result.get(k) for k in ["status", "compute_backend", "summary", "error"]}), flush=True)
    raise SystemExit(0 if result["status"] == "completed" else 1)
