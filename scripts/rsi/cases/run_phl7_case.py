"""Replay all 24 released PHL7 OPES analyses, retaining mismatches and failures."""
from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from agent.rsi import phl7_case
from agent.rsi.engine.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]


def run(output):
    output = output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    if output.exists():
        raise FileExistsError(output)
    archive = ROOT / "research_runs/rsi/case_assets/PHL7/zenodo_18650810" / phl7_case.ARCHIVE_NAME
    protocol = phl7_case.admit(archive)
    executable = ROOT / "research_runs/rsi/envs/phl7-plumed290/bin/plumed"
    version = subprocess.check_output([str(executable), "info", "--long-version"], text=True).strip()
    if version != "2.9.0":
        raise ValueError("Expected PLUMED 2.9.0; no version fallback")
    environment = {p.name: sha256_file(p) for p in (executable.parent.parent / "conda-meta").glob("*.json")}
    protocol.update(environment_metadata_sha256=environment, plumed_version=version, plumed_sha256=sha256_file(executable),
                    module_sha256=sha256_file(Path(phl7_case.__file__)), entry_sha256=sha256_file(Path(__file__)),
                    per_analysis_timeout_s=300, cpu_threads=1)
    output.mkdir(parents=True)
    shutil.copy2(Path(phl7_case.__file__), output / "phl7_case.py")
    shutil.copy2(Path(__file__), output / "run_phl7_case.py")
    atomic_json(output / "protocol.json", protocol)
    report = {"status": "running", "execution_origin": "new_analysis_of_author_outputs",
              "archive_sha256": phl7_case.ARCHIVE_SHA256, "protocol_sha256": sha256_file(output / "protocol.json"),
              "new_molecular_dynamics_performed": False, "full_paper_reproduced": False,
              "method_discrepancies": protocol["method_discrepancies"],
              "missing_for_full_computational_reproduction": protocol["missing_for_full_computational_reproduction"],
              "allocations": []}
    atomic_json(output / "report.json", report)
    env = {"PATH": str(executable.parent) + ":/usr/bin:/bin", "LANG": "C.UTF-8",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PLUMED_NUM_THREADS": "1"}
    with zipfile.ZipFile(archive) as z:
        for allocation in protocol["allocations"]:
            folder = output / "allocations" / allocation["id"]
            folder.mkdir(parents=True)
            names = {**allocation["input_members"], "reference.dat": allocation["reference_member"]}
            for name, member in names.items():
                (folder / name).write_bytes(z.read(member))
            input_hashes = {name: sha256_file(folder / name) for name in names}
            row = {"id": allocation["id"], "status": "running"}
            report["allocations"].append(row)
            atomic_json(output / "report.json", report)
            start = time.monotonic()
            try:
                argv = [str(executable), "driver", "--noatoms", "--plumed", "reweight.dat"]
                with (folder / "driver.log").open("wb") as log:
                    result = subprocess.run(argv, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                row["exit_code"] = result.returncode
                if result.returncode:
                    raise RuntimeError("PLUMED driver failed")
                if input_hashes != {name: sha256_file(folder / name) for name in names}:
                    raise ValueError("PHL7 analysis inputs changed")
                result_name = phl7_case.ANALYSES[allocation["analysis"]]
                row["comparison"] = phl7_case.compare_fes((folder / "reference.dat").read_text(), (folder / result_name).read_text())
                row["artifact_hashes"] = {**input_hashes, result_name: sha256_file(folder / result_name), "driver.log": sha256_file(folder / "driver.log")}
                row["status"] = "completed"
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            row["seconds"] = time.monotonic() - start
            atomic_json(output / "report.json", report)
    if (sha256_file(archive) != protocol["archive_sha256"]
            or sha256_file(executable) != protocol["plumed_sha256"]
            or sha256_file(Path(phl7_case.__file__)) != protocol["module_sha256"]
            or environment != {p.name: sha256_file(p) for p in (executable.parent.parent / "conda-meta").glob("*.json")}):
        report["status"] = "source_changed"
    elif all(x["status"] == "completed" for x in report["allocations"]):
        report["status"] = "completed"
        report["evidence"] = phl7_case.verify_artifacts(report, output)
    else:
        report["status"] = "completed_with_failures"
    atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "evidence": result.get("evidence")}, indent=2))
    raise SystemExit(0 if result["status"] == "completed" else 1)
