"""Fresh NucB author raw-count reconstruction and native-verified analysis."""

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
from pathlib import Path

from agent.rsi import nucb_reproduction as nucb
from agent.rsi.engine.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]


def run(output):
    output = output.absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Symlinked output is not admitted")
    output = output.resolve()
    output.relative_to(ROOT / "research_runs/rsi")
    if output.exists() or output == ROOT / "research_runs/rsi":
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {
        "status": "running",
        "execution_origin": "new_raw_count_and_analysis_computation",
        "source_commit": nucb.COMMIT,
        "limits": list(nucb.LIMITS),
        "full_paper_reproduced": False,
        "new_model_developed": False,
        "wet_lab_validation": False,
        "stages": [],
    }
    atomic_json(output / "report.json", report)
    try:
        nucb.admit(ROOT)
        protocol = {
            "source_manifest_sha256": nucb.SOURCE_MANIFEST_SHA256,
            "asset_receipt_sha256": nucb.ASSET_RECEIPT_SHA256,
            "implementation": nucb.implementation(ROOT),
            "limits": list(nucb.LIMITS),
            "allocation": "all four generations, all 16 processed/fiducial tables, landscape and library/plate analysis",
        }
        (output / "implementation").mkdir()
        for name in protocol["implementation"]:
            path = ROOT / ("src/agent/rsi/cases" if name == "nucb_reproduction.py" else "scripts/rsi/cases") / name
            shutil.copy2(path, output / "implementation" / name)
        atomic_json(output / "protocol.json", protocol)
        report["protocol_sha256"] = sha256_file(output / "protocol.json")
        python = ROOT / "research_runs/rsi/envs/nuclease/bin/python"
        common = ["--assets", str(nucb.assets_at(ROOT)), "--asset-report", str(ROOT / nucb.ASSET_RECEIPT)]
        for stage, script, extra, timeout in [
            ("raw", "run_nucb_author_reproduction.py", [], 10800),
            ("analysis", "run_nucb_analysis_reproduction.py", ["--data-root", str(output / "raw/data")], 600),
        ]:
            row = {"stage": stage, "status": "running"}
            report["stages"].append(row)
            atomic_json(output / "report.json", report)
            argv = [
                str(python),
                "-u",
                str(ROOT / "scripts/rsi" / script),
                *common,
                *extra,
                "--output",
                str(output / stage),
            ]
            with (output / (stage + ".log")).open("wb") as log:
                result = subprocess.run(
                    argv, env=nucb.clean_environment(ROOT), stdout=log, stderr=subprocess.STDOUT, timeout=timeout
                )
            row["exit_code"] = result.returncode
            row["status"] = "completed" if result.returncode == 0 else "failed"
            if result.returncode:
                raise RuntimeError(f"NucB {stage} failed; logs and partial artifacts retained")
            report[stage + "_report_sha256"] = sha256_file(output / stage / "report.json")
        report["status"] = "completed"
        report["evidence"] = nucb.verify_artifacts(report, output, ROOT)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({"status": result["status"], "evidence": result["evidence"]}))
