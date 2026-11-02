"""Typed complete McbA notebook computation or source-verified existing evidence."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
import subprocess
import sys
from pathlib import Path

from agent.rsi.cases.mcba_evidence import COMMIT, summarize_campaign
from agent.rsi.engine.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
EXISTING = {"author_multisubstrate_20261012": "20261012_mcba_multisubstrate_author_v1"}


def read(path):
    return json.loads(path.read_text())


def invoke(argv, out, name):
    env = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "HOME": str(out / "home"),
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT / "src"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    Path(env["HOME"]).mkdir(exist_ok=True)
    with (out / (name + ".log")).open("wb") as log:
        completed = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(name + " failed; original evidence retained")


def main(args):
    root = (ROOT / "research_runs/rsi").resolve()
    out = args.output.resolve()
    out.relative_to(root)
    if out == root:
        raise ValueError("Dedicated output required")
    if (args.operation == "reproduce" and args.campaign is not None) or (
        args.operation == "verify_existing" and args.campaign not in EXISTING
    ):
        raise ValueError("Unregistered or conflicting operation")
    out.mkdir(parents=True, exist_ok=False)
    source = root / "case_assets/mcba_multisubstrate_source"
    report = {
        "status": "running",
        "plugin_id": "mcba.multisubstrate.v1",
        "operation": args.operation,
        "requested_campaign": args.campaign,
        "execution_origin": "new_computation" if args.operation == "reproduce" else "verified_existing",
        "full_paper_reproduced": False,
        "wet_lab_validation": False,
        "upstream_zero_shot_regenerated": False,
    }
    atomic_json(out / "report.json", report)
    try:
        if args.operation == "reproduce":
            campaign = out / "campaign"
            invoke(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/rsi/cases/run_mcba_multisubstrate.py"),
                    "--source",
                    str(source),
                    "--output",
                    str(campaign),
                    "--execute",
                ],
                out,
                "author_run",
            )
        else:
            campaign = root / EXISTING[args.campaign]
        if read(campaign / "report.json").get("status") != "campaign_completed":
            raise ValueError("Source campaign has not terminated")
        protocol = read(campaign / "protocol.json")
        if protocol["commit"] != COMMIT:
            raise ValueError("Source version differs from the registered plugin")
        before = {
            name: sha256_file(campaign / name)
            for name in ["protocol.json", "report.json", "source_manifest.json", "runner.py"]
        }
        invoke(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts/rsi/cases/audit_mcba_multisubstrate.py"),
                "--campaign",
                str(campaign),
                "--source",
                str(source),
                "--output",
                str(out / "audit"),
            ],
            out,
            "audit",
        )
        if before != {name: sha256_file(campaign / name) for name in before}:
            raise ValueError("Source campaign changed during audit")
        original, audit = read(campaign / "report.json"), read(out / "audit/audit.json")
        summary = summarize_campaign(original, audit)
        report.update(
            status="completed",
            source_campaign=str(campaign.relative_to(root)),
            source_commit=COMMIT,
            source_hashes=before,
            source_repository="https://github.com/grantlandwehr/accelerated-enzyme-engineering",
            summary=summary,
            campaign_evidence=original,
            audit_evidence=audit,
            audit_report_sha256=sha256_file(out / "audit/audit.json"),
            scope="All unique released processed-input computational tasks in the two author notebooks; numerical/reference agreement is reported separately from completed execution",
        )
        atomic_json(out / "report.json", report)
        print(
            json.dumps({"status": "completed", "execution_origin": report["execution_origin"], **summary}), flush=True
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        atomic_json(out / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation", choices=["reproduce", "verify_existing"], required=True)
    parser.add_argument("--campaign", choices=sorted(EXISTING))
    main(parser.parse_args())
