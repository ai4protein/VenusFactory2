"""Agent-callable full CataPro reproduction or independently audited reuse."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from agent.rsi.catapro_evidence import (
    COMMIT,
    COUNTS,
    EXISTING,
    REGISTERED_PROTOCOL_SHA,
    audit_campaign,
    canonical_catapro,
    verify_artifacts,
)
from agent.rsi.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "research_runs/rsi/20261017_catapro_runtime_setup_v1"
SOURCE = ROOT / "research_runs/rsi/case_assets/CataPro" / COMMIT


def read(path):
    return json.loads(path.read_text())


def reproduce(campaign, output):
    campaign.mkdir()
    paths = [ROOT / "scripts/rsi/run_catapro_case.py", ROOT / "src/agent/rsi/catapro_case.py"]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    for p in paths:
        dest = campaign / "repository" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    protocol = {
        "source_sha256": hashes,
        "cohorts": [{"name": n, "expected_rows": c} for n, c in COUNTS.items()],
        "max_seconds_per_cohort": 21600,
        "feature_cache": "New cache within this job; pinned pretrained encoders are reused",
        "scope": "Full original-fold checkpoint replay; original training not included",
    }
    atomic_json(campaign / "protocol.json", protocol)
    execution = {"status": "running", "started_unix": time.time(), "cohorts": []}
    atomic_json(campaign / "execution.json", execution)
    inherited = os.environ.get("VENUS_RSI_CASE_LOCK_FD")
    pass_fds = (int(inherited),) if inherited is not None else ()
    for name, expected in COUNTS.items():
        if any(sha256_file(ROOT / n) != h for n, h in hashes.items()):
            raise ValueError("Case implementation changed")
        row = {"name": name, "expected_rows": expected, "status": "running"}
        execution["cohorts"].append(row)
        atomic_json(campaign / "execution.json", execution)
        command = [
            str(RUNTIME / "venv/bin/python"),
            "-u",
            str(ROOT / "scripts/rsi/run_catapro_case.py"),
            "--source",
            str(SOURCE),
            "--protein-encoder",
            str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"),
            "--molecule-encoder",
            str(SOURCE / "encoders/molt5-base-smiles2caption/7b7d4b0ab8b66b351e669b1f66272418ba15c3d9"),
            "--cache",
            str(campaign / "features.sqlite"),
            "--output",
            str(campaign / name),
            "--dataset",
            name,
            "--device",
            "cuda:0",
        ]
        with (output / (name + ".log")).open("xb") as log:
            rc = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=21600,
                pass_fds=pass_fds,
            ).returncode
        report = read(campaign / name / "report.json")
        row.update(status=report["status"], returncode=rc, report_sha256=sha256_file(campaign / name / "report.json"))
        atomic_json(campaign / "execution.json", execution)
        if rc or report["status"] != "completed" or report.get("rows_encoded") != expected:
            raise ValueError("Full cohort failed or incomplete")
    if any(sha256_file(ROOT / n) != h for n, h in hashes.items()):
        raise ValueError("Case implementation changed during execution")
    execution.update(status="completed", rows=sum(COUNTS.values()), finished_unix=time.time())
    atomic_json(campaign / "execution.json", execution)


def main(args):
    output = args.output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    if args.operation == "reproduce" and args.campaign is not None:
        raise ValueError("New computation cannot select old evidence")
    if args.operation == "verify_existing" and args.campaign not in EXISTING:
        raise ValueError("Choose a registered full campaign")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "operation": args.operation,
        "requested_campaign": args.campaign,
        "execution_origin": "new_computation" if args.operation == "reproduce" else "verified_existing",
        "source_commit": COMMIT,
        "new_training": False,
        "full_paper_reproduced": False,
        "wet_lab_validation": False,
    }
    atomic_json(output / "report.json", report)
    try:
        if args.operation == "reproduce":
            campaign = output / "campaign"
            reproduce(campaign, output)
        else:
            campaign = ROOT / EXISTING[args.campaign]
            if sha256_file(campaign / "protocol.json") != REGISTERED_PROTOCOL_SHA:
                raise ValueError("Registered source campaign protocol changed")
        audit = audit_campaign(campaign, SOURCE)
        atomic_json(output / "audit.json", audit)
        report.update(
            status="completed",
            source_campaign=str(campaign.relative_to(ROOT)),
            audit_evidence=audit,
            audit_sha256=sha256_file(output / "audit.json"),
            scope="All 95,507 released enzyme-substrate rows; 30 author CV checkpoints. Original training, autonomous model invention, unbiased held-out efficacy and wet-lab replication excluded.",
        )
        canonical_catapro(report)
        verify_artifacts(report, output, SOURCE)
        atomic_json(output / "report.json", report)
        print(
            json.dumps(
                {"status": "completed", "execution_origin": report["execution_origin"], "rows": audit["total_rows"]}
            ),
            flush=True,
        )
    except Exception as exc:
        partial = output / "campaign/execution.json"
        if partial.is_file():
            execution = read(partial)
            if execution.get("status") != "completed":
                execution.update(status="failed", error_type=type(exc).__name__, finished_unix=time.time())
                atomic_json(partial, execution)
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:300])
        atomic_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation", choices=["reproduce", "verify_existing"], required=True)
    parser.add_argument("--campaign", choices=sorted(EXISTING))
    main(parser.parse_args())
