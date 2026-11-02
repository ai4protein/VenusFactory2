"""Agent-requested fifty-fit CataPro training or read-only verified reuse."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import contextlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from agent.rsi.cases.catapro_training_evidence import (
    COMMIT,
    EXISTING,
    FEATURE_ROOT,
    REGISTERED_PROTOCOL_SHA,
    audit_training_campaign,
    canonical_training,
    verify_artifacts,
)
from agent.rsi.engine.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "research_runs/rsi/20261017_catapro_runtime_setup_v1/venv/bin/python"
SOURCE = ROOT / "research_runs/rsi/case_assets/CataPro" / COMMIT
REVIEW = ROOT / "research_runs/rsi/20261017_catapro_training_protocol_v1/pdf_review.json"


def reproduce(campaign, output):
    # Run the controller in the broker's direct child. A separate wrapper
    # could exit before its controller reaps a training process group.
    with (output / "prepare.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        entry = runpy.run_path(str(ROOT / "scripts/rsi/cases/run_catapro_training.py"))
        entry["prepare"](
            SimpleNamespace(
                output=campaign,
                source=SOURCE,
                feature_root=ROOT / FEATURE_ROOT,
                pdf_review=REVIEW,
                epochs=150,
                seed=20261017,
                stage_timeout=43200,
                gpu_lock=ROOT / "research_runs/rsi/.case_gpu.lock",
            )
        )
    with (output / "training.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        frozen = runpy.run_path(str(campaign / "repository/scripts/rsi/cases/run_catapro_training.py"))
        frozen["run"](campaign)


def main(args):
    output = args.output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    if args.operation == "reproduce" and args.campaign is not None:
        raise ValueError("New training cannot select old evidence")
    if args.operation == "verify_existing" and args.campaign not in EXISTING:
        raise ValueError("Choose a registered training campaign")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "operation": args.operation,
        "requested_campaign": args.campaign,
        "execution_origin": "new_computation" if args.operation == "reproduce" else "verified_existing",
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
                raise ValueError("Registered training protocol changed")
        audit = audit_training_campaign(campaign, SOURCE, ROOT / FEATURE_ROOT, output / "replay")
        atomic_json(output / "audit.json", audit)
        report.update(
            status="completed",
            source_campaign=str(campaign.relative_to(ROOT)),
            audit_evidence=audit,
            audit_sha256=sha256_file(output / "audit.json"),
        )
        canonical_training(report)
        verify_artifacts(report, output, ROOT)
        atomic_json(output / "report.json", report)
        print(
            json.dumps({"status": "completed", "origin": report["execution_origin"], "training_fits": 50}), flush=True
        )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:400])
        atomic_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation", choices=["reproduce", "verify_existing"], required=True)
    parser.add_argument("--campaign", choices=sorted(EXISTING))
    main(parser.parse_args())
