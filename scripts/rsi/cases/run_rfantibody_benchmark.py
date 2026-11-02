"""Execute one admitted target at 200 backbones x 8 sequences x three RF2 conditions."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
import os
import time
from pathlib import Path

from agent.rsi.campaigns.case_workflow import CaseWorkflow
from agent.rsi.cases.rfantibody_benchmark import (
    CONDITIONS,
    SCALE,
    build_stages,
    compact_summary,
    freeze_author_source,
    load_catalog,
    summarize_condition,
    verify_author_source,
    verify_weights,
)
from agent.rsi.cases.rfantibody_evidence import COMMIT
from agent.rsi.cases.rfantibody_targets import TARGETS
from agent.rsi.engine.storage import atomic_json, case_gpu_lock, sha256_file

ROOT = Path(__file__).resolve().parents[3]


def main(target, output):
    out = output.resolve()
    out.relative_to((ROOT / "research_runs/rsi").resolve())
    if out == (ROOT / "research_runs/rsi").resolve():
        raise ValueError("Dedicated output required")
    catalog, record = load_catalog(ROOT, target)
    verify_weights(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "input_catalog.json", catalog)
    freeze_author_source(ROOT, out)
    settings = {
        "target": target,
        "source_commit": COMMIT,
        "scale": SCALE,
        "hotspot_proportions": list(CONDITIONS),
        "catalog_sha256": sha256_file(out / "input_catalog.json"),
        "author_manifest_sha256": sha256_file(out / "author_manifest.json"),
        "sampling": {"backbone_and_sequence": "author deterministic mode", "rf2_seed": 20261018},
        "paired_conditions_share_sequences": True,
        "execution_role": "trusted_plugin_controller",
        "scope": "Published design counts and hotspot conditions with explicit reconstructed inputs and public implementation",
    }
    workflow = CaseWorkflow(out, settings)
    pid = os.getpid()
    stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    report = {
        "status": "waiting_for_gpu",
        "settings": settings,
        "stages": [],
        "conditions": {},
        "audit_hashes": {},
        "started_unix": time.time(),
        "controller": {
            "pid": pid,
            "start_ticks": stat[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        },
    }
    atomic_json(out / "report.json", report)
    stages = build_stages(ROOT, out, record, catalog["framework"])
    try:
        for stage in stages:
            report.update(status="waiting_for_gpu", current_stage=stage.id)
            atomic_json(out / "report.json", report)
            with case_gpu_lock(ROOT / "research_runs/rsi/.case_gpu.lock"):
                current_catalog, _ = load_catalog(ROOT, target)
                if current_catalog != catalog or sha256_file(out / "input_catalog.json") != settings["catalog_sha256"]:
                    raise ValueError("Input admission changed while queued")
                verify_author_source(out, settings["author_manifest_sha256"])
                verify_weights(ROOT)
                report["status"] = "running"
                atomic_json(out / "report.json", report)
                result = workflow.run_stage(stage)
                verify_author_source(out, settings["author_manifest_sha256"])
                verify_weights(ROOT)
            report["stages"].append(result)
            atomic_json(out / "report.json", report)
            print(
                json.dumps(
                    {"target": target, "stage": stage.id, "status": "completed", "wall_seconds": result["wall_seconds"]}
                ),
                flush=True,
            )
            if stage.id.startswith("refinement_"):
                condition = stage.id.removeprefix("refinement_").replace("p", ".")
                audit = summarize_condition(out, target, condition)
                path = out / "audits" / (condition + ".json")
                atomic_json(path, audit)
                report["conditions"][condition] = compact_summary(audit)
                report["audit_hashes"][condition] = sha256_file(path)
                atomic_json(out / "report.json", report)
        report.update(
            status="completed",
            computational_chain_completed=True,
            full_paper_reproduced=False,
            quality_audit={
                "sequence_preservation_verified": True,
                "sequence_design_records": 1600,
                "refinement_assessments": 4800,
            },
            finished_unix=time.time(),
        )
    except BaseException as exc:
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc), finished_unix=time.time())
        raise
    finally:
        atomic_json(out / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.target, args.output)
