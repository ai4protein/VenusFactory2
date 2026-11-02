"""Typed ALDE plugin: new author computation or independently verified reuse."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from agent.rsi.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
EXISTING = {
    "author_ts70_20261011": "20261011_alde_author_ts70_v1",
    "author_three_acquisition_pilot_20261011": "20261011_alde_author_onehot_dnn_v2",
}


def read(path):
    return json.loads(path.read_text())


def invoke(argv, out, name):
    env = {
        "PATH": str(Path(argv[0]).parent) + ":/usr/bin:/bin",
        "HOME": str(out / "home"),
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    Path(env["HOME"]).mkdir(exist_ok=True)
    with (out / (name + ".log")).open("wb") as stream:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(name + " failed; evidence retained")


def main(args):
    out = args.output.resolve()
    out.relative_to(ROOT / "research_runs/rsi")
    if out == ROOT / "research_runs/rsi":
        raise ValueError("Dedicated output required")
    out.mkdir(parents=True, exist_ok=False)
    python = str(ROOT / "research_runs/rsi/envs/alde/bin/python")
    report = {
        "status": "running",
        "plugin_id": "alde.onehot_dnn.v1",
        "operation": args.operation,
        "requested_campaign": args.campaign,
        "execution_origin": "new_computation" if args.operation == "reproduce" else "verified_existing",
        "full_paper_reproduced": False,
        "wet_lab_validation": False,
        "query_budget": {"initial": 96, "additional_rounds": 4, "batch_size": 96, "total": 480},
        "scope": "complete author one-hot/DNN trajectories for declared proteins, acquisition rules and seeds",
        "missing_for_complete_paper": ["other model and encoding families", "new wet-lab validation"],
    }
    atomic_json(out / "report.json", report)
    try:
        if args.operation == "reproduce":
            if args.campaign or not 1 <= args.seeds <= 70 or len(set(args.acquisitions)) != len(args.acquisitions):
                raise ValueError("Invalid reproduction allocation")
            campaign = out / "campaign"
            invoke(
                [
                    python,
                    "-u",
                    str(ROOT / "scripts/rsi/run_alde_author_reproduction.py"),
                    "--output",
                    str(campaign),
                    "--seeds",
                    str(args.seeds),
                    "--acquisitions",
                    *args.acquisitions,
                ],
                out,
                "author_run",
            )
        else:
            if args.campaign not in EXISTING:
                raise ValueError("Existing campaign must be host registered")
            campaign = ROOT / "research_runs/rsi" / EXISTING[args.campaign]
        source_report, protocol_path = campaign / "report.json", campaign / "protocol.json"
        before = {str(p): sha256_file(p) for p in [source_report, protocol_path]}
        protocol = read(protocol_path)
        if sha256_file(campaign / "implementation.py") != protocol["runner_sha256"]:
            raise ValueError("Archived author runner changed")
        source = ROOT / "research_runs/rsi/case_assets/ALDE" / protocol["commit"] / "source"
        for name, sha in protocol["source_hashes"].items():
            if sha256_file(source / name) != sha:
                raise ValueError("Author source changed")
        for protein, inputs in protocol["inputs"].items():
            if protein not in {"GB1", "TrpB"}:
                raise ValueError("Unregistered ALDE landscape")
            for folder, suffix, field in [
                ("alde_landscapes", "_fitness.csv", "fitness_sha256"),
                ("alde_onehot", "_onehot_x.pt", "encoding_sha256"),
            ]:
                if sha256_file(ROOT / "research_runs/rsi/case_assets" / folder / (protein + suffix)) != inputs[field]:
                    raise ValueError("ALDE data or encoding changed")
        report.update(source_campaign=str(campaign.relative_to(ROOT / "research_runs/rsi")), source_hashes=before)
        atomic_json(out / "report.json", report)
        invoke(
            [
                python,
                "-u",
                str(ROOT / "scripts/rsi/audit_alde_reproduction.py"),
                "--run",
                str(campaign),
                "--output",
                str(out / "audit"),
            ],
            out,
            "audit",
        )
        if {str(p): sha256_file(p) for p in [source_report, protocol_path]} != before:
            raise ValueError("Campaign changed during audit")
        audit, original = read(out / "audit/report.json"), read(source_report)
        expected = len(protocol["allocations"])
        if (
            original["status"] != "campaign_completed"
            or audit["status"] != "completed"
            or audit["allocations"] != expected
            or audit["completed_and_verified"] != expected
            or len(original["arms"]) != expected
        ):
            raise ValueError("Incomplete author trajectories")
        for actual, declared in zip(original["arms"], protocol["allocations"], strict=True):
            if actual["status"] != "completed" or any(actual[k] != v for k, v in declared.items()):
                raise ValueError("Allocation identity changed")
        report.update(
            status="completed",
            source_repository=protocol["author_repository"],
            source_commit=protocol["commit"],
            source_agent_orchestrated=protocol["agent_orchestrated"],
            allocations=expected,
            completed_and_verified=audit["completed_and_verified"],
            seeds=len(protocol["seeds"]),
            proteins=sorted(protocol["inputs"]),
            acquisitions=sorted({a["acquisition"] for a in protocol["allocations"]}),
            audit_report_sha256=sha256_file(out / "audit/report.json"),
            summaries=audit["summaries"],
            reused_results_are_new_replicates=False,
            provenance_note="Agent request provenance belongs to the outer orchestrator; original source provenance is retained",
        )
        atomic_json(out / "report.json", report)
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        atomic_json(out / "report.json", report)
        raise
    print(
        json.dumps({k: report[k] for k in ["status", "execution_origin", "allocations", "completed_and_verified"]}),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--operation", choices=["reproduce", "verify_existing"], required=True)
    p.add_argument("--campaign", choices=sorted(EXISTING))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--acquisitions", nargs="+", choices=["GREEDY", "UCB", "TS"], default=["TS"])
    main(p.parse_args())
