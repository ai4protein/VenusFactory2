"""Read-only request/job/review audit, with independent ALDE trajectory replay."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.rsi.evaluation.audit_generated_campaign import read, require_equal, verify_snapshot

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.host import write_json
from agent.rsi.engine.case_evidence import canonical_claims, validate_review

ROOT = Path(__file__).resolve().parents[3]


def main(args):
    run = args.run.resolve()
    report, protocol = read(run / "report.json"), read(run / "protocol.json")
    if report["status"] != "completed" or not report["end_to_end_verified"]:
        raise ValueError("Audit requires terminal verified reproduction")
    require_equal(digest(protocol), report["protocol_sha256"], "Protocol changed")
    for name, sha in protocol["implementation"].items():
        require_equal(file_hash(run / "implementation" / name), sha, "Archived implementation changed")
    args.output.mkdir(parents=True, exist_ok=False)
    phases, sessions = {}, []
    totals = {"model_calls": 0, "tool_calls": 0}
    for phase in ["launch", "review"]:
        execution = run / phase
        runtime, lock = read(execution / "run.json"), read(execution / "profile.lock.json")
        require_equal(runtime, report[phase]["run"], "Runtime report changed")
        require_equal(lock["sha256"], digest({k: v for k, v in lock.items() if k != "sha256"}), "Profile lock changed")
        require_equal(runtime["profile_sha256"], lock["sha256"], "Profile binding changed")
        require_equal(lock["profile"]["model"], protocol["profile"]["model"], "Model changed across turns")
        caps_dirs = list((run / (phase + "_capability") / "bundles/versions").iterdir())
        if len(caps_dirs) != 1:
            raise ValueError("Ambiguous capability")
        caps = caps_dirs[0]
        verify_snapshot(caps)
        work = run / ("workspace" if phase == "launch" else "review_workspace")
        for name, sha in lock["task"]["inputs"].items():
            require_equal(file_hash(work / name), sha, "Input changed")
        for name, sha in lock["task"]["capabilities"].items():
            require_equal(file_hash(caps / name), sha, "Capability changed")
        events = audit_trace(execution / "events.jsonl", lock["task"]["tools"], lock["profile"]["budget"]["tool_calls"])
        requests = audit_requests(execution / "model_requests.jsonl")
        if not (events["passed"] and requests["passed"] and report[phase]["scientific_audit"]["passed"]):
            raise ValueError("Incomplete execution evidence")
        for key, value in [("model_calls", requests["model_calls"]), ("tool_calls", events["tool_calls"])]:
            totals[key] += value
        phases[phase] = {
            "model_calls": requests["model_calls"],
            "tool_calls": events["tool_calls"],
            "evidence_verified": True,
        }
        sessions.append(runtime["session_id"])
    if sessions[0] == sessions[1] or any(totals[k] > protocol["profile"]["budget"][k] for k in totals):
        raise ValueError("Fresh-session or shared budget contract failed")
    jobs = read(run / "controller/jobs.json")["jobs"]
    if len(jobs) != 1:
        raise ValueError("Unexpected job count")
    job = next(iter(jobs.values()))
    require_equal(job["plugin_id"], protocol["plugin"], "Plugin changed")
    require_equal(job["parameters"], protocol["parameters"], "Allocation changed")
    if job["status"] != "completed":
        raise ValueError("Job incomplete")
    folder = Path(job["output"]).resolve()
    folder.relative_to(run)
    raw = read(folder / "report.json")
    require_equal(file_hash(folder / "report.json"), job["report_sha256"], "Domain report changed")
    require_equal(job["report_sha256"], report["review"]["source_report_sha256"], "Reviewed source changed")
    claims = read(run / "review_workspace/review.json")["claims"]
    expected = canonical_claims(protocol["plugin"], raw)
    require_equal(validate_review(claims, expected), report["review"]["verified"], "Review changed")
    replayed = False
    notebook_replayed = False
    if protocol["plugin"] == "alde.onehot_dnn.v1":
        require_equal(report["new_computation"], raw["execution_origin"] == "new_computation", "Provenance relabelled")
        require_equal(file_hash(folder / "audit/report.json"), raw["audit_report_sha256"], "Domain audit changed")
        campaign = (ROOT / "research_runs/rsi" / raw["source_campaign"]).resolve()
        campaign.relative_to(ROOT / "research_runs/rsi")
        for name in ["report.json", "protocol.json"]:
            path = campaign / name
            require_equal(file_hash(path), raw["source_hashes"][str(path)], "Source campaign changed")
        python = ROOT / "research_runs/rsi/envs/alde/bin/python"
        with (args.output / "trajectory_replay.log").open("wb") as log:
            subprocess.run(
                [
                    str(python),
                    str(ROOT / "scripts/rsi/cases/audit_alde_reproduction.py"),
                    "--run",
                    str(campaign),
                    "--output",
                    str(args.output / "trajectory_replay"),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        replay = read(args.output / "trajectory_replay/report.json")
        require_equal(replay["summaries"], raw["summaries"], "Independent round/seed summary replay differs")
        require_equal(
            replay["completed_and_verified"], raw["completed_and_verified"], "Trajectory completeness differs"
        )
        replayed = True

    elif protocol["plugin"] == "mcba.multisubstrate.v1":
        from agent.rsi.cases.mcba_evidence import summarize_campaign

        require_equal(report["new_computation"], raw["execution_origin"] == "new_computation", "Provenance relabelled")
        require_equal(file_hash(folder / "audit/audit.json"), raw["audit_report_sha256"], "Domain audit changed")
        base = (ROOT / "research_runs/rsi").resolve()
        campaign = (base / raw["source_campaign"]).resolve()
        campaign.relative_to(base)
        for name, expected_hash in raw["source_hashes"].items():
            require_equal(file_hash(campaign / name), expected_hash, "Source campaign changed")
        with (args.output / "notebook_replay.log").open("wb") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/rsi/cases/audit_mcba_multisubstrate.py"),
                    "--campaign",
                    str(campaign),
                    "--source",
                    str(base / "case_assets/mcba_multisubstrate_source"),
                    "--output",
                    str(args.output / "notebook_replay"),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
        replay = read(args.output / "notebook_replay/audit.json")
        require_equal(
            summarize_campaign(read(campaign / "report.json"), replay),
            raw["summary"],
            "Independent notebook summary replay differs",
        )
        notebook_replayed = True
    antibody_verified = None
    if protocol["plugin"] == "rfantibody.targets.v1":
        from agent.rsi.cases.rfantibody_evidence import validate_target_report, verify_target_artifacts

        validate_target_report(raw, protocol["parameters"])
        antibody_verified = verify_target_artifacts(raw, folder)
        limits = protocol.get("design_limits", {"maximum_backbones": 8, "maximum_sequences_per_backbone": 4})
        if (
            raw["settings"]["scale"]["backbones"] > limits["maximum_backbones"]
            or raw["settings"]["scale"]["sequences_per_backbone"] > limits["maximum_sequences_per_backbone"]
        ):
            raise ValueError("Antibody allocation exceeded the predeclared limits")
    result = {
        "status": "audit_completed",
        "run": str(run),
        "run_report_sha256": file_hash(run / "report.json"),
        "auditor_sha256": file_hash(Path(__file__)),
        "phases": phases,
        "job_parameters_verified": True,
        "source_bound_review_verified": True,
        "independent_trajectory_replay": replayed,
        "independent_notebook_replay": notebook_replayed,
        "antibody_artifact_verification": antibody_verified,
        "new_computation": report["new_computation"],
        "scope": "Integrity and trajectory-statistic replay, not new training or an efficacy replicate",
    }
    write_json(args.output / "report.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
