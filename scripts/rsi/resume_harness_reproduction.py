"""Resume evidence review after a terminal long-job interruption, without resubmission."""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import replace
from pathlib import Path

from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_harness_reproduction import archive_implementation, bundle
from run_protein_lifecycle import audit_events

from agent.harness import Profile, Task
from agent.harness.contracts import digest, file_hash
from agent.harness.host import execute, write_json
from agent.rsi.engine.case_evidence import render_evidence_report, validate_review
from agent.rsi.engine.reproduction_recovery import RESOURCES, RecoveryLedger, equal, load_source, phase_evidence, read
from agent.rsi.engine.source_snapshot import research_implementation_paths


async def main(args, *, credential=None, continuation=None, ceiling=None, goal_context=""):
    base = (ROOT / "research_runs/rsi").resolve()
    source_run, output = args.source_run.resolve(), args.output.resolve()
    source_run.relative_to(base)
    output.relative_to(base)
    if output == source_run or source_run in output.parents:
        raise ValueError("Recovery must leave the source run untouched")
    source = load_source(source_run)
    profile = Profile.from_dict(source["protocol"]["profile"])
    key = credential if credential is not None else credential_from_toml(args.credential_toml, profile.model)
    if (continuation is None) != (ceiling is None):
        raise ValueError("A goal continuation requires a bounded reservation and receipt")
    trigger = "persistent_goal" if continuation is not None else "operator"
    output.mkdir(parents=True, exist_ok=False)
    paths = research_implementation_paths(ROOT)
    hashes = archive_implementation(paths, output)
    ledger_root = (
        Path(continuation["goal_root"]) / "review_recovery_ledger"
        if continuation is not None
        else base / "reproduction_recovery_ledger"
    )
    ledger = RecoveryLedger(ledger_root, source)
    reservation = ledger.reserve(output, ceiling=ceiling)
    recovery_profile = replace(profile, budget=replace(profile.budget, **reservation["grant"]))
    protocol = {
        "kind": "interrupted_reproduction_review_recovery",
        "source_run": str(source_run),
        "source_hashes": source["source_hashes"],
        "original_status": source["original_status"],
        "original_profile": profile.public(),
        "profile": recovery_profile.public(),
        "ledger": str(ledger.path),
        "reservation": reservation,
        "plugin": source["job"]["plugin_id"],
        "job_id": source["job"]["job_id"],
        "submission_witnesses": source["submission_witnesses"],
        "implementation": hashes,
        "new_scientific_jobs_allowed": 0,
        **({"continuation": continuation, "goal_context_sha256": digest(goal_context)} if continuation else {}),
        "claim_scope": trigger
        + " recovery of an interrupted agent-submitted computation; structured evidence review only, not an uninterrupted run or independent efficacy evidence",
    }
    write_json(output / "protocol.json", protocol)
    report = {
        "status": "running",
        "protocol_sha256": digest(protocol),
        "original_status": source["original_status"],
        "original_run_unchanged": True,
        "agent_submission_verified": True,
        "recovered_review_verified": False,
        "end_to_end_verified": False,
        "uninterrupted_end_to_end_verified": False,
        "recovery_trigger": trigger,
        "operator_recovery_interventions": int(trigger == "operator"),
        "new_computation_in_recovery": False,
        "source_computation_origin": source["domain"].get("execution_origin", "unspecified"),
    }
    write_json(output / "report.json", report)
    started = time.monotonic()
    facts = source["facts"]
    source_sha = source["job"]["report_sha256"]
    versions, version, cap = bundle(
        output / "review_capability",
        [
            ("src/agent/rsi/engine/case_review_tools.py", "generated_tools.py"),
            ("src/agent/rsi/engine/case_evidence.py", "agent/rsi/engine/case_evidence.py"),
            ("src/agent/rsi/engine/protein_model.py", "agent/rsi/engine/protein_model.py"),
            ("src/agent/rsi/cases/mcba_evidence.py", "agent/rsi/cases/mcba_evidence.py"),
            ("src/agent/rsi/cases/rfantibody_evidence.py", "agent/rsi/cases/rfantibody_evidence.py"),
            ("src/agent/rsi/engine/storage.py", "agent/rsi/engine/storage.py"),
        ],
        {"plugin": source["job"]["plugin_id"], "source_report_sha256": source_sha},
        {
            "job_id": source["job"]["job_id"],
            "plugin": source["job"]["plugin_id"],
            "report_sha256": source_sha,
            "facts": facts,
        },
        episode={
            "context_tool": "inspect_case_evidence",
            "artifact_file": "review.json",
            "schema": 2,
            "required_tools": ["inspect_case_evidence", "submit_evidence_review"],
            "completion_repair_limit": 1,
        },
    )
    work = output / "review_workspace"
    work.mkdir()
    task = Task(
        work,
        cap,
        output / "review",
        goal_context
        + "\n\nA previous session submitted the scientific job successfully but its conversation was interrupted. The computation is now completed. Inspect its source-bound evidence and submit exactly those facts via submit_evidence_review, including every discrepancy and scope limitation. Correct a rejected submission at most once. Report the accepted file and explicitly state that this is recovery review, not a new computation or an uninterrupted successful run.",
        "Use only the two evidence tools. Do not infer biological efficacy or full-paper reproduction. No resubmission of the scientific job is available.",
        frozenset({"inspect_case_evidence", "submit_evidence_review"}),
        phase="chat",
    )
    runtime = await execute(recovery_profile, task, default_runtimes(), credential=key)
    scientific_audit = audit_events(task.output, "review", runtime, set(task.tools), {"submit_evidence_review"})
    report["review"] = {"run": runtime, "scientific_audit": scientific_audit, "source_report_sha256": source_sha}
    versions.verify(version)
    evidence = phase_evidence(output, "review", profile.public())
    report["spent"] = evidence["spent"]
    report["cumulative_spent"] = {k: reservation["spent_before"][k] + evidence["spent"][k] for k in RESOURCES}
    report["fresh_session"] = runtime["session_id"] not in source["sessions"]
    report["status"] = "review_failed"
    if (
        runtime["status"] == "completed"
        and runtime["audit"]["passed"]
        and scientific_audit["passed"]
        and report["fresh_session"]
    ):
        accepted = validate_review(read(work / "review.json")["claims"], facts)
        report["review"]["verified"] = accepted
        report.update(status="recovered_review", recovered_review_verified=True)
        rendered = render_evidence_report(source["job"]["job_id"], facts)
        (output / "verified_evidence.md").write_text(
            "Recovery of an interrupted run. Original failure retained; no new scientific job.\n\n" + rendered
        )
    try:
        equal(load_source(source_run)["source_hashes"], source["source_hashes"], "Source changed during recovery")
    except (ValueError, OSError):
        report.update(status="source_changed", original_run_unchanged=False, recovered_review_verified=False)
    report["implementation_unchanged"] = all(file_hash(ROOT / n) == sha for n, sha in hashes.items())
    if not report["implementation_unchanged"]:
        report.update(status="implementation_changed", recovered_review_verified=False)
    if any(report["cumulative_spent"][k] > getattr(profile.budget, k) for k in RESOURCES):
        report.update(status="budget_exceeded", recovered_review_verified=False)
    report["recovery_host_wall_seconds"] = round(time.monotonic() - started, 3)
    write_json(output / "report.json", report)
    ledger.settle(reservation["index"])
    print(
        {
            k: report[k]
            for k in [
                "status",
                "cumulative_spent",
                "fresh_session",
                "original_run_unchanged",
                "new_computation_in_recovery",
            ]
        },
        flush=True,
    )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-toml", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
