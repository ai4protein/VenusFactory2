"""Independent trace/source audit and numerical replay of a recovered case review."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

from run_harness_probe import ROOT
from run_protein_lifecycle import audit_events

from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.engine.case_evidence import validate_review
from agent.rsi.engine.reproduction_recovery import RESOURCES, equal, inside, load_source, phase_evidence, read


def verify_recovery(run):
    run = Path(run).resolve()
    protocol, report = read(run / "protocol.json"), read(run / "report.json")
    if report["status"] != "recovered_review" or not report["recovered_review_verified"]:
        raise ValueError("No completed recovery review to audit")
    equal(digest(protocol), report["protocol_sha256"], "Recovery protocol changed")
    for name, sha in protocol["implementation"].items():
        equal(file_hash(inside(run / "implementation", name)), sha, "Recovery source archive changed")
    source = load_source(Path(protocol["source_run"]))
    equal(source["source_hashes"], protocol["source_hashes"], "Original source changed")
    equal(source["submission_witnesses"], protocol["submission_witnesses"], "Submission attribution changed")
    equal(source["protocol"]["profile"], protocol["original_profile"], "Original profile changed")
    phase = phase_evidence(run, "review", protocol["original_profile"])
    equal(phase["lock"]["profile"], protocol["profile"], "Recovery profile differs")
    equal(phase["runtime"], report["review"]["run"], "Recovery runtime changed")
    tools = {"inspect_case_evidence", "submit_evidence_review"}
    if set(phase["lock"]["task"]["tools"]) != tools:
        raise ValueError("Recovery exposed computation tools")
    audit = audit_events(run / "review", "review", phase["runtime"], tools, {"submit_evidence_review"})
    equal(audit, report["review"]["scientific_audit"], "Scientific trace audit changed")
    if not (audit["passed"] and phase["requests"]["passed"] and phase["runtime"]["status"] == "completed"):
        raise ValueError("Recovery execution incomplete")
    if phase["runtime"]["session_id"] in source["sessions"]:
        raise ValueError("Recovery did not use a fresh session")
    equal(report["review"]["source_report_sha256"], source["job"]["report_sha256"], "Reviewed domain report differs")
    versions = list((run / "review_capability/bundles/versions").iterdir())
    bundle = read(versions[0] / "evidence.json")
    equal(
        bundle,
        {
            "job_id": source["job"]["job_id"],
            "plugin": source["job"]["plugin_id"],
            "report_sha256": source["job"]["report_sha256"],
            "facts": source["facts"],
        },
        "Review capability contains different evidence",
    )
    accepted = validate_review(read(run / "review_workspace/review.json")["claims"], source["facts"])
    equal(accepted, report["review"]["verified"], "Accepted review changed")
    for flag in ["end_to_end_verified", "uninterrupted_end_to_end_verified", "new_computation_in_recovery"]:
        equal(report[flag], False, "Recovery was relabelled: " + flag)
    trigger = "persistent_goal" if "continuation" in protocol else "operator"
    equal(report["recovery_trigger"], trigger, "Recovery intervention was hidden")
    equal(report["operator_recovery_interventions"], int(trigger == "operator"), "Recovery intervention was hidden")
    ledger = read(Path(protocol["ledger"]))
    equal(ledger["source_hashes"], source["source_hashes"], "Ledger source differs")
    spent = dict(source["spent"])
    index = protocol["reservation"]["index"]
    for i, row in enumerate(ledger["attempts"][: index + 1]):
        if row["index"] != i or row["state"] != "settled":
            raise ValueError("Unsettled or reordered recovery ledger")
        path = Path(row["output"])
        equal(file_hash(path / "report.json"), row["report_sha256"], "Prior recovery changed")
        actual = phase_evidence(path, "review", protocol["original_profile"])
        equal(actual["spent"], row["charged"], "Recovery ledger consumption differs")
        equal(spent, row["spent_before"], "Recovery omitted earlier costs")
        grant = {k: protocol["original_profile"]["budget"][k] - spent[k] for k in RESOURCES}
        if "ceiling" in row:
            cap = row["ceiling"]
            if set(cap) != set(RESOURCES) or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or v <= 0
                or (k != "wall_seconds" and type(v) is not int)
                for k, v in cap.items()
            ):
                raise ValueError("Invalid historical recovery ceiling")
            grant = {k: min(grant[k], cap[k]) for k in RESOURCES}
        equal(row["grant"], grant, "Recovery grant reset")
        for key in RESOURCES:
            equal(actual["lock"]["profile"]["budget"][key], grant[key], "Recovery exceeded admitted budget")
            spent[key] += actual["spent"][key]
        if i == index:
            equal(row["output"], str(run), "Wrong ledger attempt")
    if index >= len(ledger["attempts"]):
        raise ValueError("Missing recovery reservation")
    equal(spent, report["cumulative_spent"], "Cumulative costs changed")
    return {"source": source, "phase": phase, "spent": spent, "ledger": ledger}


def main(args):
    run = args.run.resolve()
    verified = verify_recovery(run)
    source, spent = verified["source"], verified["spent"]
    args.output.mkdir(parents=True, exist_ok=False)
    raw, plugin = source["domain"], source["job"]["plugin_id"]
    replayed = False
    if plugin in {"mcba.multisubstrate.v1", "alde.onehot_dnn.v1"}:
        base = (ROOT / "research_runs/rsi").resolve()
        campaign = inside(base, raw["source_campaign"])
        folder = Path(source["job"]["output"])
        auditfile = "audit/audit.json" if plugin == "mcba.multisubstrate.v1" else "audit/report.json"
        equal(file_hash(folder / auditfile), raw["audit_report_sha256"], "Domain audit changed")
        for name, sha in raw["source_hashes"].items():
            equal(file_hash(inside(campaign, name)), sha, "Domain source changed")
        if plugin == "mcba.multisubstrate.v1":
            command = [
                sys.executable,
                str(ROOT / "scripts/rsi/cases/audit_mcba_multisubstrate.py"),
                "--campaign",
                str(campaign),
                "--source",
                str(base / "case_assets/mcba_multisubstrate_source"),
                "--output",
                str(args.output / "domain_replay"),
            ]
        else:
            command = [
                str(base / "envs/alde/bin/python"),
                str(ROOT / "scripts/rsi/cases/audit_alde_reproduction.py"),
                "--run",
                str(campaign),
                "--output",
                str(args.output / "domain_replay"),
            ]
        with (args.output / "domain_replay.log").open("wb") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
        if plugin == "mcba.multisubstrate.v1":
            from agent.rsi.cases.mcba_evidence import summarize_campaign

            summary = summarize_campaign(read(campaign / "report.json"), read(args.output / "domain_replay/audit.json"))
            equal(summary, raw["summary"], "Replayed scientific summary differs")
        else:
            replay = read(args.output / "domain_replay/report.json")
            equal(replay["summaries"], raw["summaries"], "Replayed scientific summary differs")
            equal(replay["completed_and_verified"], raw["completed_and_verified"], "Replayed trajectory count differs")
        replayed = True
    result = {
        "status": "audit_completed",
        "run": str(run),
        "report_sha256": file_hash(run / "report.json"),
        "auditor_sha256": file_hash(Path(__file__)),
        "source_original_status": source["original_status"],
        "source_unchanged": True,
        "source_bound_review_verified": True,
        "original_submission_verified": True,
        "cumulative_spent": spent,
        "independent_domain_replay": replayed,
        "computational_agreement_passed": source["facts"].get("computational_agreement_passed"),
        "uninterrupted_end_to_end_verified": False,
        "new_computation_in_recovery": False,
        "scope": "Recovered evidence review with prior failures and costs retained; not an independent efficacy replicate",
    }
    write_json(args.output / "report.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
