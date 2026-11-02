"""Continue a failed review of completed computation within the same goal.

This does not restart a job, alter its failed episode, or turn an operator
recovery into autonomous success. Legacy protocols do not opt in implicitly.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from audit_reproduction_recovery import verify_recovery
from resume_harness_reproduction import main as recover_review
from run_harness_probe import ROOT
from run_harness_reproduction import submission_receipt_matches
from run_protein_lifecycle import audit_events

from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.case_jobs import verify_job_report
from agent.rsi.reproduction_recovery import RESOURCES, RecoveryLedger, equal, load_source, read
from agent.rsi.storage import safe_child

POLICY = "verified_completed_job.v1"


def policy_enabled(protocol, state):
    value = protocol.get("review_continuation", "none")
    if value not in {"none", POLICY}:
        raise ValueError("Unknown review continuation policy")
    admitted = (state["definition"].get("admission") or {}).get("review_continuation", "none")
    if value != admitted:
        raise ValueError("Review continuation was not admitted with the goal")
    return value == POLICY


def source_evidence(folder, declaration, recipe, *, replay_artifacts=True):
    source = load_source(folder)
    report, protocol, job = read(folder / "report.json"), source["protocol"], source["job"]
    if report["status"] not in {"review_failed", "inference_budget_exhausted"} or any(
        report.get(k) is not True
        for k in ["agent_requested", "submission_receipt_verified", "implementation_unchanged"]
    ):
        raise ValueError("Continuation requires a fully verified original launch")
    equal(protocol["plugin"], declaration["plugin"], "Continuation plugin changed")
    equal(protocol["parameters"], declaration["parameters"], "Continuation allocation changed")
    equal(protocol.get("registered_recipe_sha256"), digest(recipe) if recipe else None, "Continuation recipe changed")
    if source["domain"].get("execution_origin") == "verified_existing":
        raise ValueError("Historical evidence review is not fresh computation")
    required = {"discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"}
    if recipe is not None:
        required.add("inspect_registered_reproduction")
    runtime = report["launch"]["run"]
    if runtime.get("status") != "completed" or runtime.get("audit", {}).get("passed") is not True:
        raise ValueError("Original launch did not complete")
    if not audit_events(folder / "launch", "launch", runtime, required)["passed"]:
        raise ValueError("Original launch tool chain is incomplete")
    if not submission_receipt_matches(folder / "workspace", {job["job_id"]: job}):
        raise ValueError("Original submission receipt changed")
    if replay_artifacts:
        verify_job_report(ROOT, job, source["domain"])
    source["source_hashes"][str(folder / "workspace/case_submission.json")] = file_hash(
        folder / "workspace/case_submission.json"
    )
    return source


def capture_pending(root, episode, output, declaration, recipe):
    source = source_evidence(output, declaration, recipe)
    value = {
        "policy": POLICY,
        "source_run": str(output.relative_to(root)),
        "source_sha256": digest(source["source_hashes"]),
        "job_id": source["job"]["job_id"],
    }
    path = episode / "pending_review.json"
    write_json(path, value)
    path.chmod(0o444)
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path)}


def pending_evidence(root, record, declaration, recipe, *, replay_artifacts=True):
    if set(record) != {"path", "sha256"}:
        raise ValueError("Invalid pending review reference")
    path = safe_child(root, record["path"])
    equal(file_hash(path), record["sha256"], "Pending review receipt changed")
    value = read(path)
    if set(value) != {"policy", "source_run", "source_sha256", "job_id"} or value["policy"] != POLICY:
        raise ValueError("Invalid pending review policy")
    folder = safe_child(root, value["source_run"])
    source = source_evidence(folder, declaration, recipe, replay_artifacts=replay_artifacts)
    equal(digest(source["source_hashes"]), value["source_sha256"], "Pending scientific evidence changed")
    equal(source["job"]["job_id"], value["job_id"], "Pending job identity changed")
    return folder, source


async def continue_review(root, output, record, declaration, recipe, profile, credential, context):
    source_run, source = pending_evidence(root, record, declaration, recipe)
    original, current = deepcopy(source["protocol"]["profile"]), profile.public()
    for value in [original, current]:
        value["budget"] = {k: v for k, v in value["budget"].items() if k not in RESOURCES}
    equal(original, current, "Continuation cannot switch harness, model or inference settings")
    # source_evidence adds the submission receipt binding; the reusable recovery
    # ledger itself retains load_source's original, backward-compatible hash set.
    source = load_source(source_run)
    ledger = RecoveryLedger(root / "review_recovery_ledger", source)
    with ledger.locked() as data:
        spent = dict(source["spent"])
        for row in data["attempts"]:
            if row["state"] == "settled":
                equal(file_hash(Path(row["output"]) / "report.json"), row["report_sha256"], "Prior recovery changed")
            for key in RESOURCES:
                spent[key] += row["charged"][key]
        remaining = {k: source["protocol"]["profile"]["budget"][k] - spent[k] for k in RESOURCES}
    if any(v <= 0 for v in remaining.values()):
        return {
            "status": "review_budget_exhausted",
            "needs_input": True,
            "summary": "Completed computation retained; original review allowance exhausted. No job was resubmitted.",
        }
    return await recover_review(
        SimpleNamespace(source_run=source_run, output=output),
        credential=credential,
        continuation={"policy": POLICY, "goal_root": str(root), "pending_review": record},
        ceiling={k: getattr(profile.budget, k) for k in RESOURCES},
        goal_context=context,
    )


def verify_continuation(folder, declaration, recipe, *, replay_artifacts=True):
    verified = verify_recovery(folder)
    protocol, report = read(folder / "protocol.json"), read(folder / "report.json")
    binding = protocol.get("continuation")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"policy", "goal_root", "pending_review"}
        or binding["policy"] != POLICY
    ):
        raise ValueError("An operator recovery cannot complete a persistent goal")
    root = Path(binding["goal_root"])
    if not root.is_absolute() or any(p.is_symlink() for p in [root, *root.parents]):
        raise ValueError("Invalid continuation goal root")
    relative = folder.relative_to(root)
    if safe_child(root, str(relative)) != folder or relative.parts[0] != "episodes":
        raise ValueError("Continuation output is outside its goal")
    goal_protocol = read(safe_child(root, "protocol.json"))
    from agent.science_goal import GoalStore

    if not policy_enabled(goal_protocol, GoalStore(root).read()):
        raise ValueError("Continuation policy was not admitted")
    source_run, source = pending_evidence(
        root, binding["pending_review"], declaration, recipe, replay_artifacts=replay_artifacts
    )
    equal(str(source_run), protocol["source_run"], "Continuation source differs from pending job")
    expected_ledger = RecoveryLedger(root / "review_recovery_ledger", load_source(source_run)).path
    equal(str(expected_ledger), protocol["ledger"], "Continuation ledger left the goal")
    index = protocol["reservation"]["index"]
    row = verified["ledger"]["attempts"][index]
    # The settled record must retain the exact original reservation, including
    # its cycle cap. Settlement only changes state/charge and adds a report hash.
    for key in ["index", "output", "grant", "spent_before", "ceiling"]:
        equal(row.get(key), protocol["reservation"].get(key), "Continuation reservation changed")
    sessions = list(source["sessions"])
    attempts = []
    for item in verified["ledger"]["attempts"][: index + 1]:
        recovery = Path(item["output"])
        sid = read(recovery / "review/run.json")["session_id"]
        if not isinstance(sid, str) or not sid or sid in sessions:
            raise ValueError("Continuation needs distinct sessions across all attempts")
        sessions.append(sid)
        attempts.append({"output": str(recovery), "report_sha256": item["report_sha256"]})
    names = ["protocol.json", "report.json", "review_workspace/review.json"] + [
        "review/" + n for n in ["run.json", "profile.lock.json", "events.jsonl", "model_requests.jsonl"]
    ]
    return {
        "files": {name: file_hash(safe_child(folder, name)) for name in names},
        "source_files": source["source_hashes"],
        "recovery_attempts": attempts,
        "sessions": sessions,
        "job_id": source["job"]["job_id"],
        "facts_sha256": digest(source["facts"]),
        "new_computation": True,
        "new_computation_in_recovery": False,
        "uninterrupted_end_to_end_verified": False,
        "recovery_trigger": report["recovery_trigger"],
        "full_paper_reproduced": False,
    }
