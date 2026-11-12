"""Bind plan completion to host-owned evidence instead of generated claims."""
from __future__ import annotations

from .contracts import file_hash
from .research_verification_contract import VERIFICATION_KINDS as VERIFICATION_KINDS
from .research_workspace import safe_file

_TOOL_REQUIREMENTS = {
    "validate_research_evaluation": "data",
    "prepare_research_dataset": "data",
    "train_research_model": "model",
    "evaluate_research_model": "model",
}


def requirements(step):
    result = {step.get("verification", "artifact")}
    text = (step.get("title", "") + " " + step.get("acceptance", "")).lower()
    result.update(kind for name, kind in _TOOL_REQUIREMENTS.items() if name in text)
    return sorted(result)


def checkpoint_steps(plan, checkpoint, *, workspace, receipts, experiments, progress):
    """Only protected receipts/experiments are authoritative; files alone are not."""
    artifacts = {row["path"]: row["sha256"] for row in (checkpoint or {}).get("artifacts", [])}
    selected = set((checkpoint or {}).get("completed_steps", []))
    measured = []
    for record in experiments:
        files = record.get("files", {})
        cited = any(artifacts.get(name) == sha for name, sha in files.items())
        cited = cited or ("experiments/" + record["id"] + ".json" in artifacts)
        if not cited or not files:
            continue
        try:
            unchanged = all(file_hash(safe_file(workspace, name)) == sha for name, sha in files.items())
        except (OSError, ValueError):
            unchanged = False
        if unchanged and record.get("score_origin") == "host_validation" and record.get("fresh_process_replay") is True:
            measured.append(record)
    hashes = set(artifacts.values())
    executed = any(
        row.get("tool") == "run_research_python"
        and type(row.get("result", {}).get("exit_code")) is int
        and row["result"]["exit_code"] == 0
        and row["result"].get("submitted_source_sha256") in hashes
        for row in receipts
    )
    partitioned = any(
        row.get("tool") in {"validate_research_evaluation", "prepare_research_dataset"}
        and row.get("result", {}).get("passed") is True
        and row["result"].get("source_sha256") in hashes
        and artifacts.get(row["result"].get("evaluation_file")) == row["result"].get("source_sha256")
        for row in receipts
    )
    from .research_analysis_evidence import checkpointed
    analysis_jobs = checkpointed(receipts, workspace, artifacts)
    facts = {
        "artifact": bool(artifacts),
        "execution": executed or bool(measured) or bool(analysis_jobs),
        "data": partitioned or bool(measured),
        "model": bool(measured),
        "revision": any(row.get("parent") and row.get("variant") in {"candidate", "ablation"} for row in measured),
        "review": any(row.get("status") == "approved" for row in progress.get("publication_reviews", [])),
        "delivery": bool(progress.get("deliveries")),
    }
    guidance = {
        "artifact": "Checkpoint the actual result files.",
        "execution": "Run the checkpointed script successfully, or checkpoint the host analysis result from a completed queued computation. Writing a function or a report alone does not execute it.",
        "data": "Validate the split with validate_research_evaluation and checkpoint the validated evaluation file.",
        "model": "Use train_research_model or evaluate_research_model to record a measured generation; checkpoint its unchanged runtime files or experiments/<id>.json.",
        "revision": "Measure a revised generation against its baseline and checkpoint that generation.",
        "review": "Submit the candidate and complete independent Review.",
        "delivery": "Finish a reviewed deliverable package.",
    }
    result = []
    for step in plan["steps"]:
        if step["id"] not in selected:
            continue
        kinds = requirements(step)
        missing = [kind for kind in kinds if not facts.get(kind, False)]
        result.append({
            "id": step["id"], "passed": not missing, "requirements": kinds,
            "analysis_job_ids": analysis_jobs if "execution" in kinds else [],
            "experiment_ids": [row["id"] for row in measured] if any(k in kinds for k in ("model", "revision")) else [],
            "issues": [guidance.get(kind, "Unsupported verification requirement.") for kind in missing],
        })
    return result
