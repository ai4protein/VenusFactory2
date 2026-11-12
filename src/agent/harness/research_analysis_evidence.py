"""Capture authenticated queued computations and bind their checkpoint artifacts.

These records attest execution, not scientific usefulness, data splits or model
quality. Scientific acceptance and independent model Review remain separate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from mcp_tools.scheduler.receipt_binding import value_digest
from utils.scoped_files import open_file, write_bytes

SHA = re.compile(r"[a-f0-9]{64}")
JOB = re.compile(r"[a-f0-9]{32}")
FIELDS = {"version", "tool", "category", "workspace_sha256", "arguments_sha256",
          "result_sha256", "input_files", "output_files"}
MAX_BYTES = 100 * 1024**2
MAX_VERIFICATION_BYTES = 32 * 1024**2
MAX_VERIFICATION_FILES = 128


class VerificationBudget:
    """Bound complete-file verification; a partial read never becomes a digest."""
    def __init__(self):
        self.bytes_left = MAX_VERIFICATION_BYTES
        self.files_left = MAX_VERIFICATION_FILES

    def read(self, workspace, name, *, max_bytes=MAX_BYTES):
        if self.files_left <= 0 or self.bytes_left <= 0:
            raise ValueError("Analysis verification budget exhausted")
        self.files_left -= 1
        limit = min(max_bytes, self.bytes_left)
        with open_file(workspace, name, max_bytes=limit) as stream:
            raw = stream.read(limit + 1)
        self.bytes_left -= len(raw)
        if len(raw) > limit:
            raise ValueError("Analysis verification byte budget exhausted")
        return raw


def workspace_digest(workspace):
    return hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()


def file_digest(workspace, name, *, budget=None):
    if budget is not None:
        return hashlib.sha256(budget.read(workspace, name)).hexdigest()
    size, digest = 0, hashlib.sha256()
    with open_file(workspace, name, max_bytes=MAX_BYTES) as stream:
        while chunk := stream.read(1024**2):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError("Analysis artifact is too large")
            digest.update(chunk)
    return digest.hexdigest()


def unchanged_files(workspace, files, *, budget=None):
    if not isinstance(files, dict) or len(files) > 100:
        raise ValueError("Invalid analysis file binding")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str) or not SHA.fullmatch(expected):
            raise ValueError("Invalid analysis file digest")
        if file_digest(workspace, name, budget=budget) != expected:
            raise ValueError("Analysis input or output changed")


def validate_evidence(evidence, workspace, *, budget=None):
    from mcp_tools.scheduler.analysis_evidence import ANALYSIS_CATEGORIES

    if (not isinstance(evidence, dict) or set(evidence) != FIELDS
            or type(evidence["version"]) is not int or evidence["version"] != 1
            or evidence["category"] not in ANALYSIS_CATEGORIES
            or not isinstance(evidence["tool"], str) or not evidence["tool"]
            or evidence["workspace_sha256"] != workspace_digest(workspace)):
        raise ValueError("Invalid queued analysis evidence")
    for key in ("arguments_sha256", "result_sha256"):
        if not isinstance(evidence[key], str) or not SHA.fullmatch(evidence[key]):
            raise ValueError("Invalid queued analysis digest")
    unchanged_files(workspace, evidence["input_files"], budget=budget)
    unchanged_files(workspace, evidence["output_files"], budget=budget)


def capture(service, *, job, name, arguments, value, owner, parent_job, session_id):
    """Called only by the host after its authenticated scheduler invocation."""
    if job is None or not isinstance(job, dict):
        return None
    result = job.get("result")
    if not isinstance(result, dict):
        return None
    evidence = result.get("analysis_evidence")
    if evidence is None:
        return None
    if (not owner or not parent_job or not session_id
            or not isinstance(job.get("id"), str) or not JOB.fullmatch(job["id"])
            or job.get("tool") != name or job.get("status") != "completed"
            or job.get("parent_job") != parent_job or job.get("session_id") != session_id
            or job.get("arguments_sha256") != value_digest(arguments)
            or result.get("is_error") is not False or "value" not in result
            or value_digest(result["value"]) != value_digest(value)):
        raise ValueError("Queued analysis task binding mismatch")
    validate_evidence(evidence, service.workspace)
    if (evidence["tool"] != name or evidence["category"] != job.get("category")
            or evidence["arguments_sha256"] != value_digest(arguments)
            or evidence["result_sha256"] != value_digest(value)):
        raise ValueError("Queued analysis result binding mismatch")
    from .session_boundary import public_value
    document = public_value({"tool": name, "job_number": job.get("job_number"),
        "category": evidence["category"], "arguments": arguments,
        "input_files": evidence["input_files"], "output_files": evidence["output_files"],
        "result": value}, service.workspace)
    raw = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    if len(raw) > 16 * 1024**2:
        raise ValueError("Analysis result exceeds the artifact limit")
    artifact = {"path": "analysis/" + job["id"] + ".json", "sha256": hashlib.sha256(raw).hexdigest()}
    write_bytes(service.workspace, artifact["path"], raw)
    saved = {"version": 1, "task_id": job["id"], "parent_job": parent_job,
             "session_id": session_id, "owner_sha256": hashlib.sha256(owner.encode()).hexdigest(),
             "evidence": copy.deepcopy(evidence), "artifact": artifact}
    service.record("queued_analysis", {"tool": name, "arguments_sha256": value_digest(arguments)}, saved)
    from .research_analysis_provenance import sequence_input_feedback
    inputs = sequence_input_feedback(service.workspace, getattr(service, "registered_inputs", []), arguments)
    return {**artifact, "tool": name, "job_number": job.get("job_number"),
            **({"input_check": inputs} if inputs is not None else {}),
            "instruction": "Inspect this result and cite its path in the research checkpoint."}


def checkpointed(receipts, workspace, artifacts, *, budget=None):
    """Recheck protected records and all cited bytes; never trust workspace JSON."""
    budget = budget or VerificationBudget()
    accepted = []
    for row in receipts:
        if row.get("tool") != "queued_analysis":
            continue
        try:
            saved = row["result"]
            if (set(saved) != {"version", "task_id", "parent_job", "session_id", "owner_sha256", "evidence", "artifact"}
                    or type(saved["version"]) is not int or saved["version"] != 1
                    or not JOB.fullmatch(saved["task_id"]) or not saved["parent_job"] or not saved["session_id"]
                    or not SHA.fullmatch(saved["owner_sha256"])):
                continue
            artifact = saved["artifact"]
            if set(artifact) != {"path", "sha256"} or artifact["path"] != "analysis/" + saved["task_id"] + ".json":
                continue
            if artifacts.get(artifact["path"]) != artifact["sha256"]:
                continue
            if file_digest(workspace, artifact["path"], budget=budget) != artifact["sha256"]:
                continue
            validate_evidence(saved["evidence"], workspace, budget=budget)
            accepted.append(saved["task_id"])
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return accepted



def checkpoint_diagnostic(service, checkpoint):
    """Offer current verified records when an execution claim omitted its evidence.

    This is a repair hint, not step acceptance. Other execution evidence continues
    to use the same host validator as final review.
    """
    from .research_plan_evidence import checkpoint_steps, requirements

    selected = set(checkpoint["completed_steps"])
    steps = [step["id"] for step in service.plan["steps"]
             if step["id"] in selected and "execution" in requirements(step)]
    if not steps:
        return None

    candidates = []
    receipts = service.evidence_receipts(checkpoint)
    budget = VerificationBudget()
    for row in receipts:
        saved = row.get("result", {})
        artifact = saved.get("artifact") if isinstance(saved, dict) else None
        if row.get("tool") != "queued_analysis" or not isinstance(artifact, dict):
            continue
        path, sha = artifact.get("path"), artifact.get("sha256")
        if not isinstance(path, str) or not isinstance(sha, str):
            continue
        if checkpointed([row], service.workspace, {path: sha}, budget=budget):
            candidates.append({"path": path, "sha256": sha, "tool": saved["evidence"]["tool"]})
    if not candidates:
        return None

    # Check execution alone; data/model/Review requirements retain their normal
    # downstream checks and must not be replaced by an analysis citation.
    probe = {"steps": [{"id": steps[0], "verification": "execution"}]}
    checks = checkpoint_steps(probe, {**checkpoint, "completed_steps": [steps[0]]},
        workspace=service.workspace, receipts=receipts,
        experiments=service.development.records(), progress={})
    if checks[0]["passed"]:
        return None
    unique = {item["path"]: item for item in candidates}
    return {
        "code": "analysis_evidence_not_cited",
        "step_ids": steps,
        "available_evidence": list(unique.values()),
        "message": "Partial work was saved, but the claimed execution steps do not cite verified computation evidence.",
        "repair": (
            "Inspect the relevant available_evidence file, then call submit_research_checkpoint again "
            "with its exact path added to artifacts, keeping your report and other result files. "
            "Keep a step in completed_steps only if its scientific acceptance criteria are actually satisfied. "
            "If work is unfinished, submit completed_steps=[] to save a partial checkpoint. "
            "Do not rerun a valid calculation merely to obtain a missing citation. "
            "These records prove execution, not target relevance, report accuracy or scientific acceptance."
        ),
    }
