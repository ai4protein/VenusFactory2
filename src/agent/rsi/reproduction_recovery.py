"""Read-only provenance checks and cumulative budgets for interrupted case reviews.

A recovery is a new, explicitly attributed review episode. It neither changes
an earlier failure nor submits another scientific computation.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace

from .case_evidence import canonical_claims
from .case_plugins import PLUGINS
from .versions import VersionStore

RESOURCES = ("model_calls", "tool_calls", "wall_seconds")


def read(path):
    return json.loads(Path(path).read_text())


def equal(actual, expected, label):
    if digest(actual) != digest(expected):
        raise ValueError(label)


def inside(base, name):
    path = (Path(base) / name).resolve()
    path.relative_to(Path(base).resolve())
    return path


def phase_evidence(run, phase, original_profile):
    folder = run / phase
    runtime, lock = read(folder / "run.json"), read(folder / "profile.lock.json")
    equal(digest({k: v for k, v in lock.items() if k != "sha256"}), lock["sha256"], "Changed profile lock")
    equal(runtime["profile_sha256"], lock["sha256"], "Changed runtime binding")
    equal(
        lock["profile"].get("request_recovery", "none"),
        original_profile.get("request_recovery", "none"),
        "Changed request recovery policy",
    )
    for key in ["model", "harness", "mode", "required_controls"]:
        equal(lock["profile"][key], original_profile[key], "Changed execution profile: " + key)
    for key, value in lock["profile"]["budget"].items():
        if key not in RESOURCES:
            equal(value, original_profile["budget"][key], "Changed inference setting: " + key)
    if runtime.get("binding_intact") is not True:
        raise ValueError("Runtime binding was not preserved")
    cap_root = run / (phase + "_capability") / "bundles/versions"
    caps = list(cap_root.iterdir())
    if len(caps) != 1:
        raise ValueError("Ambiguous phase capability")
    manifest = read(caps[0] / "manifest.json")
    VersionStore(cap_root.parent).verify(manifest["version_id"])
    work = run / ("workspace" if phase == "launch" else "review_workspace")
    for prefix, hashes in [(work, lock["task"]["inputs"]), (caps[0], lock["task"]["capabilities"])]:
        for name, sha in hashes.items():
            equal(file_hash(inside(prefix, name)), sha, "Changed bound file")
    trace = audit_trace(folder / "events.jsonl", lock["task"]["tools"], lock["profile"]["budget"]["tool_calls"])
    requests = audit_requests(folder / "model_requests.jsonl")
    if not trace["passed"]:
        raise ValueError("Incomplete or invalid tool trace")
    records = [json.loads(s) for s in (folder / "model_requests.jsonl").read_text().splitlines()]
    ids = [row["call"] for row in records if row["event"] == "request"]
    replies = [row["call"] for row in records if row["event"] == "response"]
    transports = [row["call"] for row in records if row["event"] == "transport_error"]
    if not ids or len(set(ids)) != len(ids) or any(replies.count(i) > 1 for i in replies):
        raise ValueError("Missing or ambiguous request accounting")
    if set(ids) != set(replies) | set(transports):
        raise ValueError("Unterminated requests require host reconciliation")
    if any(row["event"] in {"policy_rejected", "budget_rejected"} for row in records):
        raise ValueError("Policy or budget rejection cannot be bypassed by recovery")
    spent = {"model_calls": len(ids), "tool_calls": trace["tool_calls"], "wall_seconds": runtime["host_wall_seconds"]}
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in spent.values()):
        raise ValueError("Invalid phase accounting")
    for key in RESOURCES:
        if (
            spent[key] > lock["profile"]["budget"][key]
            or lock["profile"]["budget"][key] > original_profile["budget"][key]
        ):
            raise ValueError("Phase budget exceeded")
    equal(trace, runtime["audit"]["events"], "Changed event accounting")
    equal(requests, runtime["audit"]["requests"], "Changed request accounting")
    return {"runtime": runtime, "lock": lock, "spent": spent, "requests": requests, "trace": trace}


def load_source(run):
    run = Path(run).resolve()
    report, protocol = read(run / "report.json"), read(run / "protocol.json")
    if report["status"] not in {"launch_or_job_failed", "review_failed", "inference_budget_exhausted"}:
        raise ValueError("Recovery requires a terminal interrupted reproduction")
    equal(digest(protocol), report["protocol_sha256"], "Source protocol changed")
    for name, sha in protocol["implementation"].items():
        equal(file_hash(inside(run / "implementation", name)), sha, "Archived source changed")
    phases = {}
    for phase in ["launch", "review"]:
        if phase in report:
            phases[phase] = phase_evidence(run, phase, protocol["profile"])
            equal(phases[phase]["runtime"], report[phase]["run"], "Source runtime changed")
    if "launch" not in phases:
        raise ValueError("No attributable launch")
    jobs = read(run / "controller/jobs.json")["jobs"]
    if len(jobs) != 1 or protocol["maximum_jobs"] != 1:
        raise ValueError("Recovery requires exactly one admitted job")
    job = next(iter(jobs.values()))
    if job["status"] != "completed" or job["exit_code"] != 0:
        raise ValueError("Scientific job has not completed; do not restart it")
    equal(job["plugin_id"], protocol["plugin"], "Plugin changed")
    equal(job["parameters"], protocol["parameters"], "Allocation changed")
    entry = PLUGINS[protocol["plugin"]].entry
    equal(job["entry_sha256"], protocol["implementation"][entry], "Job entry changed")
    rows = [json.loads(s) for s in (run / "launch/events.jsonl").read_text().splitlines()]
    ends = {e["tool_call_id"]: e for e in rows if e.get("kind") == "tool_result"}
    witnesses = []
    for row in rows:
        if row.get("kind") != "tool_call_start" or row.get("tool_name") != "mcp__venusfactory__submit_reproduction":
            continue
        result = ends[row["tool_call_id"]]
        if result.get("is_error"):
            continue
        returned = json.loads(result["output"])
        if not returned.get("success"):
            continue
        equal(row["args"]["plugin_id"], job["plugin_id"], "Submitted plugin differs")
        equal(json.loads(row["args"]["parameters_json"]), job["parameters"], "Submitted allocation differs")
        for key in ["job_id", "plugin_id", "parameters", "entry_sha256"]:
            equal(returned[key], job[key], "Submitted job identity differs")
        witnesses.append(row["tool_call_id"])
    if not witnesses:
        raise ValueError("No successful agent submission witness")
    folder = inside(run, job["output"])
    domain = folder / "report.json"
    equal(file_hash(domain), job["report_sha256"], "Domain report changed")
    raw = read(domain)
    facts = canonical_claims(job["plugin_id"], raw)
    equal(facts, job["evidence"], "Job evidence differs from domain report")
    reported_jobs = report.get("jobs", [])
    if len(reported_jobs) != 1:
        raise ValueError("Terminal job summary missing")
    for key in ["job_id", "status", "report_sha256", "evidence"]:
        equal(reported_jobs[0][key], job[key], "Terminal job summary changed")
    spent = {k: sum(p["spent"][k] for p in phases.values()) for k in RESOURCES}
    paths = [run / "report.json", run / "protocol.json", run / "controller/jobs.json", domain]
    for phase in phases:
        paths += [
            run / phase / name for name in ["run.json", "profile.lock.json", "events.jsonl", "model_requests.jsonl"]
        ]
    return {
        "run": str(run),
        "original_status": report["status"],
        "protocol": protocol,
        "job": job,
        "facts": facts,
        "domain": raw,
        "spent": spent,
        "source_hashes": {str(p): file_hash(p) for p in paths},
        "sessions": [p["runtime"]["session_id"] for p in phases.values()],
        "submission_witnesses": witnesses,
    }


class RecoveryLedger:
    """One durable reservation per attempt; an unreconciled crash spends its grant.

    Reserving under flock prevents two recoveries spending the same remaining
    budget. Only a terminal, hashed report can release unused resources.
    """

    def __init__(self, root, source):
        self.source = source
        identity = digest({"run": source["run"], "protocol": digest(source["protocol"])})
        self.path = Path(root) / (identity + ".json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = (
                read(self.path)
                if self.path.exists()
                else {"source_hashes": self.source["source_hashes"], "attempts": []}
            )
            equal(data["source_hashes"], self.source["source_hashes"], "Recovery source changed")
            yield data
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(data, indent=2) + "\n")
            os.replace(temp, self.path)

    def reserve(self, output, *, ceiling=None):
        with self.locked() as data:
            spent = dict(self.source["spent"])
            for row in data["attempts"]:
                if row["output"] == str(Path(output).resolve()):
                    raise ValueError("Recovery output already reserved")
                if row["state"] == "settled":
                    equal(
                        file_hash(Path(row["output"]) / "report.json"), row["report_sha256"], "Prior recovery changed"
                    )
                for key in RESOURCES:
                    spent[key] += row["charged"][key]
            remaining = {k: self.source["protocol"]["profile"]["budget"][k] - spent[k] for k in RESOURCES}
            if ceiling is not None:
                if set(ceiling) != set(RESOURCES) or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    or v <= 0
                    or (k != "wall_seconds" and type(v) is not int)
                    for k, v in ceiling.items()
                ):
                    raise ValueError("Invalid recovery reservation ceiling")
                remaining = {k: min(remaining[k], ceiling[k]) for k in RESOURCES}
            if any(v <= 0 for v in remaining.values()):
                raise ValueError("Cumulative recovery budget exhausted or reserved")
            record = {
                "index": len(data["attempts"]),
                "output": str(Path(output).resolve()),
                "state": "reserved",
                "charged": remaining,
                "grant": remaining,
                "spent_before": spent,
                **({"ceiling": dict(ceiling)} if ceiling is not None else {}),
            }
            data["attempts"].append(record)
        return record

    def settle(self, index):
        with self.locked() as data:
            row = data["attempts"][index]
            if row["state"] != "reserved":
                raise ValueError("Recovery already settled")
            path = Path(row["output"])
            report = read(path / "report.json")
            if report["status"] not in {
                "recovered_review",
                "review_failed",
                "source_changed",
                "implementation_changed",
                "budget_exceeded",
            }:
                raise ValueError("Only terminal recoveries can settle")
            evidence = phase_evidence(path, "review", self.source["protocol"]["profile"])
            equal(evidence["runtime"], report["review"]["run"], "Recovery runtime changed")
            for key in RESOURCES:
                equal(evidence["lock"]["profile"]["budget"][key], row["grant"][key], "Recovery grant changed")
            equal(evidence["spent"], report["spent"], "Recovery accounting changed")
            row.update(state="settled", charged=evidence["spent"], report_sha256=file_hash(path / "report.json"))
        return row
