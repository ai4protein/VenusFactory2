"""Host-recorded access failures that require restoration before another episode."""

from __future__ import annotations

import json
from pathlib import Path

from agent.harness.contracts import file_hash


def access_wait(execution, goal_root):
    """Return a waiting outcome only for the final paired 401/402/403 response.

    This reads host request records, never model prose. A recovered request or a
    completed executor does not trigger waiting. Existing usage is not altered.
    """
    execution, goal_root = Path(execution), Path(goal_root).resolve()
    run_path, trace = execution / "run.json", execution / "model_requests.jsonl"
    if not run_path.exists() or not trace.exists():
        return {}
    for path in (run_path, trace):
        path.resolve().relative_to(goal_root)
        if any(p.is_symlink() for p in (path, *path.parents) if p != goal_root.parent):
            raise ValueError("Symlinked provider-access evidence")
    run = json.loads(run_path.read_text())
    if run.get("status") == "completed":
        return {}
    records = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    requests = [row for row in records if row.get("event") == "request"]
    if not requests:
        return {}
    last_call = requests[-1].get("call")
    if type(last_call) is not int:
        return {}
    responses = [row for row in records if row.get("event") == "response" and row.get("call") == last_call]
    if len(responses) != 1:
        return {}
    response = responses[0]
    status = response.get("status")
    if type(status) is not int or status not in {401, 402, 403} or response.get("completed") is not False:
        return {}
    return {
        "needs_input": True,
        "summary": f"Model provider returned HTTP {status}; restore provider access before resuming. Failed attempt and usage retained.",
        "dependency_failure": {
            "kind": "model_provider_access",
            "http_status": status,
            "call": last_call,
            "run_record": str(run_path.relative_to(goal_root)),
            "run_sha256": file_hash(run_path),
            "request_record": str(trace.relative_to(goal_root)),
            "request_sha256": file_hash(trace),
            "automatic_retry": False,
            "model_fallback": False,
        },
    }
