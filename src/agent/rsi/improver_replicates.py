"""A global development/scoring barrier for independent policy-revision repeats."""

from pathlib import Path

from agent.harness.contracts import file_hash
from agent.harness.host import write_json
from agent.science_goal import GoalStore


def interleaved_schedule(schedules):
    if set(schedules) != {"r00", "r01", "r02"}:
        raise ValueError("Three predeclared independent API revisions are required")
    groups = sorted(schedules)
    if any(len(rows) != 18 or len({r["id"] for r in rows}) != 18 for rows in schedules.values()):
        raise ValueError("Every revision requires all 18 matched offspring allocations")
    result = []
    for slot in range(18):
        offset = slot % len(groups)
        for group in groups[offset:] + groups[:offset]:
            result.append({"replicate": group, "allocation": schedules[group][slot]["id"]})
    return result


def freeze_batch_barrier(root, schedule, audit_root, read):
    """Require every independently replayed goal before publishing any future score."""
    if len(schedule) != 54 or len({(r["replicate"], r["allocation"]) for r in schedule}) != 54:
        raise ValueError("Keep the full 54-allocation denominator")
    if any(sum(r["replicate"] == group for r in schedule) != 18 for group in ("r00", "r01", "r02")):
        raise ValueError("Every independent revision must retain all 18 offspring")
    records = []
    for row in schedule:
        group = row["replicate"]
        if group not in {"r00", "r01", "r02"} or Path(row["allocation"]).name != row["allocation"]:
            raise ValueError("Invalid allocation identity")
        folder = root / "offspring" / group / "arms" / row["allocation"]
        state = GoalStore(folder).read()
        result = read(folder / "comparison_result.json")
        audit_path = audit_root / "development" / group / (row["allocation"] + ".json")
        evidence = read(audit_path)
        if state["status"] not in {"completed", "budget_exhausted"} or state["inflight"] is not None:
            raise ValueError("Every development allocation must be terminal")
        if result.get("status") != state["status"] or result.get("implementation_unchanged") is not True:
            raise ValueError("Development result or implementation is inconsistent")
        if any((folder / name).exists() for name in ("future_metrics.json", "selected_future_metrics.json")):
            raise ValueError("Future scores appeared before the 54-allocation barrier")
        if evidence.get("status") != "passed" or evidence.get("future_scoring_deferred") is not True:
            raise ValueError("Every allocation needs a successful independent deferred audit")
        files = {name: file_hash(folder / name) for name in ("goal.json", "protocol.json", "comparison_result.json")}
        if evidence.get("source_bindings") != files:
            raise ValueError("Allocation changed after independent replay")
        records.append(
            {**row, "status": state["status"], "source_bindings": files, "audit_sha256": file_hash(audit_path)}
        )
    barrier = {
        "count": 54,
        "independent_revisions": 3,
        "future_metrics_present": False,
        "allocations": records,
        "scope": "Independent API revision attempts; shared retrospective proteins and partitions are not independent biological replicates",
    }
    write_json(root / "development_barrier.json", barrier)
    return barrier
