"""Synthetic global-barrier checks; never substitute for scientific outcomes."""

import json

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import file_hash
from agent.harness.host import write_json
from agent.rsi.improver_replicates import freeze_batch_barrier, interleaved_schedule
from agent.science_goal import GoalSpec, GoalStore


def read(path):
    return json.loads(path.read_text())


def schedules():
    return {group: [{"id": f"allocation_{i:02d}"} for i in range(18)] for group in ("r00", "r01", "r02")}


def fixture(root, audit_root):
    profile = Profile(
        "science_expert", "langgraph", ModelBinding("fixture", "fixture", "https://example.org/v1"), Budget()
    )
    rows = interleaved_schedule(schedules())
    for group in schedules():
        (audit_root / "development" / group).mkdir(parents=True)
    for index, row in enumerate(rows):
        arm = root / "offspring" / row["replicate"] / "arms" / row["allocation"]
        store = GoalStore.create(arm, GoalSpec("Synthetic model goal", ("accepted",)), profile)
        state = store.mutate(lambda s, index=index: s.update(status="completed" if index % 3 else "budget_exhausted"))
        write_json(arm / "protocol.json", {"synthetic": True})
        write_json(arm / "comparison_result.json", {"status": state["status"], "implementation_unchanged": True})
        write_json(
            audit_root / "development" / row["replicate"] / (row["allocation"] + ".json"),
            {
                "status": "passed",
                "future_scoring_deferred": True,
                "source_bindings": {
                    name: file_hash(arm / name) for name in ("goal.json", "protocol.json", "comparison_result.json")
                },
            },
        )
    return rows


def test_schedule_counterbalances_revisions_and_retains_every_allocation():
    rows = interleaved_schedule(schedules())
    assert len(rows) == 54
    assert [r["replicate"] for r in rows[:9]] == ["r00", "r01", "r02", "r01", "r02", "r00", "r02", "r00", "r01"]
    for group in schedules():
        assert [r["allocation"] for r in rows if r["replicate"] == group] == [r["id"] for r in schedules()[group]]
    partial = schedules()
    partial["r02"].pop()
    with pytest.raises(ValueError):
        interleaved_schedule(partial)


def test_global_barrier_retains_successes_and_budget_exhaustion(tmp_path):
    root, audits = tmp_path / "batch", tmp_path / "audits"
    rows = fixture(root, audits)
    barrier = freeze_batch_barrier(root, rows, audits, read)
    assert barrier["count"] == 54 and barrier["independent_revisions"] == 3
    assert {r["status"] for r in barrier["allocations"]} == {"completed", "budget_exhausted"}
    assert read(root / "development_barrier.json") == barrier


@pytest.mark.parametrize(
    "mutation",
    ["missing", "pending", "failed_audit", "changed_after_replay", "early_score", "worker_failed", "missing_audit"],
)
def test_no_future_barrier_on_incomplete_changed_or_unverified_evidence(tmp_path, mutation):
    root, audits = tmp_path / "batch", tmp_path / "audits"
    rows = fixture(root, audits)
    first = rows[0]
    arm = root / "offspring" / first["replicate"] / "arms" / first["allocation"]
    audit_file = audits / "development" / first["replicate"] / (first["allocation"] + ".json")
    if mutation == "missing":
        rows.pop()
    elif mutation == "pending":
        GoalStore(arm).mutate(lambda s: s.update(status="active"))
    elif mutation == "failed_audit":
        evidence = read(audit_file)
        evidence["status"] = "failed"
        write_json(audit_file, evidence)
    elif mutation == "changed_after_replay":
        write_json(arm / "protocol.json", {"synthetic": "changed"})
    elif mutation == "early_score":
        write_json(arm / "selected_future_metrics.json", {"score": 0.9})
    elif mutation == "worker_failed":
        write_json(arm / "comparison_result.json", {"status": "worker_error", "implementation_unchanged": True})
    elif mutation == "missing_audit":
        audit_file.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze_batch_barrier(root, rows, audits, read)
    assert not (root / "development_barrier.json").exists()
