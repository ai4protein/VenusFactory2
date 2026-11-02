import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from audit_reap_transfer import audit, verify_barrier
from run_reap_transfer_campaign import freeze_development_barrier

from agent.harness.contracts import file_hash
from agent.rsi.reap_experience import design


def fixture(root):
    groups = design()
    for group in groups:
        folder = root / f"group_{group['id']:02d}"
        folder.mkdir()
        arms = [
            {
                "id": i,
                "family": "P450" if i < 2 else "SaSrtA",
                "mode": "science_agent" if i % 2 else "science_expert",
                "condition": group["condition"],
                "status": "development_failed",
            }
            for i in range(4)
        ]
        (folder / "report.json").write_text(
            json.dumps({"status": "development_completed", "source_intact": True, "arms": arms})
        )
    identity = freeze_development_barrier(root, groups)
    (root / "report.json").write_text(json.dumps({"development_barrier_sha256": identity}))
    for group in groups:
        path = root / f"group_{group['id']:02d}/report.json"
        report = json.loads(path.read_text())
        report.update(status="campaign_completed", future_scoring_started_after_global_development=True)
        for arm in report["arms"]:
            arm["future_scoring"] = {"status": "no_verified_candidate"}
        path.write_text(json.dumps(report))
    return groups


def test_all_failures_still_have_36_immutable_allocations(tmp_path):
    groups = fixture(tmp_path)
    assert verify_barrier(tmp_path, groups)["allocations"] == 36


@pytest.mark.parametrize("mutation", ["dropped_arm", "rewritten_status", "earlier_future_score"])
def test_barrier_rejects_changed_or_premature_outcomes(tmp_path, mutation):
    groups = fixture(tmp_path)
    folder = tmp_path / "group_00"
    if mutation == "earlier_future_score":
        path = folder / "development_terminal.json"
        report = json.loads(path.read_text())
        report["arms"][0]["future_metrics"] = {"spearman": 0.99}
        path.write_text(json.dumps(report))
        barrier_path = tmp_path / "development_barrier.json"
        barrier = json.loads(barrier_path.read_text())
        barrier["groups"][0]["terminal_sha256"] = file_hash(path)
        barrier_path.write_text(json.dumps(barrier))
        (tmp_path / "report.json").write_text(json.dumps({"development_barrier_sha256": file_hash(barrier_path)}))
    else:
        path = folder / "report.json"
        report = json.loads(path.read_text())
        if mutation == "dropped_arm":
            report["arms"].pop()
        else:
            report["arms"][0]["status"] = "completed"
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_barrier(tmp_path, groups)


def test_incomplete_real_campaign_is_not_audited_as_complete(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"status": "running", "completed_allocations": 4}))
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="Complete 36"):
        audit(tmp_path, tmp_path, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()
