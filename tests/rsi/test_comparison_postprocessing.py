"""A terminal experiment must not remain running after its audit fails."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases import run_reap_goal_comparison as comparison

from agent.harness.contracts import file_hash
from agent.harness.host import write_json


def fixture(root):
    rows = [{"id": "accepted", "status": "completed"}, {"id": "exhausted", "status": "budget_exhausted"}]
    for row in rows:
        folder = root / "arms" / row["id"]
        folder.mkdir(parents=True)
        write_json(folder / "goal.json", row)
    report = {"status": "running", "allocations": rows, "planned": 2, "independent_audit_required": True}
    write_json(root / "report.json", report)
    return rows, report, {row["id"]: (root / "arms" / row["id"] / "goal.json").read_bytes() for row in rows}


def read_report(root):
    return json.loads((root / "report.json").read_text())


@pytest.mark.parametrize("failure_stage", ["development_barrier", "analysis", "publish_analysis"])
def test_postprocessing_failure_is_terminal_and_preserves_allocations(tmp_path, monkeypatch, failure_stage):
    rows, report, before = fixture(tmp_path)
    original_write = comparison.write_json

    def barrier(root, supplied):
        assert supplied == rows
        assert read_report(root)["postprocessing_stage"] == "development_barrier"
        if failure_stage == "development_barrier":
            raise ValueError("frozen development evidence changed")
        write_json(root / "development_barrier.json", {"all_terminal": True})
        return file_hash(root / "development_barrier.json")

    def analyze(root, supplied):
        assert supplied == rows
        assert read_report(root)["postprocessing_stage"] == "analysis"
        (root / "partial_replay.json").write_text('{"completed_models": 3}')
        if failure_stage == "analysis":
            raise RuntimeError("generated model failed (137)")
        return {"rows": rows}

    def write(path, value):
        if path.name == "analysis.json" and failure_stage == "publish_analysis":
            raise OSError("analysis output unavailable")
        original_write(path, value)

    monkeypatch.setattr(comparison, "write_json", write)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        comparison.finish_campaign(tmp_path, rows, report, analyze=analyze, barrier=barrier)
    saved = read_report(tmp_path)
    assert saved["status"] == "postprocessing_failed"
    assert saved["postprocessing_stage"] == failure_stage
    assert saved["allocations"] == rows and saved["planned"] == 2
    assert saved["future_scoring_completed"] is False
    assert saved["independent_audit_required"] is True
    assert saved["postprocessing_error"]["type"] in {"ValueError", "RuntimeError", "OSError"}
    assert saved["finished_unix"] > 0
    assert "analysis_sha256" not in saved
    for row in rows:
        assert (tmp_path / "arms" / row["id"] / "goal.json").read_bytes() == before[row["id"]]
    if failure_stage != "development_barrier":
        assert saved["development_barrier_sha256"] == file_hash(tmp_path / "development_barrier.json")
        assert (tmp_path / "partial_replay.json").exists()


def test_success_publishes_analysis_before_completed_state(tmp_path):
    rows, report, before = fixture(tmp_path)
    comparison.finish_campaign(tmp_path, rows, report, barrier=lambda *_: "sealed", analyze=lambda *_: {"rows": rows})
    saved = read_report(tmp_path)
    assert saved["status"] == "campaign_completed"
    assert saved["postprocessing_stage"] == "completed"
    assert saved["future_scoring_completed"] is True
    assert saved["analysis_sha256"] == file_hash(tmp_path / "analysis.json")
    assert saved["development_barrier_sha256"] == "sealed"
    assert saved["allocations"] == rows
    assert "postprocessing_error" not in saved
    for row in rows:
        assert (tmp_path / "arms" / row["id"] / "goal.json").read_bytes() == before[row["id"]]
