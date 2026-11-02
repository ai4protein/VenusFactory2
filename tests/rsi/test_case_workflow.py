import json
import sys
from dataclasses import replace

import pytest

from agent.rsi.case_workflow import CaseStage, CaseWorkflow


def make_stage(tmp_path, code):
    input_path = tmp_path / "input.txt"
    input_path.write_text("frozen")
    output = tmp_path / "run/result.txt"

    def validate(path):
        if path.read_text() != "scientific result":
            raise ValueError("invalid result")
        return {"valid": True}

    return CaseStage("stage", (sys.executable, "-c", code, str(output)), tmp_path, (input_path,), output, 5, validate)


def test_resume_checks_inputs_and_outputs(tmp_path):
    stage = make_stage(tmp_path, "import sys,pathlib;pathlib.Path(sys.argv[1]).write_text('scientific result')")
    workflow = CaseWorkflow(tmp_path / "run", {"case": "fixture"})
    first = workflow.run_stage(stage)
    assert workflow.run_stage(stage) == first
    stage.output.write_text("tampered")
    with pytest.raises(ValueError, match="output changed"):
        workflow.run_stage(stage)
    stage.output.write_text("scientific result")
    stage.inputs[0].write_text("changed")
    with pytest.raises(ValueError, match="inputs or parameters"):
        workflow.run_stage(stage)


def test_invalid_artifact_not_completion(tmp_path):
    stage = make_stage(tmp_path, "import sys,pathlib;pathlib.Path(sys.argv[1]).write_text('wrong')")
    workflow = CaseWorkflow(tmp_path / "run", {})
    with pytest.raises(ValueError, match="invalid result"):
        workflow.run_stage(stage)
    assert json.loads((workflow.root / "stage.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="failed/partial"):
        workflow.run_stage(stage)


def test_launch_error_recorded(tmp_path):
    stage = replace(make_stage(tmp_path, ""), argv=("/nonexistent-science-binary",))
    workflow = CaseWorkflow(tmp_path / "run", {})
    with pytest.raises(FileNotFoundError):
        workflow.run_stage(stage)
    assert json.loads((workflow.root / "stage.json").read_text())["status"] == "failed"


def test_host_credentials_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_PROVIDER_TOKEN", "must not appear")
    stage = make_stage(
        tmp_path,
        "import os,sys,pathlib; assert 'PRIVATE_PROVIDER_TOKEN' not in os.environ;pathlib.Path(sys.argv[1]).write_text('scientific result')",
    )
    assert CaseWorkflow(tmp_path / "run", {}).run_stage(stage)["status"] == "completed"
