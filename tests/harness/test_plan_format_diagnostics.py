"""Repair plan schema mistakes without dropping scientific requirements."""
import copy
import json
from types import SimpleNamespace

import pytest

from agent.harness.autonomous_research import AutonomousResearch, parse_plan


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [True, False])
async def test_step_level_needs_input_reports_its_location_and_keeps_evidence(tmp_path, repair):
    plan = {"summary": "Train a measured predictor", "deliverable": "model",
            "steps": [{"id": "train", "title": "Fit and evaluate",
                       "acceptance": "Save trained weights and held-out predictions",
                       "needs_input": []}], "needs_input": []}
    invalid = json.dumps(plan)
    fixed = copy.deepcopy(plan)
    fixed["steps"][0].pop("needs_input")
    observed = []
    async def consult(role, context):
        observed.append(context)
        if len(observed) == 1:
            return invalid
        hint = context["format_repair"]
        assert hint["previous_response"] == invalid
        assert "Step 1" in hint["validation_error"]
        assert "unsupported fields: needs_input" in hint["validation_error"]
        assert "top level" in hint["validation_error"]
        return json.dumps(fixed) if repair else invalid

    runner = AutonomousResearch(None, "fixture", {}, consult=consult)
    request = runner.structured("CB", {"objective": "Train a protein predictor"},
                               SimpleNamespace(budget=SimpleNamespace(model_calls=12)),
                               SimpleNamespace(calls=0), tmp_path, parse_plan)
    if repair:
        result = await request
        assert result["steps"][0]["acceptance"] == plan["steps"][0]["acceptance"]
        assert result["steps"][0]["status"] == "pending"
        assert result["steps"][0]["evidence"] == []
    else:
        with pytest.raises(ValueError, match="unsupported fields"):
            await request
    assert len(observed) == 2
    assert json.loads((tmp_path / "activity.json").read_text())[0]["content"] == invalid
