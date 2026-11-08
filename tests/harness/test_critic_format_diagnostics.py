"""Actionable critic format repair must not relax scientific acceptance."""
import json
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.autonomous_research import AutonomousResearch
from agent.harness.native_expert_team import critic_decision


@pytest.mark.parametrize("value,field", [
    ({"ready_for_host_validation": False, "summary": "Work pending", "issues": []}, "exactly decision"),
    ({"decision": False, "summary": "Work pending", "issues": []}, "decision must be"),
    ({"decision": ["needs_revision"], "summary": "Work pending", "issues": []}, "decision must be"),
    ({"decision": "needs_revision", "summary": " ", "issues": []}, "summary must be"),
    ({"decision": "needs_revision", "summary": "Work pending", "issues": "No measurements"}, "issues must be"),
])
def test_invalid_review_reports_the_field_without_reinterpreting_it(value, field):
    with pytest.raises(ValueError, match=field):
        critic_decision(json.dumps(value))


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [True, False])
async def test_format_repair_preserves_revision_and_raw_evidence(tmp_path, repair):
    invalid = json.dumps({"ready_for_host_validation": False,
                          "summary": "Training is still pending", "issues": ["No trained model"]})
    valid = {"decision": "needs_revision", "summary": "Training is still pending",
             "issues": ["No trained model"]}
    observed = []
    async def consult(role, context):
        assert role == "SC"
        observed.append(context)
        if len(observed) == 1:
            return invalid
        hint = context["format_repair"]
        assert hint["previous_response"] == invalid
        assert "exactly decision, summary and issues" in hint["validation_error"]
        return json.dumps(valid) if repair else invalid

    profile = Profile("science_expert", "expert-kimi-code",
        ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
        Budget(model_calls=12))
    driver = AutonomousResearch(None, "fixture", {}, consult=consult)
    request = driver.structured("SC", {"execution": {"receipts": []}}, profile,
                                SimpleNamespace(calls=0), tmp_path, critic_decision)
    if repair:
        assert await request == valid
    else:
        with pytest.raises(ValueError, match="exactly decision"):
            await request
    assert len(observed) == 2
    records = json.loads((tmp_path / "activity.json").read_text())
    assert records[0]["content"] == invalid and len(records) == 2
    assert all(row["role"] == "SC" for row in records)
