"""Historical default-role requests remain verifiable after strategy support."""

import json

import pytest
from test_native_expert_audit import completed as completed
from test_native_expert_audit import critic_outputs as critic_outputs
from test_native_expert_audit import critic_prefix as critic_prefix
from test_native_expert_audit import historical_revision as historical_revision
from test_native_expert_audit import native_artifact as native_artifact
from test_native_expert_audit import role_budget as role_budget

from agent.harness import historical_tool_audit as historical
from agent.harness.contracts import digest
from agent.harness.host import write_json
from agent.harness.native_expert_audit import audit_native_expert


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_revision", ["_legacy_pre_tool_policy"], indirect=True)
@pytest.mark.parametrize("attack", [None, "unknown_source", "capsule", "artifact"])
async def test_pre_tool_policy_capsule_preserves_default_role_evidence(completed, monkeypatch, attack):
    task, result = completed
    lock = json.loads((task.output / "profile.lock.json").read_text())
    for name, sha in historical.PINNED.items():
        if name.startswith("harness/"):
            lock["adapter_source"][name.removeprefix("harness/")] = sha
    if attack == "unknown_source":
        lock["adapter_source"]["native_expert_audit.py"] = "0" * 64
    lock["sha256"] = digest({k: v for k, v in lock.items() if k != "sha256"})
    write_json(task.output / "profile.lock.json", lock)
    result["profile_sha256"] = lock["sha256"]
    if attack == "capsule":
        monkeypatch.setattr(historical, "file_hash", lambda path: "0" * 64)
    elif attack == "artifact":
        write_json(task.workspace / "probe.json", {"length": 999})
    if attack:
        with pytest.raises(ValueError):
            audit_native_expert(task.output, task.workspace, task.capability, result=result)
    else:
        assert audit_native_expert(task.output, task.workspace, task.capability, result=result)["passed"]
