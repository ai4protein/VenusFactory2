"""Only identical navigation guidance moves to the existing native instructions."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from agent.harness import web_native_worker
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.research_context_projection import project_research_context
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.session_boundary import SECURITY_INSTRUCTIONS
from agent.harness.tool_disclosure import GUIDANCE


def context(guidance=GUIDANCE):
    return {"objective": "Repair the measured model without changing its validation protocol",
            "available_tools": {"count": 17, "categories": [{"id": "models", "description": "Use enabled tools"}],
                                "discovery": guidance},
            "model_development": {"repair_source": {"code": "def fit(training, seed): return {}",
                                 "sha256": "a" * 64, "source_check": {"valid": False}}},
            "progress": {"execution_feedback": {"pending": [{"error": "Missing predict entrypoint"}]},
                         "sc_revision": {"issues": ["No measured revision"]}}}


def test_identical_discovery_guidance_is_audited_without_reference_files():
    original = context()
    before = copy.deepcopy(original)
    result = project_research_context(original, inline_discovery_guidance=GUIDANCE)
    expected = copy.deepcopy(original)
    expected["available_tools"].pop("discovery")
    assert result.context == expected and original == before
    assert result.reference_payloads == {}
    assert result.instruction_replacements == {"/available_tools/discovery": {
        "sha256": hashlib.sha256(GUIDANCE.encode()).hexdigest(), "bytes": len(GUIDANCE.encode()),
        "replacement": "task.instructions"}}
    assert len(json.dumps(result.context)) < len(json.dumps(original))


@pytest.mark.parametrize("inline", [None, "", GUIDANCE + " ", GUIDANCE.replace("Discover", "Find", 1)])
def test_missing_or_different_guidance_never_removes_context(inline):
    original = context()
    result = project_research_context(original, inline_discovery_guidance=inline)
    assert result.context == original
    assert result.reference_payloads == result.instruction_replacements == {}


@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
@pytest.mark.parametrize("case", ["research_identical", "research_different", "agent"])
@pytest.mark.asyncio
async def test_actual_native_task_keeps_one_authoritative_copy_and_security(tmp_path, monkeypatch, harness, case):
    for key, value in {"VENUS_RUNTIME_ROOT": tmp_path / "runtime", "VENUS_JOB_QUEUE_ROOT": tmp_path / "queue",
                       "VENUS_USAGE_DB": tmp_path / "usage.sqlite", "VENUS_ACCESS_DB": tmp_path / "access.sqlite",
                       "PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                       "WEBUI_V2_MODE": "local", "VENUS_SCIENTIST_ONLINE": "0"}.items():
        monkeypatch.setenv(key, str(value))
    for name in ("VENUS_QUEUE_JOB_ID", "VENUS_QUEUE_OWNER", "VENUS_QUEUE_LEASE_TOKEN",
                 "VENUS_CAPABILITY_POLICY_SNAPSHOT", "VENUS_MODEL_RUNTIMES_CONFIG", "VENUS_SCIENTIST_PBS_RECORD"):
        monkeypatch.delenv(name, raising=False)
    source = context(GUIDANCE + " Keep this additional scientific requirement." if case == "research_different" else GUIDANCE)
    workspace, control = tmp_path / "workspace", tmp_path / "control"
    service = ResearchWorkspace(workspace, control / "tools", {"steps": []})
    tools = service.tools()
    if case == "agent":
        tools = [tool for tool in tools if tool.name != "submit_research_checkpoint"]
    profile = Profile("science_agent", harness, ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                      Budget(model_calls=2, tool_calls=4, wall_seconds=30))
    request = {"session_id": "guidance-fixture", "workspace": str(workspace), "control": str(control),
               "profile": profile.public(), "budget": profile.public()["budget"], "messages": [],
               "runtimes": {}, "base_url": "http://127.0.0.1:1", "token": "synthetic-relay-only",
               "text": json.dumps(source), "instructions": "Preserve the scientific objective."}
    before = copy.deepcopy(request)
    observed = []

    async def adapter(runtime):
        assert runtime.profile.harness == harness
        assert runtime.task.instructions.count(GUIDANCE) == 1
        assert runtime.task.instructions.count(SECURITY_INSTRUCTIONS) == 1
        actual = json.loads(runtime.task.prompt.rsplit("## Current user request\n", 1)[1])
        expected = copy.deepcopy(source)
        if case == "research_identical":
            expected["available_tools"].pop("discovery")
        assert actual == expected
        assert runtime.task.interaction == (None if case == "agent" else "research_execution")
        return {"status": "completed", "text": "Native context inspected."}

    monkeypatch.setattr(web_native_worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=adapter), None)))
    await web_native_worker.run(request, tools=tools, event_sink=observed.append)
    assert request == before
    audit_path = control / "native/context-projection.json"
    if case == "research_identical":
        audit = json.loads(audit_path.read_text())
        assert audit["reference_files"] == {}
        assert audit["instruction_replacements"]["/available_tools/discovery"] == {
            "sha256": hashlib.sha256(GUIDANCE.encode()).hexdigest(), "bytes": len(GUIDANCE.encode()),
            "replacement": "task.instructions"}
        assert audit["original_sha256"] == hashlib.sha256(before["text"].encode()).hexdigest()
        assert request["token"] not in json.dumps(audit)
    else:
        assert not audit_path.exists()
    completed = next(event for event in observed if event["kind"] == "completed")
    assert completed["messages"][0]["content"] == before["text"]
