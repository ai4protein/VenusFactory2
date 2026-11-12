"""Research semantics are independent of the native Web permission policy."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from agent.harness.adapters import task_message
from agent.harness.contracts import Budget, ModelBinding, Profile, Task, digest
from agent.harness.research_workspace import ResearchWorkspace
from agent.rsi.engine.runtime import build_task_message


def task_at(tmp_path):
    paths = [tmp_path / name for name in ("workspace", "capability", "output")]
    for path in paths:
        path.mkdir()
    return Task(*paths, "Use frozen split A; compare validation only.",
                "Develop and review a useful predictor.", frozenset({"ask_user"}),
                mcp_entry="web_tools.py", phase="chat")


def test_explicit_research_interaction_changes_binding_without_permission_changes(tmp_path):
    original = task_at(tmp_path)
    research = replace(original, interaction="research_execution")
    assert original.interaction_mode == "chat"
    assert research.interaction_mode == "research_execution"
    assert original.phase == research.phase == "chat"
    assert research.tools == original.tools and research.mcp_entry == original.mcp_entry
    before, after = original.binding(), research.binding()
    assert "interaction" not in before
    assert after.pop("interaction") == "research_execution"
    assert before == after
    assert digest(original.binding()) != digest(research.binding())
    prompt = task_message(research)
    assert research.prompt in prompt and research.instructions in prompt
    assert "preserve progress and wait" in prompt
    assert "do not invent a persistent research goal" not in prompt
    assert "create a goal, ask questions" not in prompt
    assert "Use only the declared" not in task_message(original)  # Existing chat format.


@pytest.mark.parametrize("changes", [
    {"interaction": "unknown"},
    {"interaction": "research_execution", "phase": "develop"},
    {"interaction": "research_execution", "mcp_entry": "generated_tools.py"},
    {"interaction": "research_execution", "tools": frozenset()},
])
def test_research_interaction_does_not_reclassify_other_execution_surfaces(tmp_path, changes):
    with pytest.raises(ValueError, match="Research interaction requires explicit Web MCP tools"):
        replace(task_at(tmp_path), **changes)


def test_default_chat_and_bounded_messages_remain_unchanged():
    assert build_task_message("policy", "request", interaction="chat") == (
        "policy\nAnswer the user's current question or carry out the requested task. "
        "Use the available scientific tools only when useful. Direct answers and clarification questions "
        "are valid; do not invent a persistent research goal or require a file artifact. "
        "If information is missing, ask the user in your reply.\n\nrequest"
    )
    assert build_task_message("policy", "request") == (
        "Research task instructions (execute in this turn):\npolicy"
        "\nUse only the declared venusfactory MCP tools. Do not enter plan mode, "
        "create a goal, ask questions, or invoke native tools. Complete the task now.\n\nTask:\nrequest"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
async def test_research_asks_user_through_same_mcp_and_stops_subsequent_work(tmp_path, monkeypatch, harness):
    from agent.harness import web_native_worker as worker

    work, control = tmp_path / "work", tmp_path / "control"
    work.mkdir()
    control.mkdir()
    service = ResearchWorkspace(work, control / "tools", {"steps": []})
    events, cancelled = [], []

    async def adapter(ctx):
        assert ctx.task.interaction_mode == "research_execution"
        assert ctx.task.phase == "chat" and ctx.task.mcp_entry == "web_tools.py"
        prompt = task_message(ctx.task)
        assert "Use ask_user or request_confirmation" in prompt
        assert "do not invent a persistent research goal" not in prompt
        connection = json.loads((ctx.task.capability / "connection.json").read_text())
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=5,
                    headers={"Authorization": "Bearer " + connection["token"]}) as client:
                async def rpc(method, params):
                    response = await client.post(connection["url"],
                        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
                    return response.json()

                listing = await rpc("tools/list", {})
                names = {tool["name"] for tool in listing["result"]["tools"]}
                assert {"ask_user", "request_confirmation", "submit_research_checkpoint"} <= names
                assert not {"Bash", "exec_command", "evaluate_research_model"} & names
                denied = await rpc("tools/call", {"name": "Bash", "arguments": {"command": "pwd"}})
                assert "error" in denied
                question = await rpc("tools/call", {"name": "ask_user", "arguments": {
                    "questions": [{"id": "target", "question": "Which protein is the target?"}]}})
                assert "waiting_for_input" in json.dumps(question)
                denied = await rpc("tools/call", {"name": "write_session_file", "arguments": {
                    "path": "unexpected.txt", "content": "must not write after waiting"}})
                assert "error" in denied
                await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=adapter), None)))
    profile = Profile("science_agent", harness, ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                      Budget(tool_calls=3))
    await asyncio.wait_for(worker.run({"profile": profile.public(), "workspace": str(work),
        "control": str(control), "session_id": "research-input-fixture", "runtimes": {},
        "base_url": "http://127.0.0.1:1", "token": "fixture", "text": "Develop a useful model"},
        tools=service.tools(), event_sink=events.append), 20)
    assert cancelled and not (work / "unexpected.txt").exists()
    assert any(event["kind"] == "interaction" for event in events)
    assert not any(event["kind"] == "completed" for event in events)
    assert not (control / "native/capability/connection.json").exists()
