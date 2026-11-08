"""Research saves survive exhausted execution budgets without granting more compute."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
from agent.harness import web_native_worker
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.research_workspace import ResearchWorkspace

def test_workspace_scan_tolerates_disappearing_atomic_save(tmp_path, monkeypatch):
    from agent.harness import session_sandbox
    (tmp_path / "result.json").write_text("{}")
    monkeypatch.setattr(session_sandbox.os, "walk", lambda *a, **kw: iter([
        (str(tmp_path), [], ["removed.tmp", "result.json"])]))
    session_sandbox.check_workspace(tmp_path)

@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_workspace_scan_still_rejects_links_and_special_files(tmp_path, kind):
    from agent.harness.session_sandbox import check_workspace
    original = tmp_path / "data.json"; original.write_text("{}")
    target = tmp_path / "unsafe"
    if kind == "symlink": target.symlink_to(original)
    elif kind == "hardlink": os.link(original, target)
    else: os.mkfifo(target)
    with pytest.raises(ValueError):
        check_workspace(tmp_path)

@pytest.mark.asyncio
@pytest.mark.parametrize("broker", [False, True])
async def test_checkpoint_can_be_repaired_after_execution_allowance(tmp_path, monkeypatch, broker):
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {})
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "tools", {"steps": [{"id": "inspect"}]})
    observed = []
    (tmp_path / "control").mkdir()
    class Client:
        async def run(self, context):
            connection = json.loads((context.task.capability / "connection.json").read_text())
            async with httpx.AsyncClient(trust_env=False) as client:
                async def invoke(name, args):
                    if broker:
                        name, args = "call_tool", {"name": name, "arguments": args}
                    response = await client.post(connection["url"],
                        headers={"Authorization": "Bearer " + connection["token"]},
                        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": name, "arguments": args}})
                    return response.json()
                for name in ("result.json", "notes.md"):
                    result = await invoke("write_session_file", {"path": name, "content": "observed"})
                    assert not result.get("error"), result
                denied = await invoke("write_session_file", {"path": "extra.txt", "content": "must not run"})
                assert denied["result"]["isError"] is True
                assert "Execution budget reached" in denied["result"]["content"][0]["text"]
                bad = await invoke("submit_research_checkpoint", {
                    "summary": "Partial work", "completed_steps": ["invented"],
                    "artifacts": ["result.json"], "needs_input": []})
                assert bad["result"]["isError"] is True and not service.checkpoint
                saved = await invoke("submit_research_checkpoint", {
                    "summary": "Partial work", "completed_steps": [],
                    "artifacts": ["result.json"], "needs_input": []})
                assert not saved.get("error"), saved
                assert service.checkpoint["completed_steps"] == []
                denied = await invoke("write_session_file", {"path": "after.txt", "content": "must not run"})
                assert denied["result"]["isError"] is True
                assert "Checkpoint saved" in denied["result"]["content"][0]["text"]
            return {"status": "completed", "text": "Partial work saved."}
    monkeypatch.setattr(web_native_worker, "builtin_registry",
                        lambda: SimpleNamespace(resolve=lambda *a: (Client(), None)))
    profile = Profile("science_agent", "deepseek-harness",
                      ModelBinding("fixture", "unused", "http://127.0.0.1:1/v1"), Budget(tool_calls=2))
    await web_native_worker.run({"session_id": "checkpoint-fixture", "workspace": str(service.workspace),
        "control": str(tmp_path / "control"), "profile": profile.public(), "runtimes": {},
        "base_url": "http://127.0.0.1:1", "token": "unused", "text": "Save partial work"},
        tools=service.tools(), event_sink=observed.append)
    assert not (service.workspace / "extra.txt").exists()
    assert not (service.workspace / "after.txt").exists()
    completed = next(event for event in observed if event["kind"] == "completed")
    assert completed["tool_calls"] == 2 and completed["checkpoint_calls"] == 2
