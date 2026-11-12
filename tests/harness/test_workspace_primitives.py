"""Scoped file operations are usable without loading scientific catalogs."""
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.harness.contracts import Profile, Budget, ModelBinding
from agent.harness.research_workspace import ResearchWorkspace
from mcp_tools.tools.tool_surface import WORKSPACE_PRIMITIVES


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
async def test_native_workspace_primitives_keep_boundaries(tmp_path, monkeypatch, harness):
    from agent.harness import web_native_worker as worker

    workspace = tmp_path / "work"
    control = tmp_path / "control"
    workspace.mkdir()
    control.mkdir()
    (tmp_path / "other-user.txt").write_text("outside workspace")
    service = ResearchWorkspace(workspace, control / "tools", {"steps": []})
    disabled = set()
    from mcp_tools.tools import runtime_tool_policy
    original_allowed = runtime_tool_policy.assert_tool_allowed

    def available(name):
        if name in disabled:
            raise PermissionError("disabled")
        return original_allowed(name)

    monkeypatch.setattr(runtime_tool_policy, "assert_tool_allowed", available)

    async def adapter(ctx):
        connection = json.loads((ctx.task.capability / "connection.json").read_text())
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            async def rpc(method, params):
                response = await client.post(connection["url"],
                    headers={"Authorization": "Bearer " + connection["token"]},
                    json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
                return response.json()

            async def call(name, arguments):
                return await rpc("tools/call", {"name": name, "arguments": arguments})

            surface = (await rpc("tools/list", {}))["result"]["tools"]
            names = {tool["name"] for tool in surface}
            assert WORKSPACE_PRIMITIVES <= names
            assert len(names) <= 11
            assert "evaluate_research_model" not in names
            assert "run_session_shell" not in names
            assert (await call("call_tool", {"name": "evaluate_research_model",
                                            "arguments": {}}))["result"]["isError"] is True
            saved = await call("write_session_file", {"path": "sum.py",
                "content": "import json\nfrom pathlib import Path\nPath('result.json').write_text(json.dumps({'total': sum([2, 4, 7])}))\n"})
            assert not saved["result"]["isError"]
            read = await call("read_session_file", {"path": "sum.py"})
            assert not read["result"]["isError"]
            version = json.JSONDecoder().raw_decode(read["result"]["content"][0]["text"])[0]["sha256"]
            arguments = {"path":"sum.py", "old_text":"sum([2, 4, 7])", "new_text":"sum([2, 4, 8])", "expected_sha256":version}
            edited = await call("edit_session_file", arguments)
            assert not edited["result"]["isError"]
            assert (await call("edit_session_file", arguments))["result"]["isError"] is True
            executed = await call("run_session_python", {"path": "sum.py", "timeout_seconds": 30})
            assert not executed["result"]["isError"]
            assert json.loads((workspace / "result.json").read_text()) == {"total": 14}
            assert (await call("read_session_file", {"path": "../other-user.txt"}))["result"]["isError"] is True
            assert (await call("write_session_file", {"path": "../other-user.txt", "content": "overwritten"}))["result"]["isError"] is True
            assert (await call("edit_session_file", {**arguments, "path":"../other-user.txt"}))["result"]["isError"] is True
            assert (tmp_path / "other-user.txt").read_text() == "outside workspace"
            disabled.add("edit_session_file")
            assert "edit_session_file" not in {tool["name"] for tool in (await rpc("tools/list", {}))["result"]["tools"]}
            assert (await call("edit_session_file", arguments))["result"]["isError"] is True
            disabled.add("write_session_file")
            assert "write_session_file" not in {tool["name"] for tool in
                (await rpc("tools/list", {}))["result"]["tools"]}
            assert (await call("write_session_file", {"path": "denied.txt", "content": "test"}))["result"]["isError"] is True
            assert not (workspace / "denied.txt").exists()
        return {"status": "completed", "text": "Checked actual outputs."}

    monkeypatch.setattr(worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=adapter), None)))
    profile = Profile("science_agent", harness, ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                      Budget(tool_calls=16))
    await worker.run({"profile": profile.public(), "workspace": str(workspace), "control": str(control),
        "session_id": "fixture", "runtimes": {}, "base_url": "http://127.0.0.1:1",
        "token": "fixture", "text": "Analyze these data"}, tools=service.tools(), event_sink=lambda event: None)
