import json
from types import SimpleNamespace

import pytest

from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.tool_disclosure import ToolDisclosure, DisclosureError


def test_scientist_tools_are_discoverable_without_disclosing_schemas(tmp_path):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    enabled = {tool.name: tool for tool in service.tools()}
    catalog = ToolDisclosure(enabled, schema=lambda tool: tool.args_schema.model_json_schema(),
                             allowed=lambda tool: tool.name in enabled,
                             direct={"submit_research_checkpoint"})
    for name in ["get_research_strategy_guide", "evaluate_research_model",
                 "get_research_review_requirements", "propose_research_capability"]:
        page = catalog.search({"category": "training", "query": name})
        assert page["tools"][0]["name"] == name
        assert set(page["tools"][0]) == {"name", "description"}
        misplaced = catalog.search({"category": "rsi", "query": name})
        assert [c["id"] for c in misplaced["categories"]] == ["training"]
        assert not misplaced["tools"]
        with pytest.raises(DisclosureError):
            catalog.resolve("call_tool", {"name": name, "arguments": {}})
    enabled.pop("evaluate_research_model")
    assert not catalog.search({"category": "training", "query": "evaluate_research_model"})["tools"]


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
async def test_discovery_reports_checkpoint_budget_and_denied_calls_are_not_charged(tmp_path, monkeypatch, harness):
    import httpx
    from agent.harness import web_native_worker as worker
    from agent.harness.contracts import Profile, Budget, ModelBinding

    work, control = tmp_path / "work", tmp_path / "control"
    work.mkdir(); control.mkdir()
    service = ResearchWorkspace(work, control / "tools", {"steps": []})
    events = []

    async def adapter(ctx):
        connection = json.loads((ctx.task.capability / "connection.json").read_text())
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            async def call(name, args):
                response = await client.post(connection["url"],
                    headers={"Authorization": "Bearer " + connection["token"]},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})
                return response.json()
            rejected = await call("call_tool", {"name": "evaluate_research_model", "arguments": {}})
            assert rejected["result"]["isError"] is True
            assert events[-1]["kind"] == "research_tool_status"
            first_usage = next(e for e in events if e.get("kind") == "tool_usage")
            assert first_usage["tool_calls"] == 0
            assert first_usage["transport_calls"] == 1
            found = await call("search_tools", {"category": "training", "query": "evaluate_research_model"})
            assert "3 tool calls remain" in found["result"]["content"][-1]["text"]
            assert "5 read-only discovery calls remain" in found["result"]["content"][-1]["text"]
            described = await call("describe_tool", {"name": "evaluate_research_model"})
            assert "3 tool calls remain" in described["result"]["content"][-1]["text"]
            for _ in range(4):
                assert "result" in await call("search_tools", {})
            assert (await call("search_tools", {}))["result"]["isError"] is True
            assert ctx.profile.budget.tool_calls == 11  # 3 executions + 6 lookups + 2 checkpoint attempts
            saved = await call("submit_research_checkpoint",
                {"summary": "Need another iteration", "completed_steps": [], "artifacts": [], "needs_input": []})
            assert not saved["result"]["isError"]
            assert "Checkpoint saved" in saved["result"]["content"][-1]["text"]
            assert (await call("search_tools", {}))["result"]["isError"] is True
        return {"status": "completed", "text": "Partial progress saved."}

    monkeypatch.setattr(worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=adapter), None)))
    profile = Profile("science_agent", harness, ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                      Budget(tool_calls=3))
    await worker.run({"profile": profile.public(), "workspace": str(work), "control": str(control),
                     "session_id": "fixture", "runtimes": {}, "base_url": "http://127.0.0.1:1",
                     "token": "fixture", "text": "Research"}, tools=service.tools(), event_sink=events.append)
    assert service.checkpoint["summary"] == "Need another iteration"
    assert [e["tool_calls"] for e in events if e.get("kind") == "tool_usage"] == [0, 0, 0, 0, 0, 0, 0, 0]
    assert [e.get("checkpoint_calls", 0) for e in events if e.get("kind") == "tool_usage"][-1] == 1
    assert [e["transport_calls"] for e in events if e.get("kind") == "tool_usage"][-1] == 8
