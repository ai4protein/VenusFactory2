"""Selected Scientist tools expose complete contracts without expanding the catalogue."""
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.tool_disclosure import ToolDisclosure


def test_only_selected_enabled_tools_receive_full_descriptions():
    mapping = {f"tool_{i}": SimpleNamespace(
        name=f"tool_{i}", description=f"Short description {i}.\n\nPrivate detail {i}.",
        metadata={"discovery_category": "compute"}) for i in range(1000)}
    catalog = ToolDisclosure(mapping, schema=lambda tool: {"type": "object", "properties": {}},
                             allowed=lambda tool: tool.name != "tool_2",
                             direct={"tool_0", "tool_1", "tool_2"}, detailed={"tool_0", "tool_2"})
    surface = {tool["name"]: tool for tool in catalog.surface()}
    assert surface["tool_0"]["description"] == mapping["tool_0"].description
    assert surface["tool_1"]["description"] == "Short description 1."
    assert "tool_2" not in surface and "tool_999" not in surface
    page = catalog.search({"category": "compute", "query": "tool_999"})
    assert page["tools"][0]["description"] == "Short description 999."
    assert "inputSchema" not in page["tools"][0]


@pytest.mark.parametrize("direct,detailed", [
    ({"tool_0"}, {"tool_1"}),
    ({"tool_0", "tool_1", "tool_2", "tool_3"}, {"tool_0", "tool_1", "tool_2", "tool_3"})])
def test_full_contracts_require_bounded_admitted_selection(direct, detailed):
    mapping = {name: SimpleNamespace(name=name, description="Example")
               for name in direct | detailed}
    with pytest.raises(ValueError, match="at most three admitted direct tools"):
        ToolDisclosure(mapping, schema=lambda tool: {}, allowed=lambda tool: True,
                       direct=direct, detailed=detailed)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
async def test_harness_receives_complete_selected_training_contract(tmp_path, monkeypatch, harness):
    from agent.harness import web_native_worker as worker
    from agent.harness.contracts import Profile, Budget, ModelBinding

    work, control = tmp_path / "work", tmp_path / "control"
    work.mkdir(); control.mkdir()
    service = ResearchWorkspace(work, control / "tools", {"steps": []})
    expected = next(tool.description for tool in service.tools() if tool.name == "train_research_model")

    async def adapter(ctx):
        connection = json.loads((ctx.task.capability / "connection.json").read_text())
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            headers = {"Authorization": "Bearer " + connection["token"]}
            response = await client.post(connection["url"], headers=headers,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
            surface = {tool["name"]: tool for tool in response.json()["result"]["tools"]}
            train = surface["train_research_model"]
            assert train["description"] == expected
            assert 'training["train"]' in train["description"]
            assert "host saves fitted state and evaluates held-out data in isolation" in train["description"]
            assert "not the dataset" in train["inputSchema"]["properties"]["source_file"]["description"]
            assert "evaluate_research_model" not in surface
            response = await client.post(connection["url"], headers=headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "params": {"name": "list_session_files", "arguments": {}}})
            assert "result" in response.json()
        return {"status": "completed", "text": "The selected training contract is available."}

    monkeypatch.setattr(worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=adapter), None)))
    profile = Profile("science_agent", harness, ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                      Budget(tool_calls=3))
    await worker.run({"profile": profile.public(), "workspace": str(work), "control": str(control),
                     "research_focus": ["train_research_model"], "session_id": "fixture",
                     "runtimes": {}, "base_url": "http://127.0.0.1:1", "token": "fixture",
                     "text": "Develop a predictor"}, tools=service.tools())
