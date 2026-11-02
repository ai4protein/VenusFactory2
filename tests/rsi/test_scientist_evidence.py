import pytest

from agent.graph.research import search


class Context:
    tool_history = []

    def add_tool_call(self, *args, **kwargs):
        pass


def state():
    return {
        "protein_context": Context(),
        "research_idx": 0,
        "search_idx": 0,
        "research_sections": [{"section_name": "Task", "search_queries": ["protein task"]}],
        "history": [],
        "ui_lang": "en",
    }


@pytest.mark.asyncio
async def test_injected_evidence_never_uses_global_search(monkeypatch):
    def forbidden(*args):
        raise AssertionError("global tool escaped restricted scope")

    monkeypatch.setattr(search, "_run_section_search", forbidden)
    calls = []

    async def evidence(query):
        calls.append(query)
        return ([{"source": "frozen task"}], [])

    result = await search.research_search_node(
        state(), {"configurable": {"chains": {"research_search_runner": evidence}}}
    )
    assert calls == ["protein task"] and result["current_search_results"] == [{"source": "frozen task"}]


@pytest.mark.asyncio
async def test_failed_restricted_evidence_has_no_global_fallback(monkeypatch):
    def forbidden(*args):
        raise AssertionError("global fallback escaped")

    monkeypatch.setattr(search, "_run_section_search", forbidden)

    async def broken(query):
        raise RuntimeError("restricted source unavailable")

    with pytest.raises(RuntimeError, match="restricted source unavailable"):
        await search.research_search_node(state(), {"configurable": {"chains": {"research_search_runner": broken}}})


def test_typed_mcp_arguments_remove_graph_context_fields():
    from langchain_core.tools import StructuredTool

    from agent.rsi.engine.scientist_runtime import argument_model

    tool = StructuredTool(
        name="inspect",
        description="Inspect task",
        args_schema=argument_model("Inspect", {"type": "object", "properties": {}}),
        func=lambda **kwargs: kwargs,
    )
    assert tool.invoke({"files": [], "last_sequence": None}) == {}
    schema = argument_model("Source", {"properties": {"source_code": {"type": "string"}}, "required": ["source_code"]})
    assert schema(source_code="actual source", files=[]).model_dump() == {"source_code": "actual source"}
