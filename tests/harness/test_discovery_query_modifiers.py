"""Real progressive navigation must reach relevant bounded catalog results."""
import json

import pytest

from agent.harness.tool_disclosure import ToolDisclosure
from mcp_tools.tools import runtime_tool_policy
from mcp_tools.tools.capability_policy import CapabilityStore, canonical
from mcp_tools.tools.discovery import service
from mcp_tools.tools.discovery.matching import query_groups, relevance
from mcp_tools.tools.discovery.tools_agent import search_research_tools_tool


@pytest.fixture(autouse=True)
def isolated_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from agent.harness import research_workspace
    monkeypatch.setattr(research_workspace, "LIBRARY", tmp_path / "published")


def navigate(query, category="compute"):
    # These are the real navigation tool and schema. The inner tool uses
    # available_tools()/get_tools(), not a fabricated catalog or patched search.
    tool = search_research_tools_tool
    disclosure = ToolDisclosure({tool.name: tool},
        schema=lambda item: item.tool_call_schema.model_json_schema(),
        allowed=lambda item: runtime_tool_policy.is_tool_allowed(item.name))
    page = disclosure.metadata("search_tools", {"category": category, "query": query})
    assert page["match_mode"] == "category_navigation"
    detail = disclosure.metadata("describe_tool", page["next_action"]["arguments"])
    assert detail["suggested_arguments"] == {"query": query}
    return json.loads(tool.invoke({**detail["suggested_arguments"], "kind": "tools"}))


@pytest.mark.parametrize("mode", ["local", "online"])
@pytest.mark.parametrize("query", ["sequence analysis", "sequences analyses", "蛋白质序列分析"])
def test_navigation_suggestion_finds_physchem_in_real_default_catalog_page(monkeypatch, mode, query):
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    result = navigate(query)
    names = [row["name"] for row in result["results"]]
    assert result["success"] is True
    assert "calculate_physchem_from_sequence" in names
    assert len(names) <= 10  # Actual inner-tool default page size.
    assert result["matching_models"] == 0
    assert all(row["kind"] == "tool" for row in result["results"])
    assert result["matching_tools"] < result["available_tools"]


@pytest.mark.parametrize("query", ["sequence TotallyUnknownProteinModel analysis",
                                   "TotallyUnknownProteinModel-2048 analysis",
                                   "not_a_registered_sequence_analysis_tool"])
def test_unknown_named_constraint_never_falls_back_to_general_tools(query):
    result = json.loads(search_research_tools_tool.invoke({"query": query, "kind": "tools"}))
    assert result["results"] == []
    assert result["matching_tools"] == 0


def test_exact_tool_id_is_preserved_and_ranks_before_descriptive_mentions():
    identity = "calculate_physchem_from_sequence"
    assert query_groups(identity) == [(identity,)]
    exact = relevance(query_groups(identity), identity.replace("_", " "), identity)
    mention = relevance(query_groups(identity), "another tool", "Uses " + identity + " output.")
    assert exact > mention
    result = json.loads(search_research_tools_tool.invoke({"query": identity, "kind": "tools"}))
    assert result["results"][0]["name"] == identity


@pytest.mark.parametrize("query", ["analysis", "analyses", "分析"])
def test_generic_descriptor_only_query_does_not_expand_to_all_catalog_entries(query):
    groups = query_groups(query)
    assert groups
    assert relevance(groups, "Solubility predictor", "Predict soluble probability") is None
    result = json.loads(search_research_tools_tool.invoke({"query": query, "kind": "tools", "limit": 3}))
    assert len(result["results"]) <= 3
    assert result["matching_tools"] < result["available_tools"]


def test_live_tool_disable_remains_effective_on_next_navigation_request():
    assert "calculate_physchem_from_sequence" in [row["name"] for row in navigate("sequence analysis")["results"]]
    CapabilityStore().set("tool:" + canonical("calculate_physchem_from_sequence"), False, 0, {}, "fixture")
    assert "calculate_physchem_from_sequence" not in [row["name"] for row in navigate("sequence analysis")["results"]]


def test_category_filter_does_not_return_unrelated_compute_tools():
    from mcp_tools.tools.tools_agent_hub import get_tools
    mapping = {tool.name: tool for tool in get_tools()}
    disclosure = ToolDisclosure(mapping,
        schema=lambda item: item.tool_call_schema.model_json_schema(),
        allowed=lambda item: runtime_tool_policy.is_tool_allowed(item.name))
    page = disclosure.metadata("search_tools", {"category": "databases",
        "query": "calculate_physchem_from_sequence"})
    assert "calculate_physchem_from_sequence" not in {row["name"] for row in page.get("tools", [])}


def test_explicit_kind_filter_and_exact_model_name_remain_distinct():
    # A real platform catalog selection; mounted adapters constrain the results.
    tools = service.available_tools()
    catalog = [{"name": tool.name, "description": tool.description,
                "parameters": tool.tool_call_schema.model_json_schema()} for tool in tools]
    models = service.search("ProSST-2048", kind="models", tool_catalog=catalog)
    assert models["matching_tools"] == 0
    assert models["results"]
    assert all(row["kind"] != "tool" for row in models["results"])
    assert any(row.get("name") == "ProSST-2048" for row in models["results"])
    tools_only = service.search("sequence analysis", kind="tools", tool_catalog=catalog)
    assert all(row["kind"] == "tool" for row in tools_only["results"])
    unknown = service.search("ProSST-999999", kind="models", tool_catalog=catalog)
    assert unknown["results"] == []


@pytest.mark.parametrize("identity", ["sequence_analysis_v2", "analysis-model-v3", "ESM2-650M", "model.version-1"])
def test_compound_ids_are_not_split_by_alias_or_descriptor_groups(identity):
    assert query_groups(identity) == [(identity.lower(),)]


def test_descriptor_does_not_remove_substantive_and_constraints():
    assert relevance(query_groups("sequence analysis unknown-scope"), "Sequence tool",
                     "Calculate protein sequence properties") is None
    assert relevance(query_groups("sequence analysis"), "Sequence tool",
                     "Calculate protein sequence properties") is not None


def test_hyphenated_sequence_descriptor_reaches_real_default_catalog_page():
    result = navigate("sequence-analysis")
    assert "calculate_physchem_from_sequence" in {row["name"] for row in result["results"]}
    assert len(result["results"]) <= 10


def test_hyphenated_stability_descriptor_keeps_substantive_alias_constraint():
    groups = query_groups("stability-analysis")
    assert groups == query_groups("stability analysis")
    assert relevance(groups, "Thermostability", "Predict thermal stability") is not None
    assert relevance(groups, "Sequence properties", "Calculate protein sequence properties") is None
