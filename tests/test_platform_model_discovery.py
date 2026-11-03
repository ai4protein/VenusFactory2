"""Plaza and all native harnesses must discover the same enabled protein models."""
import json
from types import SimpleNamespace
import pytest
from mcp_tools.models import platform_catalog
from mcp_tools.tools.discovery import service
from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
from mcp_tools.tools import runtime_tool_policy as policy
from mcp_tools.tools.capability_policy import CapabilityStore, selector
from agent.harness.tool_disclosure import ToolDisclosure
from agent.harness.web_native_worker import web_tool_name


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(policy, "is_online_mode", lambda: True)
    from mcp_tools.models.pretrained.design_docking import service as external
    monkeypatch.setattr(external, "availability", lambda name: (name not in {"bindcraft","boltzgen"}, None))
    from agent.harness import research_workspace
    monkeypatch.setattr(research_workspace, "capability_cards", lambda: [])


def tools():
    return [
        SimpleNamespace(name="zero_shot_mutation_structure_prediction", description="Score mutations from a structure.\n\nParameters: model_name selects ProSST or ESM-IF1.",
                        parameters={"type":"object","properties":{"model_name":{"type":"string"},"pdb_file":{"type":"string"}}}),
        SimpleNamespace(name="ligandmpnn_design", description="Design sequences conditioned on a ligand.",
                        parameters={"type":"object","properties":{"pdb_file":{"type":"string"}}})
    ]


@pytest.mark.parametrize("harness", ["kimi-code","deepseek-harness","codex","claude-code"])
def test_named_models_search_describe_and_route_to_original_tool(harness):
    mapping = {web_tool_name(t.name,harness):t for t in tools()}
    disclosure = ToolDisclosure(mapping, schema=lambda t:t.parameters, allowed=lambda t:is_tool_allowed(t.name))
    for query, expected in [("ProSST-2048","zero_shot_mutation_structure_prediction"),("LigandMPNN","ligandmpnn_design")]:
        result = disclosure.metadata("search_tools", {"query":query})
        name = web_tool_name(expected,harness)
        assert name in {x["name"] for x in result["tools"]}
        assert len(json.dumps(result)) < 1500
        detail = disclosure.metadata("describe_tool", {"name":name})
        if query == "ProSST-2048":
            assert detail["matching_models"] == [{"name":"ProSST-2048","arguments":{"model_name":"ProSST-2048"}}]
            assert disclosure.resolve("call_tool", {"name":name,"arguments":detail["matching_models"][0]["arguments"]})[2] == {"model_name":"ProSST-2048"}
        assert disclosure.resolve("call_tool", {"name":name,"arguments":{}})[1].name == expected
        assert disclosure.metadata("describe_tool", {"name":expected})["name"] == name
        assert disclosure.resolve("call_tool", {"name":expected,"arguments":{}})[1].name == expected


def test_online_inventory_matches_plaza_and_honors_live_variant_switch():
    from backend.web_v2.model_platform import Platform
    catalog = [dict(name=t.name, description=t.description, parameters=t.parameters) for t in tools()]
    rows = service.search("ProSST", kind="models", tool_catalog=catalog)
    plaza = [c for c in Platform.official(None) if c.get("available") and c.get("model","").startswith("ProSST-")]
    assert {r["entry_id"] for r in rows["results"]} == {c["id"] for c in plaza}
    assert rows["model_catalog_status"] == "available"
    card = next(c for c in plaza if c["name"] == "ProSST-2048")
    CapabilityStore().set("model:"+card["id"],False,0,selector(card),"fixture")
    assert not service.search("ProSST-2048",kind="models",tool_catalog=catalog)["results"]
    ligand = service.search("LigandMPNN", kind="models",tool_catalog=catalog)["results"]
    assert len(ligand) == 1 and ligand[0]["tools"] == ["ligandmpnn_design"]
    CapabilityStore().set("tool:ligandmpnn_design",False,0,{},"fixture")
    assert not service.search("LigandMPNN",kind="models",tool_catalog=catalog)["results"]


def test_unmounted_models_are_not_offered():
    assert not service.search("LigandMPNN",kind="models",tool_catalog=[])["results"]


def test_short_description_does_not_limit_search_index():
    tool=SimpleNamespace(name="unfamiliar_tool", description="Generic title.\n\nSupports specialized kidney assay.")
    d=ToolDisclosure({tool.name:tool},schema=lambda t:{"properties":{"method":{"enum":["rare_model"]}}},allowed=lambda t:True)
    for query in ("kidney", "rare_model"):
        assert d.metadata("search_tools",{"query":query})["tools"] == [{"name":tool.name,"description":"Generic title."}]


def test_skill_discovery_is_bounded_and_prompt_does_not_grow(monkeypatch):
    from agent import skills
    rows=[{"skill_id":"model_"+str(i),"description":"Test a protein model "*300,"source":"test"} for i in range(10000)]
    monkeypatch.setattr(skills,"get_skills_metadata",lambda:rows)
    result=skills.build_list_skills_response("protein",limit=3)
    assert result["total"] == 10000 and len(result["items"]) == 3 and result["next_offset"] == 3
    assert len(json.dumps(result)) < 2000
    assert len(skills.get_skills_catalog_for_agent()) < 500
    assert skills.build_list_skills_response(limit=100)["success"] is False
