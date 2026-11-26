"""Plaza and all native harnesses must discover the same enabled protein models."""
import json
from types import SimpleNamespace
import pytest
from mcp_tools.models import platform_catalog
from mcp_tools.tools.discovery import service
from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
from mcp_tools.tools import runtime_tool_policy as policy
from mcp_tools.tools.capability_policy import CapabilityStore, selector
from agent.harness.tool_disclosure import ToolDisclosure, DisclosureError
from agent.harness.web_native_worker import web_tool_name


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(policy, "is_online_mode", lambda: True)
    from mcp_tools.models.pretrained.design_docking import service as external
    monkeypatch.setattr(external, "availability", lambda name: (name not in {"bindcraft","boltzgen"}, None))
    from agent.harness import research_workspace
    monkeypatch.setattr(research_workspace, "LIBRARY", tmp_path / "published-capabilities")


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
        call_args = {"model_name":"ProSST-2048"} if query == "ProSST-2048" else {}
        assert disclosure.resolve("call_tool", {"name":name,"arguments":call_args})[1].name == expected
        assert disclosure.metadata("describe_tool", {"name":expected})["name"] == name
        assert disclosure.resolve("call_tool", {"name":expected,"arguments":call_args})[1].name == expected


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


def test_discovery_index_uses_only_name_and_summary():
    tool = SimpleNamespace(name="unfamiliar_tool", description="Generic title.\n\nSupports specialized kidney assay.", metadata={"discovery_category": "compute"})
    d = ToolDisclosure({tool.name: tool}, schema=lambda t: {"properties": {"method": {"enum": ["rare_model"]}}}, allowed=lambda t: True)
    for query in ("kidney", "rare_model"):
        assert d.metadata("search_tools", {"category": "compute", "query": query})["tools"] == []
    assert d.metadata("search_tools", {"category": "compute", "query": "unfamiliar_tool"})["tools"] == [
        {"name": tool.name, "description": "Generic title."}]


def test_skill_discovery_is_bounded_and_prompt_does_not_grow(monkeypatch):
    from agent import skills
    rows=[{"skill_id":"model_"+str(i),"description":"Test a protein model "*300,"source":"test"} for i in range(10000)]
    monkeypatch.setattr(skills,"get_skills_metadata",lambda:rows)
    result=skills.build_list_skills_response("protein",limit=3)
    assert result["total"] == 10000 and len(result["items"]) == 3 and result["next_offset"] == 3
    assert len(json.dumps(result)) < 2000
    assert len(skills.get_skills_catalog_for_agent()) < 500
    assert skills.build_list_skills_response(limit=100)["success"] is False

def test_followup_description_preserves_live_models_without_prior_search():
    tool = tools()[0]
    tool.parameters["properties"]["model_name"].update(default="ESM-IF1", description="Default: ESM-IF1")
    d = ToolDisclosure({tool.name:tool}, schema=lambda t:t.parameters, allowed=lambda t:True)
    detail = d.metadata("describe_tool", {"name":tool.name})
    assert any(m["arguments"] == {"model_name":"ProSST-2048"} for m in detail["matching_models"])
    assert "default" not in detail["inputSchema"]["properties"]["model_name"]
    assert "model_name" in detail["inputSchema"]["required"]
    assert tool.parameters["properties"]["model_name"]["default"] == "ESM-IF1"
    assert "Parameters:" not in detail["description"]
    with pytest.raises(DisclosureError, match="explicitly"):
        d.resolve("call_tool", {"name":tool.name, "arguments":{"pdb_file":"test.pdb"}})
    assert d.resolve("call_tool", {"name":tool.name, "arguments":{"pdb_file":"test.pdb","model_name":"ProSST-2048"}})[2]["model_name"] == "ProSST-2048"


@pytest.mark.parametrize("model_field,dataset_field", [
    ("model_name", "dataset_name"), ("model", "dataset"),
])
def test_model_choices_follow_declared_fields_without_changing_original_schema(
    monkeypatch, model_field, dataset_field
):
    from mcp_tools.tools.discovery import platform_models as registry

    name = "predict_custom_model"
    schema = {
        "type": "object",
        "properties": {
            model_field: {"type": "string", "enum": ["ESM2-8M"], "default": "legacy-default"},
            dataset_field: {"type": "string", "enum": ["solubility"]},
            "task": {"type": "string", "enum": ["regression"]},
            "sequence": {"type": "string"},
        },
        "required": ["sequence"],
    }
    original = json.loads(json.dumps(schema))
    card = {
        "entry_id": "fixture-solubility",
        "name": "Solubility · ESM2-8M",
        "selection": {
            "model": "ESM2-8M", "dataset": "solubility", "task": "regression",
            "repository": "metadata-is-not-an-input",
        },
    }
    monkeypatch.setattr(registry, "platform_models", lambda query, names: [(1, card["entry_id"], card)])
    tool = SimpleNamespace(name=name, description="Predict solubility.")
    disclosure = ToolDisclosure({name: tool}, schema=lambda _: schema, allowed=lambda _: True)

    detail = disclosure.metadata("describe_tool", {"name": name})
    arguments = detail["matching_models"][0]["arguments"]
    assert arguments == {
        model_field: "ESM2-8M", dataset_field: "solubility", "task": "regression"
    }
    assert detail["scientific_inputs"] == ["sequence"]
    assert set(detail["inputSchema"]["required"]) == {"sequence", model_field, dataset_field}
    assert "default" not in detail["inputSchema"]["properties"][model_field]
    assert schema == original
    with pytest.raises(DisclosureError, match="explicitly"):
        disclosure.resolve("call_tool", {"name": name, "arguments": {"sequence": "ACDE"}})
    assert disclosure.resolve("call_tool", {
        "name": name, "arguments": {**arguments, "sequence": "ACDE"}
    })[2]["sequence"] == "ACDE"


def test_described_model_choices_are_bounded_and_refresh_live_availability(monkeypatch):
    from mcp_tools.tools.discovery import platform_models as registry

    name = "zero_shot_mutation_structure_prediction"
    live = [{
        "entry_id": "model-" + str(i), "name": "Variant " + str(i),
        "selection": {"model": "Variant " + str(i)},
    } for i in range(27)]
    monkeypatch.setattr(registry, "platform_models", lambda query, names: [
        (1, card["entry_id"], card) for card in live
    ])
    tool = SimpleNamespace(name=name, description="Score mutations.")
    disclosure = ToolDisclosure({name: tool}, schema=lambda _: {
        "type": "object", "properties": {"model_name": {"type": "string"}}
    }, allowed=lambda _: True)
    first = disclosure.metadata("describe_tool", {"name": name})
    assert first["model_count"] == 27 and len(first["matching_models"]) == 20
    assert first["more_models"]
    live[:] = live[-1:]
    second = disclosure.metadata("describe_tool", {"name": name})
    assert second["matching_models"] == [{
        "name": "Variant 26", "arguments": {"model_name": "Variant 26"}
    }]
    assert second["model_count"] == 1


@pytest.mark.parametrize("next_search", [
    {"category": "compute"},
    {"category": "compute", "query": "zero_shot_mutation_structure_prediction"},
])
def test_new_search_discards_previous_model_choice_hint(next_search):
    tool = tools()[0]
    disclosure = ToolDisclosure(
        {tool.name: tool}, schema=lambda t: t.parameters, allowed=lambda _: True
    )
    disclosure.metadata("search_tools", {"query": "ProSST-2048"})
    first = disclosure.metadata("describe_tool", {"name": tool.name})
    assert first["matching_models"] == [{
        "name": "ProSST-2048", "arguments": {"model_name": "ProSST-2048"}
    }]
    disclosure.metadata("search_tools", next_search)
    second = disclosure.metadata("describe_tool", {"name": tool.name})
    choices = {row["arguments"]["model_name"] for row in second["matching_models"]}
    assert {"ProSST-2048", "ProSST-128", "ESM-IF1"} <= choices


@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
def test_searching_another_tool_preserves_the_explicit_model_selection(harness):
    mapping = {web_tool_name(t.name, harness): t for t in tools()}
    disclosure = ToolDisclosure(mapping, schema=lambda t: t.parameters, allowed=lambda _: True)
    disclosure.metadata("search_tools", {"category": "compute", "query": "ProSST-2048"})
    disclosure.metadata("search_tools", {"category": "compute", "query": "LigandMPNN"})
    name = web_tool_name("zero_shot_mutation_structure_prediction", harness)
    detail = disclosure.metadata("describe_tool", {"name": name})
    assert detail["matching_models"] == [{"name": "ProSST-2048", "arguments": {"model_name": "ProSST-2048"}}]
    assert detail["model_count"] == 1
    disclosure.metadata("search_tools", {"category": "compute", "query": "ProSST-128"})
    revised = disclosure.metadata("describe_tool", {"name": name})
    assert revised["matching_models"] == [{"name": "ProSST-128", "arguments": {"model_name": "ProSST-128"}}]


def test_retained_model_selection_still_checks_live_availability():
    mapping = {t.name: t for t in tools()}
    disclosure = ToolDisclosure(mapping, schema=lambda t: t.parameters, allowed=lambda _: True)
    disclosure.metadata("search_tools", {"category": "compute", "query": "ProSST-2048"})
    disclosure.metadata("search_tools", {"category": "compute", "query": "LigandMPNN"})
    card = next(c for c in platform_catalog.official_cards() if c["name"] == "ProSST-2048")
    CapabilityStore().set("model:" + card["id"], False, 0, selector(card), "fixture")
    detail = disclosure.metadata("describe_tool", {"name": "zero_shot_mutation_structure_prediction"})
    assert not detail.get("matching_models")
