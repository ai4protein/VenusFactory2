"""Agent boundary tests use synthetic storage; no scientific model claims."""

import json

import pytest
from fastmcp import Client
from test_scientist_model_library import storage as model_storage  # noqa: F401

from mcp_tools.models.evolved import service
from mcp_tools.models.evolved.tools_agent import EVOLVED_MODEL_TOOLS, predict_with_evolved_model_tool
from mcp_tools.models.evolved.tools_mcp import mcp


@pytest.fixture
def published(request, monkeypatch):
    store, root, verified, calls = request.getfixturevalue("model_storage")
    verified["origin"].update(stage="activity", paper={"title": "Synthetic fixture", "doi": "fixture"})
    from agent.rsi.engine.model_plugins import get_model_plugin
    verified["packet"]["assessment"] = {"metrics": {"skill": 0.5},
                                          "plugin_sha256": get_model_plugin(verified["key"]["plugin_id"]).source_sha256}
    result = store.publish(root, "b" * 64)
    monkeypatch.setattr(service, "configured_library", lambda: store)
    monkeypatch.setattr(service, "assert_tool_allowed", lambda name: None)
    seen = []

    class Plugin:
        def predict(self, model, rows, scope):
            seen.extend(rows)
            return [{"id": row["id"], "sequence": row["sequence"], "prediction": 0.25} for row in rows]

    monkeypatch.setattr("agent.rsi.engine.scientist_model_library.get_model_plugin", lambda name: Plugin())
    return store, root, verified, result["entry_id"], seen


def test_catalog_is_paginated_and_does_not_claim_current_verification(published):
    store, root, verified, entry, seen = published
    result = service.dispatch("list", query="Synthetic", stage="activity", limit=1)
    assert result["success"] and result["matching_entries"] == 1
    assert not result["models"][0]["verified_now"]
    assert result["models"][0]["entry_id"] == entry
    assert service.dispatch("list", stage="stability")["models"] == []
    assert service.dispatch("list", offset=1)["models"] == []
    assert not service.dispatch("list", limit=51)["success"]
    assert not service.dispatch("list", limit=True)["success"]
    assert str(root) not in json.dumps(result)


def test_describe_returns_scope_without_sources_data_or_internal_paths(published):
    store, root, verified, entry, seen = published
    result = service.dispatch("describe", entry_id=entry)
    assert result["success"] and not result["verified_now"]
    assert result["validation_metrics_status"] == "recorded"
    assert result["scope"] == verified["key"]["scope"]
    assert result["validation_metrics"] == {"skill": 0.5}
    assert result["input_fields"] == ["id", "sequence"]
    assert result["input_contract"]["kind"] == "sequence"
    serialized = json.dumps(result)
    for private in ("source_code", "training_sequences", "goal_sha256", "origin_goal", str(root)):
        assert private not in serialized


@pytest.fixture(params=["generated_context.v1", "generated_context.v2", "generated_context.v3"])
def context_published(request, monkeypatch):
    from agent.rsi.engine.context_interfaces import module_for

    store, root, verified, _ = request.getfixturevalue("model_storage")
    interface = request.param
    columns = [
        {"name": f"measured_{i}", "unit": "arbitrary_units", "minimum": 0, "maximum": 100}
        for i in range(2 if interface.endswith("v1") else 19)
    ]
    scope = {
        "protein": "synthetic_fixture",
        "assay": "synthetic measured response",
        "target": "response",
        "unit": "arbitrary_units",
        "context": {"source": "synthetic_test_only"},
        "covariates": columns,
        "grouping": {"level": "biological_replicate", "evidence_sha256": "a" * 64},
    }
    if not interface.endswith("v1"):
        scope["interface_version"] = interface
        scope["grouping"]["level"] = "deleted_gene"
    verified["key"].update(plugin_id=interface, scope=scope)
    verified["packet"]["model"].update(plugin_id=interface, scope=scope)
    from agent.rsi.engine.model_plugins import get_model_plugin
    verified["packet"]["assessment"] = {"metrics": {"skill": 0.5},
                                          "plugin_sha256": get_model_plugin(verified["key"]["plugin_id"]).source_sha256}
    entry = store.publish(root, "b" * 64)["entry_id"]
    seen = []

    class Plugin:
        def predict(self, model, rows, model_scope):
            module_for(interface).validate_rows(rows, model_scope, labels=False)
            seen.extend(rows)
            return [{"id": row["id"], "prediction": sum(row["values"])} for row in rows]

    monkeypatch.setattr(service, "configured_library", lambda: store)
    monkeypatch.setattr(service, "assert_tool_allowed", lambda name: None)
    monkeypatch.setattr("agent.rsi.engine.scientist_model_library.get_model_plugin", lambda name: Plugin())
    inputs = [
        {"id": "query", "protein": scope["protein"], "group": "recorded_group", "values": list(range(len(columns)))}
    ]
    return entry, scope, inputs, seen


def test_context_contract_matches_actual_prediction_inputs(context_published):
    entry, scope, rows, seen = context_published
    card = service.dispatch("describe", entry_id=entry)
    assert card["success"] and card["input_fields"] == ["id", "protein", "group", "values"]
    contract = card["input_contract"]
    assert contract["kind"] == "measured_context" and not contract["target_required"]
    assert contract["protein"] == scope["protein"]
    assert contract["values"]["columns"] == scope["covariates"]
    assert contract["group"]["level"] == scope["grouping"]["level"]
    result = json.loads(
        predict_with_evolved_model_tool.invoke(
            {"entry_id": entry, "inputs": [{**rows[0], "target": 12345}], "scope": card["scope"]},
            config={"configurable": {"thread_id": "context-chat"}},
        )
    )
    assert result["success"] and result["receipt"]["session_id"] == "context-chat"
    assert seen == rows
    assert result["predictions"][0]["prediction"] == sum(rows[0]["values"])


@pytest.mark.parametrize("invalid", ["sequence_only", "wrong_protein", "missing_value", "nonfinite", "out_of_range"])
@pytest.mark.parametrize("use_reference", [False, True])
def test_context_contract_does_not_weaken_input_validation(context_published, invalid, use_reference):
    entry, scope, rows, seen = context_published
    row = {**rows[0], "values": list(rows[0]["values"])}
    if invalid == "sequence_only":
        row = {"id": "query", "sequence": "AA"}
    elif invalid == "wrong_protein":
        row["protein"] = "another_protein"
    elif invalid == "missing_value":
        row["values"].pop()
    elif invalid == "nonfinite":
        row["values"][0] = float("nan")
    else:
        row["values"][0] = 101
    binding = ({"scope_ref": service.dispatch("describe", entry_id=entry)["scope_ref"]}
               if use_reference else {"scope": scope})
    result = service.dispatch("predict", entry_id=entry, inputs=[row], **binding)
    assert not result["success"] and not seen


@pytest.mark.asyncio
@pytest.mark.parametrize("use_reference", [False, True])
async def test_mcp_context_description_and_inference(context_published, use_reference):
    entry, scope, rows, seen = context_published
    async with Client(mcp) as client:
        card = (await client.call_tool("describe_evolved_model", {"entry_id": entry})).data
        assert card["input_fields"] == list(rows[0])
        result = (
            await client.call_tool(
                "predict_with_evolved_model", {"entry_id": entry, "inputs": rows,
                    **({"scope_ref": card["scope_ref"]} if use_reference else {"scope": card["scope"]})}
            )
        ).data
        assert result["success"] and result["session_binding"] == "mcp_transport_session"
        assert seen == rows


def test_predict_requires_exact_scope_and_strips_labels(published):
    store, root, verified, entry, seen = published
    rows = [{"id": "test", "sequence": "AA", "target": 12345}]
    failed = service.dispatch("predict", entry_id=entry, inputs=rows, scope={})
    assert not failed["success"] and not seen
    result = service.dispatch("predict", entry_id=entry, inputs=rows, scope=verified["key"]["scope"])
    assert result["success"] and result["session_binding"] == "tool_request"
    assert result["receipt"]["session_id"].startswith("agent_tool_request_")
    assert seen == [{"id": "test", "sequence": "AA"}]
    assert not service.dispatch("predict", entry_id=entry, inputs=rows * 257, scope=verified["key"]["scope"])["success"]


def test_retirement_or_invalid_evidence_blocks_actual_use(published, monkeypatch):
    store, root, verified, entry, seen = published

    def unavailable(*args, **kwargs):
        raise ValueError("Internal evidence path /private/goal has changed")

    monkeypatch.setattr(store, "resolve", unavailable)
    result = service.dispatch(
        "predict", entry_id=entry, inputs=[{"id": "x", "sequence": "AA"}], scope=verified["key"]["scope"]
    )
    assert not result["success"] and not seen
    assert "/private" not in json.dumps(result)
    recorded = service.dispatch("describe", entry_id=entry)
    assert recorded["success"] and not recorded["verified_now"]
    assert recorded["validation_metrics_status"] == "recorded"


@pytest.mark.parametrize("action", ["list", "describe", "predict"])
def test_online_mode_blocks_every_entry_before_library_access(monkeypatch, action):
    def deny(name):
        raise RuntimeError("Unavailable in online mode")

    def no_access():
        pytest.fail("Library must not be accessed in disabled mode")

    monkeypatch.setattr(service, "assert_tool_allowed", deny)
    monkeypatch.setattr(service, "configured_library", no_access)
    assert service.dispatch(action)["error_code"] == "runtime_mode_disabled"


def test_model_cannot_select_a_library_path(published):
    assert not service.dispatch("list", library_root="/private/another_owner")["success"]
    for tool in EVOLVED_MODEL_TOOLS:
        fields = tool.tool_call_schema.model_json_schema()["properties"]
        assert not {"config", "ctx", "session_id", "library_root", "path"} & fields.keys()


def test_langchain_tool_receives_session_from_host_configuration(published):
    store, root, verified, entry, seen = published
    result = json.loads(
        predict_with_evolved_model_tool.invoke(
            {"entry_id": entry, "inputs": [{"id": "test", "sequence": "AA"}], "scope": verified["key"]["scope"]},
            config={"configurable": {"thread_id": "host-chat-session"}},
        )
    )
    assert result["success"]
    assert result["receipt"]["session_id"] == "host-chat-session"
    assert result["session_binding"] == "chat_session"


@pytest.mark.asyncio
async def test_mcp_discovers_and_calls_the_same_library_in_one_transport_session(published):
    store, root, verified, entry, seen = published
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools} == set(service.TOOL_NAMES.values())
        for tool in tools:
            assert not {"ctx", "session_id", "library_root"} & tool.inputSchema["properties"].keys()
        listing = (await client.call_tool("list_evolved_models", {})).data
        assert listing["models"][0]["entry_id"] == entry
        description = (await client.call_tool("describe_evolved_model", {"entry_id": entry})).data
        arguments = {"entry_id": entry, "inputs": [{"id": "test", "sequence": "AA"}], "scope": description["scope"]}
        first = (await client.call_tool("predict_with_evolved_model", arguments)).data
        second = (await client.call_tool("predict_with_evolved_model", arguments)).data
        assert first["success"] and second["success"]
        assert first["receipt"]["session_id"] == second["receipt"]["session_id"]
        assert first["receipt_id"] != second["receipt_id"]
        assert first["session_binding"] == "mcp_transport_session"


def test_context_discovery_is_compact_and_details_remain_on_demand(context_published):
    entry, scope, _, _ = context_published
    query = scope["covariates"][-1]["name"]
    result = service.dispatch("list", query=query)
    assert result["success"] and result["matching_entries"] == 1
    card = result["models"][0]
    assert card["entry_id"] == entry
    assert card["name"] and card["description"]
    assert card["describe"] == {"tool": "describe_evolved_model", "arguments": {"entry_id": entry}}
    assert not card["verified_now"]
    assert not {"scope", "covariates", "paper", "input_contract"} & card.keys()
    assert scope["grouping"]["evidence_sha256"] not in json.dumps(card)
    assert len(json.dumps(card)) < 1200
    full = service.dispatch("describe", entry_id=entry)
    assert full["scope"] == scope
    assert full["input_contract"]["values"]["columns"] == scope["covariates"]


def test_warm_metadata_does_not_bypass_current_policy_or_evidence(published, monkeypatch):
    store, root, verified, entry, _ = published
    assert service.dispatch("list")["matching_entries"] == 1
    from mcp_tools.tools.capability_policy import CapabilityStore
    monkeypatch.setattr(CapabilityStore, "enabled", lambda *args: False)
    assert service.dispatch("list")["matching_entries"] == 0
    from agent.rsi.engine.storage import atomic_json
    atomic_json(root / "goal.json", {"changed_after_discovery": True})
    assert not service.dispatch("describe", entry_id=entry)["success"]


def test_inspection_keeps_full_validation_out_of_metadata_reads(published, monkeypatch):
    store, root, verified, entry, seen = published
    def no_execution(*args, **kwargs):
        pytest.fail("Reading model details must not replay native research or numerical experiments")
    monkeypatch.setattr(store, "resolve", no_execution)
    monkeypatch.setattr("agent.rsi.engine.scientist_model_library.validate_model_goal", no_execution)
    description = service.dispatch("describe", entry_id=entry)
    assert description["success"] and not description["verified_now"] and not seen
    assert description["scope"] == verified["key"]["scope"]
    assert description["validation_metrics_status"] == "recorded"


def test_inspection_checks_live_runtime_after_discovery(published, monkeypatch):
    store, root, verified, entry, _ = published
    assert service.dispatch("list")["matching_entries"] == 1
    monkeypatch.setattr(service, "runtime_implementation", lambda _: "0" * 64)
    assert not service.dispatch("describe", entry_id=entry)["success"]


@pytest.mark.parametrize("attack", ["packet", "entry", "index", "origin", "goal", "protocol", "symlink"])
def test_inspection_rejects_changed_publication_bindings(published, attack):
    from agent.rsi.engine.storage import atomic_json
    store, root, verified, entry, _ = published
    if attack in {"goal", "protocol"}:
        path = root / (attack + ".json")
    else:
        pattern = {"packet": "packets/*.json", "entry": "entries/*.json",
                   "index": "index/*.json", "origin": "origins/*/*.json",
                   "symlink": "packets/*.json"}[attack]
        path = next(store.root.glob(pattern))
    path.chmod(0o644)
    if attack == "symlink":
        saved = path.with_suffix(".fixture")
        path.rename(saved)
        path.symlink_to(saved)
    else:
        atomic_json(path, {"changed": True})
    assert not service.dispatch("describe", entry_id=entry)["success"]


def test_inspection_rebinds_after_reading_publication(published, monkeypatch):
    from agent.rsi.engine import scientist_model_library as library
    from agent.rsi.engine.storage import atomic_json
    store, root, verified, entry, _ = published
    model_key = library.model_key
    def change_index(model, policy):
        key = model_key(model, policy)
        index = next((store.root / "index").glob("*.json"))
        index.chmod(0o644)
        atomic_json(index, {"entry_id": "0" * 64})
        return key
    monkeypatch.setattr(library, "model_key", change_index)
    assert not service.dispatch("describe", entry_id=entry)["success"]


def test_registered_model_search_uses_shared_topic_matching(published):
    store, root, verified, entry, _ = published
    for query in (entry, "model " + entry, "Scientist model " + entry[:16]):
        result = service.dispatch("list", query=query)
        assert result["success"] and result["matching_entries"] == 1
        assert result["models"][0]["entry_id"] == entry
    assert service.dispatch("list", query="unregistered-protein")["matching_entries"] == 0

def test_reference_prediction_keeps_the_complete_context_and_host_session(context_published):
    from agent.harness.contracts import digest
    from mcp_tools.models.evolved.scope import scope_reference

    entry, scope, rows, seen = context_published
    card = service.dispatch("describe", entry_id=entry)
    assert card["scope_ref"] == scope_reference(entry, scope)
    result = json.loads(predict_with_evolved_model_tool.invoke(
        {"entry_id": entry, "inputs": rows, "scope_ref": card["scope_ref"]},
        config={"configurable": {"thread_id": "reference-chat"}},
    ))
    assert result["success"] and seen == rows
    assert result["receipt"]["scope_sha256"] == digest(scope)
    assert result["receipt"]["session_id"] == "reference-chat"


@pytest.mark.parametrize("binding", ["absent", "wrong", "malformed", "other_entry", "contradictory_scope"])
def test_invalid_scope_binding_never_starts_numerical_replay(published, monkeypatch, binding):
    from mcp_tools.models.evolved.scope import scope_reference

    store, root, verified, entry, seen = published
    scope = verified["key"]["scope"]
    reference = service.dispatch("describe", entry_id=entry)["scope_ref"]
    def must_not_execute(*args, **kwargs):
        pytest.fail("Invalid applicability binding reached numerical replay")
    monkeypatch.setattr(store, "resolve", must_not_execute)
    arguments = {
        "absent": {},
        "wrong": {"scope_ref": "0" * 64},
        "malformed": {"scope_ref": "private-input-marker"},
        "other_entry": {"scope_ref": scope_reference("0" * 64, scope)},
        "contradictory_scope": {"scope_ref": reference, "scope": {}},
    }[binding]
    result = service.dispatch("predict", entry_id=entry, inputs=[{"id": "x", "sequence": "AA"}], **arguments)
    assert not result["success"] and result["error_code"] == "invalid_model_scope" and not seen
    assert "private-input-marker" not in json.dumps(result)


def test_incomplete_inline_scope_names_missing_field_before_replay(context_published, monkeypatch):
    entry, scope, rows, seen = context_published
    field = "interface_version" if "interface_version" in scope else "target"
    incomplete = {key: value for key, value in scope.items() if key != field}
    store = service.configured_library()
    monkeypatch.setattr(store, "resolve", lambda *a, **k: pytest.fail("Scope rejection must precede replay"))
    result = service.dispatch("predict", entry_id=entry, inputs=rows, scope=incomplete)
    assert result["error_code"] == "invalid_model_scope" and field in result["error"]
    assert "scope_ref" in result["error"] and not seen


def test_reference_does_not_bypass_retirement_or_revalidation(published, monkeypatch):
    store, root, verified, entry, seen = published
    reference = service.dispatch("describe", entry_id=entry)["scope_ref"]
    def unavailable(*args, **kwargs):
        raise ValueError("private-runtime-marker")
    monkeypatch.setattr(store, "resolve", unavailable)
    result = service.dispatch("predict", entry_id=entry, inputs=[{"id": "x", "sequence": "AA"}],
                              scope_ref=reference)
    assert not result["success"] and not seen and "private-runtime-marker" not in json.dumps(result)


def test_reference_does_not_bypass_admin_policy(published, monkeypatch):
    from mcp_tools.tools import runtime_tool_policy
    store, root, verified, entry, seen = published
    reference = service.dispatch("describe", entry_id=entry)["scope_ref"]
    monkeypatch.setattr(runtime_tool_policy, "assert_tool_call_allowed",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("disabled fixture")))
    monkeypatch.setattr(store, "inspect_publication", lambda *a: pytest.fail("Disabled entry inspected"))
    assert not service.dispatch("predict", entry_id=entry, inputs=[{"id": "x", "sequence": "AA"}],
                                scope_ref=reference)["success"]


def test_reference_changes_with_entry_scope_and_interface_version():
    from mcp_tools.models.evolved.scope import scope_reference
    scope = {"protein": "fixture", "interface_version": "generated_context.v2"}
    reference = scope_reference("a" * 64, scope)
    assert reference != scope_reference("b" * 64, scope)
    assert reference != scope_reference("a" * 64, {**scope, "interface_version": "generated_context.v3"})
    assert reference != scope_reference("a" * 64, {"protein": "fixture"})
