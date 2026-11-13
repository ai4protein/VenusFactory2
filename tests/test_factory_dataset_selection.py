"""Real factory interfaces must preserve the selected dataset without loading weights."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness.session_boundary import mcp_input_schema, prepare_tool_arguments
from agent.harness.tool_disclosure import DisclosureError, ToolDisclosure
from agent.harness.web_native_worker import web_tool_name
from mcp_tools.models import platform_catalog
from mcp_tools.models.factory.prediction import fintuned_operations as operations
from mcp_tools.models.factory.prediction.output_scale import output_scale
from mcp_tools.models.factory.runners import finetuned
from mcp_tools.tools.capability_policy import CapabilityStore, selector
from mcp_tools.tools.predict.tools_agent import predict_protein_function_tool
from mcp_tools.tools.runtime_tool_policy import is_tool_allowed


@pytest.fixture(autouse=True)
def private_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("VENUS_CKPT_AUTO_DOWNLOAD", "0")
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    from agent.harness import research_workspace
    from mcp_tools.models import identity
    from mcp_tools.models.pretrained.design_docking import service
    monkeypatch.setattr(research_workspace, "LIBRARY", tmp_path / "published")
    # Only hardware readiness is synthetic; real cards and administrator policy remain.
    monkeypatch.setattr(identity, "runtime_availability", lambda *a, **k: (True, None))
    monkeypatch.setattr(service, "cards", lambda: [])


def catalog(harness="kimi-code"):
    tool = predict_protein_function_tool
    name = web_tool_name(tool.name, harness)
    return ToolDisclosure({name: tool}, schema=mcp_input_schema,
                          allowed=lambda value: is_tool_allowed(value.name)), name


def selected(disclosure, name, dataset):
    page = disclosure.metadata("search_tools", {"category": "factory_models", "query": dataset})
    assert name in {row["name"] for row in page["tools"]}
    detail = disclosure.metadata("describe_tool", {"name": name})
    choice = next(row for row in detail["matching_models"]
                  if row["arguments"].get("dataset") == dataset
                  and row["arguments"].get("model_name") == "ESM2-650M")
    assert {"model_name", "dataset"} <= set(detail["inputSchema"]["required"])
    assert "dataset" not in detail["scientific_inputs"]
    return choice["arguments"]


@pytest.fixture
def runtime_double(tmp_path, monkeypatch):
    base = tmp_path / "ckpt"
    configs = {}
    for dataset in ("DeepSol", "DeepSoluE", "ProtSolM", "DeepLocBinary"):
        adapter = base / dataset / "esm2_t33_650M_UR50D"
        adapter.mkdir(parents=True)
        config = {"dataset": "tyang816/" + ("DeepLocBinary_ESMFold" if dataset == "DeepLocBinary" else dataset),
                  "problem_type": "single_label_classification", "num_labels": 2}
        (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(config))
        configs[dataset] = config
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">engineering-fixture\nACDEFGHIKLMNPQRSTV\n")
    state = SimpleNamespace(base=base, fasta=fasta, calls=[], loads=[], runtime_args=[], configs=configs)

    def load(args):
        state.runtime_args.append(vars(args).copy())
        state.loads.append(args.adapter_path)
        args._inference_checkpoint_config = json.loads((Path(args.adapter_path) / "lr5e-4_bt12k_ga8.json").read_text())
        return None, None, None, "cpu"

    module = SimpleNamespace(parse_fasta=lambda path: [("engineering-fixture", "ACDEFGHIKLMNPQRSTV")],
                             load_model_and_tokenizer=load,
                             predict=lambda *a: {"predicted_class": 1, "probabilities": [0.1, 0.9]})
    monkeypatch.setattr(operations, "_get_module", lambda key: (module, lambda *a: {"input_ids": [], "attention_mask": []}))
    monkeypatch.setattr(operations, "_ensure_adapter_dir", lambda path: path)
    monkeypatch.setattr(finetuned, "profile", lambda name: ({}, None))
    monkeypatch.setattr(finetuned, "command", lambda name, argv: argv)

    def execute(argv):
        # Exercise the actual in_runtime JSON transport; only process/neural compute is doubled.
        request = json.loads(Path(argv[-2]).read_text())
        state.calls.append(request)
        implementation = getattr(operations, request["operation"]).runtime_implementation
        result = implementation(**request["arguments"])
        Path(argv[-1]).write_text(json.dumps(result))

    monkeypatch.setattr(finetuned, "execute", execute)
    return state


def invoke(disclosure, name, arguments, state, tmp_path):
    arguments = {**arguments, "fasta_file": str(state.fasta), "ckpt_base": str(state.base),
                 "output_file": str(tmp_path / "prediction.csv")}
    _, tool, values = disclosure.resolve("call_tool", {"name": name, "arguments": arguments})
    values = prepare_tool_arguments(tool.name, tool, values, tmp_path)
    return json.loads(tool.invoke(values))


@pytest.mark.parametrize("mode", ["local", "online"])
@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
@pytest.mark.parametrize("dataset", ["DeepSol", "DeepSoluE", "ProtSolM"])
def test_real_discovery_schema_runner_and_metadata_keep_dataset(
    tmp_path, monkeypatch, runtime_double, mode, harness, dataset
):
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: mode == "online")
    disclosure, name = catalog(harness)
    arguments = selected(disclosure, name, dataset)
    assert arguments == {"model_name": "ESM2-650M", "dataset": dataset, "task": "Solubility"}
    with pytest.raises(DisclosureError, match="explicitly"):
        disclosure.resolve("call_tool", {"name": name, "arguments": {
            "model_name": "ESM2-650M", "task": "Solubility", "fasta_file": "input.fasta"}})
    result = invoke(disclosure, name, arguments, runtime_double, tmp_path)
    assert result["status"] == "success"
    assert runtime_double.calls[0]["arguments"]["dataset"] == dataset
    expected = runtime_double.base / dataset / "esm2_t33_650M_UR50D"
    assert runtime_double.loads == [str(expected)]
    meta = result["biological_metadata"]
    assert meta["dataset_name"] == dataset and meta["adapter_path"] == str(expected)
    assert meta["output_scale"] == output_scale(runtime_double.configs[dataset])
    assert meta["classification"]["mapping_status"] == "unavailable"
    assert meta["classification"]["reason"] == "dataset_mapping_not_verified"
    assert "decoded" not in meta["classification"]
    path = tmp_path / "prediction.csv"
    assert result["file_info"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as handle:
        assert list(csv.DictReader(handle))[0]["predicted_class"] == "1"


@pytest.mark.parametrize("disabled,requested,allowed", [
    ("ProtSolM", "ProtSolM", False), ("DeepSol", "ProtSolM", True),
])
def test_live_model_switch_uses_dataset_without_confusing_shared_backbone(
    tmp_path, monkeypatch, runtime_double, disabled, requested, allowed
):
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: True)
    disclosure, name = catalog()
    arguments = selected(disclosure, name, requested)
    card = next(card for card in platform_catalog.official_cards()
                if card.get("dataset") == disabled and card.get("model") == "ESM2-650M")
    store = CapabilityStore()
    revision = store.set("model:" + card["id"], False, 0, selector(card), "fixture operator")
    result = invoke(disclosure, name, arguments, runtime_double, tmp_path)
    assert (result["status"] == "success") is allowed
    assert bool(runtime_double.calls) is allowed
    assert bool(runtime_double.loads) is allowed
    # A fresh search must see the switch, without another model's stale selection.
    again, again_name = catalog()
    again.metadata("search_tools", {"query": requested})
    details = again.metadata("describe_tool", {"name": again_name})
    choices = [row["arguments"] for row in details.get("matching_models", [])]
    assert ({"model_name": "ESM2-650M", "dataset": requested, "task": "Solubility"} in choices) is allowed
    store.set("model:" + card["id"], True, revision, selector(card), "fixture operator")
    restored, restored_name = catalog()
    assert selected(restored, restored_name, disabled)["dataset"] == disabled


@pytest.mark.parametrize("dataset", ["", "Unknown", "DeepLocBinary", "../DeepSol", "/workspace/DeepSol"])
def test_invalid_or_cross_task_dataset_stops_before_weight_lookup(tmp_path, monkeypatch, dataset):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">fixture\nACDE\n")
    monkeypatch.setattr(operations, "_ensure_adapter_dir", lambda _: pytest.fail("Must validate before weight lookup"))
    result = json.loads(operations.predict_protein_function.runtime_implementation(
        str(fasta), "Solubility", "ESM2-650M", dataset=dataset))
    assert result["status"] == "error" and result["error"]["type"] == "ValidationError"


def test_missing_selected_weights_never_fall_back_to_default(tmp_path, monkeypatch, runtime_double):
    missing = "ProtSolM"
    adapter = runtime_double.base / missing / "esm2_t33_650M_UR50D"
    (adapter / "lr5e-4_bt12k_ga8.json").unlink()
    adapter.rmdir()
    result = json.loads(operations.predict_protein_function.runtime_implementation(
        str(runtime_double.fasta), "Solubility", "ESM2-650M", ckpt_base=str(runtime_double.base), dataset=missing))
    assert result["status"] == "error" and missing in result["error"]["message"]
    assert not runtime_double.loads


def test_explicit_dataset_cannot_claim_a_private_override(tmp_path, runtime_double):
    result = json.loads(operations.predict_protein_function.runtime_implementation(
        str(runtime_double.fasta), "Solubility", "ESM2-650M",
        adapter_path=str(runtime_double.base / "DeepSol" / "esm2_t33_650M_UR50D"), dataset="ProtSolM"))
    assert result["status"] == "error" and "not both" in result["error"]["message"]
    assert not runtime_double.loads


def test_default_and_existing_private_adapter_calls_remain_compatible(tmp_path, runtime_double):
    for override in (None, str(runtime_double.base / "DeepSol" / "esm2_t33_650M_UR50D")):
        result = json.loads(operations.predict_protein_function.runtime_implementation(
            str(runtime_double.fasta), "Solubility", "ESM2-650M", override,
            str(runtime_double.base), str(tmp_path / "legacy.csv")))
        expected_dataset = "tyang816/DeepSol" if override else "DeepSol"
        assert result["status"] == "success" and result["biological_metadata"]["dataset_name"] == expected_dataset


def test_known_classification_stays_bound_to_actual_selected_config(tmp_path, runtime_double):
    result = json.loads(operations.predict_protein_function.runtime_implementation(
        str(runtime_double.fasta), "Membrane Protein", "ESM2-650M", ckpt_base=str(runtime_double.base),
        output_file=str(tmp_path / "classification.csv"), dataset="DeepLocBinary"))
    assert result["status"] == "success"
    meta = result["biological_metadata"]
    assert meta["classification"]["decoded"][0]["predicted_labels"] == ["Soluble"]
    assert meta["classification"]["checkpoint_config_sha256"] == meta["output_scale"]["checkpoint_config_sha256"]


@pytest.mark.asyncio
async def test_mcp_schema_and_call_preserve_dataset(tmp_path, monkeypatch):
    from fastmcp import Client

    from mcp_tools.tools.predict import tools_mcp
    captured = []
    monkeypatch.setattr(tools_mcp, "predict_protein_function", lambda **kwargs: captured.append(kwargs) or json.dumps({"status": "fixture"}))
    async with Client(tools_mcp.mcp) as client:
        definitions = await client.list_tools()
        schema = next(tool.inputSchema for tool in definitions if tool.name == "predict_protein_function")
        assert "dataset" in schema["properties"]
        await client.call_tool("predict_protein_function", {
            "fasta_file": "fixture.fasta", "task": "Solubility", "model_name": "ESM2-650M", "dataset": "ProtSolM"})
    assert captured[0]["dataset"] == "ProtSolM"


def test_http_body_and_adapter_preserve_dataset(monkeypatch):
    from mcp_tools.tools.predict import tools_api
    captured = []
    monkeypatch.setattr(tools_api, "predict_protein_function", lambda **kwargs: captured.append(kwargs) or json.dumps({"status": "fixture"}))
    body = tools_api.PredictProteinFunctionBody(
        fasta_file="fixture.fasta", task="Solubility", model_name="ESM2-650M", dataset="ProtSolM")
    # The outer decorators are policy/quota; the transport itself is inspected here.
    function = tools_api.api_predict_protein_function
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    assert function(body)["status"] == "fixture"
    assert captured[0]["dataset"] == "ProtSolM"


def generic_private(disclosure, name):
    page = disclosure.metadata("search_tools", {"category": "factory_models", "query": "predict_protein_function"})
    assert name in {row["name"] for row in page["tools"]}
    detail = disclosure.metadata("describe_tool", {"name": name})
    assert "model_name" in detail["inputSchema"]["required"]
    assert "dataset" not in detail["inputSchema"]["required"]
    return detail


@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
def test_generic_private_adapter_uses_real_disclosure_boundary_and_runner(tmp_path, runtime_double, harness):
    disclosure, name = catalog(harness)
    generic_private(disclosure, name)
    adapter = runtime_double.base / "DeepLocBinary" / "esm2_t33_650M_UR50D"
    result = invoke(disclosure, name, {
        "task": "Membrane Protein", "model_name": "ESM2-650M", "adapter_path": str(adapter)
    }, runtime_double, tmp_path)
    assert result["status"] == "success"
    assert runtime_double.calls[0]["arguments"]["dataset"] is None
    assert runtime_double.loads == [str(adapter)]
    meta = result["biological_metadata"]
    assert meta["dataset_name"] == "tyang816/DeepLocBinary_ESMFold"
    assert meta["classification"]["decoded"][0]["predicted_labels"] == ["Soluble"]
    assert meta["classification"]["checkpoint_config_sha256"] == meta["output_scale"]["checkpoint_config_sha256"]
    output = tmp_path / "prediction.csv"
    assert result["file_info"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_named_catalog_choice_cannot_be_downgraded_by_private_call_arguments(tmp_path, runtime_double):
    disclosure, name = catalog()
    choice = selected(disclosure, name, "ProtSolM")
    private = {"task": "Solubility", "model_name": "ESM2-650M", "adapter_path": str(
        runtime_double.base / "DeepSol" / "esm2_t33_650M_UR50D")}
    for args in (private, {**choice, **private}):
        with pytest.raises(DisclosureError, match="cannot be replaced"):
            invoke(disclosure, name, args, runtime_double, tmp_path)
    # Re-describing the selected tool retains selection. An explicit generic search changes intent.
    disclosure.metadata("describe_tool", {"name": name})
    with pytest.raises(DisclosureError, match="cannot be replaced"):
        invoke(disclosure, name, private, runtime_double, tmp_path)
    assert not runtime_double.calls
    generic_private(disclosure, name)
    assert invoke(disclosure, name, private, runtime_double, tmp_path)["status"] == "success"


def test_generic_private_route_keeps_explicit_backbone_without_catalog_options(tmp_path, monkeypatch, runtime_double):
    from mcp_tools.tools.discovery import platform_models
    monkeypatch.setattr(platform_models, "platform_models", lambda *a, **k: [])
    disclosure, name = catalog()
    generic_private(disclosure, name)
    arguments = {"task": "Solubility", "adapter_path": str(runtime_double.base / "DeepSol" / "esm2_t33_650M_UR50D")}
    with pytest.raises(DisclosureError, match="explicitly"):
        invoke(disclosure, name, arguments, runtime_double, tmp_path)
    assert not runtime_double.calls
    result = invoke(disclosure, name, {**arguments, "model_name": "ESM2-650M"}, runtime_double, tmp_path)
    assert result["status"] == "success"


@pytest.mark.parametrize("override_kind", ["missing", "regular_file"])
def test_invalid_private_adapter_never_falls_back_to_disabled_default(
    tmp_path, monkeypatch, runtime_double, override_kind
):
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: True)
    card = next(card for card in platform_catalog.official_cards()
                if card.get("dataset") == "DeepSol" and card.get("model") == "ESM2-650M")
    CapabilityStore().set("model:" + card["id"], False, 0, selector(card), "fixture operator")
    adapter = tmp_path / "private-adapter"
    if override_kind == "regular_file":
        adapter.write_text("not a directory")
    disclosure, name = catalog()
    generic_private(disclosure, name)
    result = invoke(disclosure, name, {
        "task": "Solubility", "model_name": "ESM2-650M", "adapter_path": str(adapter)
    }, runtime_double, tmp_path)
    assert result["status"] == "error" and result["error"]["type"] == "ValidationError"
    assert "explicit adapter_path" in result["error"]["message"]
    assert not runtime_double.loads


def test_private_route_keeps_disabled_model_and_tool_checks(tmp_path, monkeypatch, runtime_double):
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: True)
    disclosure, name = catalog()
    generic_private(disclosure, name)
    card = next(card for card in platform_catalog.official_cards()
                if card.get("dataset") == "DeepSol" and card.get("model") == "ESM2-650M")
    store = CapabilityStore()
    store.set("model:" + card["id"], False, 0, selector(card), "fixture operator")
    private = {"task": "Solubility", "model_name": "ESM2-650M", "adapter_path": str(
        runtime_double.base / "DeepSol" / "esm2_t33_650M_UR50D")}
    result = invoke(disclosure, name, private, runtime_double, tmp_path)
    assert result["status"] == "error" and not runtime_double.calls
    store.set("tool:predict_protein_function", False, 0, {}, "fixture operator")
    with pytest.raises(DisclosureError, match="unavailable"):
        invoke(disclosure, name, private, runtime_double, tmp_path)
    assert not runtime_double.calls and not runtime_double.loads


def test_private_adapter_outside_session_stops_before_runner(tmp_path, runtime_double):
    disclosure, name = catalog()
    generic_private(disclosure, name)
    private = {"task": "Solubility", "model_name": "ESM2-650M", "adapter_path": str(tmp_path.parent / "other-session" / "adapter")}
    with pytest.raises(ValueError, match="conversation"):
        invoke(disclosure, name, private, runtime_double, tmp_path)
    assert not runtime_double.calls and not runtime_double.loads


def test_private_adapter_missing_inference_snapshot_does_not_claim_task_default(tmp_path, monkeypatch, runtime_double):
    original = operations._prediction_interpretation
    def without_snapshot(args, rows):
        del args._inference_checkpoint_config
        return original(args, rows)
    monkeypatch.setattr(operations, "_prediction_interpretation", without_snapshot)
    disclosure, name = catalog()
    generic_private(disclosure, name)
    result = invoke(disclosure, name, {
        "task": "Solubility", "model_name": "ESM2-650M", "adapter_path": str(
            runtime_double.base / "DeepSol" / "esm2_t33_650M_UR50D")
    }, runtime_double, tmp_path)
    assert result["status"] == "success"
    meta = result["biological_metadata"]
    assert meta["dataset_name"] is None
    assert meta["classification"]["mapping_status"] == "unavailable"


@pytest.mark.parametrize("operation", ["protein", "residue"])
def test_checkpoint_config_cannot_redirect_host_adapter_or_output(tmp_path, runtime_double, operation):
    selected_adapter = runtime_double.base / "ProtSolM" / "esm2_t33_650M_UR50D"
    other_adapter = runtime_double.base / "DeepLocBinary" / "esm2_t33_650M_UR50D"
    redirected_output = tmp_path / "not-selected.csv"
    config_path = selected_adapter / "lr5e-4_bt12k_ga8.json"
    injected = {**runtime_double.configs["ProtSolM"],
                "adapter_path": str(other_adapter), "output_csv": str(redirected_output),
                "output_file": str(redirected_output), "fasta_file": "not-selected.fasta",
                "task": "Membrane Protein", "model_name": "ProtBert",
                "_inference_checkpoint_config": runtime_double.configs["DeepLocBinary"],
                "_checkpoint_config": {"dataset": "unexpected"}}
    config_path.write_text(json.dumps(injected))
    if operation == "protein":
        disclosure, name = catalog()
        arguments = selected(disclosure, name, "ProtSolM")
        result = invoke(disclosure, name, arguments, runtime_double, tmp_path)
        output = tmp_path / "prediction.csv"
    else:
        output = tmp_path / "residue.csv"
        result = json.loads(operations.predict_residue_function.runtime_implementation(
            str(runtime_double.fasta), "Activity Site", "ESM2-650M", str(selected_adapter),
            str(runtime_double.base), str(output)))
    assert result["status"] == "success"
    assert runtime_double.loads == [str(selected_adapter)]
    actual = runtime_double.runtime_args[0]
    assert actual["adapter_path"] == str(selected_adapter)
    assert actual["output_csv"] == str(output)
    assert not {"output_file", "fasta_file", "task", "model_name", "_checkpoint_config"} & actual.keys()
    # Core projection is not proof of the loader's later, actual configuration read.
    assert "_inference_checkpoint_config" not in actual
    assert result["biological_metadata"]["adapter_path"] == str(selected_adapter)
    if operation == "protein":
        assert result["biological_metadata"]["dataset_name"] == "ProtSolM"
    assert result["biological_metadata"]["classification"]["mapping_status"] == "unavailable"
    assert output.is_file() and not redirected_output.exists()
    assert result["file_info"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
