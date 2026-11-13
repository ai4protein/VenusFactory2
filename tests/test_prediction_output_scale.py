"""Output interpretation follows checkpoint config, never the task's display name."""

import json
from types import SimpleNamespace

import pytest

from mcp_tools.models.factory.prediction.output_scale import (
    checkpoint_output_scale,
    output_scale,
    result_output_scale,
)


@pytest.mark.parametrize(
    "normalization,space,unit",
    [
        ("standard", "standardized_target", "dimensionless"),
        ("min_max", "min_max_normalized_target", "dimensionless"),
        (None, "original_target", "unknown"),
        ("unrecognized", "unknown", "unknown"),
    ],
)
def test_regression_scale_does_not_invent_units_or_inverse_transform(tmp_path, normalization, space, unit):
    config = {"problem_type": "regression", "normalize": normalization, "dataset": "DeepET_Topt"}
    (tmp_path / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(config))
    scale = checkpoint_output_scale(tmp_path)
    assert scale["value_space"] == space and scale["unit"] == unit
    assert scale["inverse_transform_available"] is False and scale["inverse_transform_applied"] is False
    assert len(scale["checkpoint_config_sha256"]) == 64
    assert "Celsius" not in json.dumps(scale) and "120" not in json.dumps(scale)


def test_missing_normalization_is_unknown_not_original_target(tmp_path):
    assert checkpoint_output_scale(tmp_path)["value_space"] == "unknown"
    (tmp_path / "lr5e-4_bt12k_ga8.json").write_text('{"problem_type":"regression"}')
    assert checkpoint_output_scale(tmp_path)["normalization"] == "unknown"
    assert (
        output_scale({"problem_type": "regression", "normalize": None, "target_unit": "Celsius"})["unit"] == "Celsius"
    )
    assert (
        output_scale({"problem_type": "regression", "normalize": "standard", "target_unit": "Celsius"})["unit"]
        == "dimensionless"
    )


@pytest.mark.parametrize(
    "problem", ["single_label_classification", "multi_label_classification", "residue_single_label_classification"]
)
def test_classification_is_never_labelled_a_standardized_physical_quantity(problem):
    scale = output_scale({"problem_type": problem, "normalize": "standard"})
    assert scale["value_space"] == "class_index_and_probability"
    assert scale["normalization"] == "not_applicable" and scale["unit"] == "not_applicable"


def test_worker_result_must_match_the_checkpoint_interpretation():
    scale = output_scale({"problem_type": "regression", "normalize": "standard"})
    result = {"biological_metadata": {"output_scale": scale}}
    assert result_output_scale(result, scale) == scale
    with pytest.raises(ValueError, match="does not match"):
        result_output_scale({"biological_metadata": {"output_scale": {**scale, "unit": "Celsius"}}}, scale)
    with pytest.raises(ValueError, match="does not match"):
        result_output_scale({}, scale)
    assert result_output_scale({})["value_space"] == "unknown"


def test_factory_response_marks_standardized_output_without_changing_csv(tmp_path, monkeypatch):
    from mcp_tools.models.factory.prediction import fintuned_operations as operations

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {"problem_type": "regression", "normalize": "standard"}
    (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(config))
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">query\nMKTAYIAKQRQISFVK\n")
    path = tmp_path / "result.csv"
    def load(args):
        args._inference_checkpoint_config = dict(config or {})
        return None, None, None, "cpu"

    module = SimpleNamespace(
        parse_fasta=lambda _: [("query", "MKTAYIAKQRQISFVK")],
        load_model_and_tokenizer=load,
        predict=lambda *args: {"prediction": -0.9709106683731079},
    )
    monkeypatch.setattr(
        operations, "_get_module", lambda _: (module, lambda *args: {"input_ids": [], "attention_mask": []})
    )
    result = json.loads(
        operations.predict_protein_function.runtime_implementation(
            str(fasta), "Optimal Temperature", "ESM2-650M", str(adapter), output_file=str(path)
        )
    )
    assert result["status"] == "success"
    assert result["biological_metadata"]["output_scale"] == output_scale(config)
    assert json.loads(result["content_preview"])[0]["prediction"] == -0.9709106683731079
    assert "-0.9709106683731079" in path.read_text() and "Celsius" not in path.read_text()


@pytest.mark.asyncio
async def test_prediction_scale_reaches_report_source_and_specialist_context(tmp_path, monkeypatch):
    from backend.web_v2 import research_predictions as predictions
    from backend.web_v2 import research_report as report
    from backend.web_v2.research_chapter_context import source_context

    scale = output_scale({"problem_type": "regression", "normalize": "standard"})
    spec = {
        "id": "temperature",
        "task": "Optimal Temperature",
        "model": "ESM2-650M",
        "dataset": "DeepET_Topt",
        "tool": "predict_protein_function",
        "adapter_path": "unused",
        "output_scale": scale,
    }
    monkeypatch.setattr(predictions, "candidates", lambda context: [spec])

    async def execute(spec, fasta, output, progress):
        output.write_text("header,sequence,prediction\nquery,MKTAYIAKQRQISFVK,-0.9709106683731079\n")
        return {"status": "success", "biological_metadata": {"output_scale": scale}}

    real_run = predictions.run_predictions

    async def predict(context, update, folder):
        return await real_run(context, update, folder, execute=execute)

    monkeypatch.setattr(predictions, "run_predictions", predict)
    monkeypatch.setattr(report, "is_online_mode", lambda: False)
    state = await report.run_research(
        {"analyses": ["ai_predictions"], "model_id": "fixture", "sequence": "MKTAYIAKQRQISFVK"},
        lambda state: None,
        synthesize=False,
        service_cache=tmp_path / "remote",
    )
    assert state["predictions"][0]["output_scale"] == scale
    assert state["sections"][0]["data"]["output_scale"] == scale
    assert json.loads(state["sources"][0]["text"])["output_scale"] == scale
    assert "standardized_target" in json.dumps(source_context(state["sources"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {"problem_type": "residue_single_label_classification", "normalize": "standard"},
        {"normalize": "standard"},
        None,
    ],
)
async def test_residue_tool_response_reaches_researcher_without_changing_config_hash_or_csv(
    tmp_path, monkeypatch, config
):
    import csv
    import io

    from backend.web_v2 import research_predictions as predictions
    from mcp_tools.models.factory.prediction import fintuned_operations as operations

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    if config is not None:
        (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(config))
    scale = checkpoint_output_scale(adapter)
    sequence = "MKTAYIAKQRQISFVK"
    predicted = {"predicted_class": [1] * len(sequence), "probabilities": [[0.2, 0.8]] * len(sequence)}
    def load(args):
        args._inference_checkpoint_config = dict(config or {})
        return None, None, None, "cpu"

    module = SimpleNamespace(
        parse_fasta=lambda _: [("query", sequence)],
        load_model_and_tokenizer=load,
        predict=lambda *args: predicted,
    )
    monkeypatch.setattr(
        operations, "_get_module", lambda _: (module, lambda *args: {"input_ids": [], "attention_mask": []})
    )
    spec = {
        "id": "binding",
        "task": "Binding Site",
        "model": "ESM2-650M",
        "dataset": operations._residue_task_to_dataset("Binding Site"),
        "tool": "predict_residue_function",
        "adapter_path": str(adapter),
        "output_scale": scale,
    }
    monkeypatch.setattr(predictions, "candidates", lambda context: [spec])
    captured = []

    async def execute(spec, fasta, output, progress):
        # Exercise the actual tool implementation's config loading, response
        # envelope and CSV writer; only neural model execution is doubled.
        result = operations.predict_residue_function.runtime_implementation(
            str(fasta), spec["task"], spec["model"], spec["adapter_path"], output_file=str(output)
        )
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["biological_metadata"]["output_scale"] == scale == output_scale(config)
        captured.append(output.read_bytes())
        return result

    folder = tmp_path / "predictions"
    rows = await predictions.run_predictions({"sequence": sequence}, lambda rows: None, folder, execute=execute)
    assert rows[0]["status"] == "completed" and rows[0]["output_scale"] == scale
    assert json.loads(rows[0]["values"][0]["predicted_class"]) == predicted["predicted_class"]
    assert json.loads(rows[0]["values"][0]["probabilities"]) == predicted["probabilities"]
    expected = io.StringIO(newline="")
    writer = csv.DictWriter(expected, fieldnames=["header", "sequence", *predicted])
    writer.writeheader()
    writer.writerow(
        {"header": "query", "sequence": sequence, **{key: json.dumps(value) for key, value in predicted.items()}}
    )
    assert (folder / "binding.csv").read_bytes() == captured[0] == expected.getvalue().encode()
    if config and "problem_type" in config:
        assert scale["value_space"] == "class_index_and_probability"
        assert scale["normalization"] == "not_applicable"
    else:
        # The loader's fallback type must not rewrite the actual config's hash.
        assert scale["value_space"] == "unknown"
