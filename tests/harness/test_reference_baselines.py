"""Fixed reference controls are software fixtures, never RSI biological results."""
import hashlib
import json
import math

import numpy as np
import pytest

from agent.harness.research_reference_baselines import reference_baseline
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.research_focus import selected_tools
from agent.rsi.engine import traditional_baselines as original


def observations(start, stop, prefix):
    return [{"id": prefix + str(i), "group": prefix + str(i),
             "inputs": {"seq": "A" * (i + 3) + "G" * (50 - i)}, "target": i / 13 + math.sin(i)}
            for i in range(start, stop)]


def test_overview_is_compact_and_has_no_source_or_fitted_model():
    overview = reference_baseline()
    assert len(overview["references"]) == 4
    assert len(json.dumps(overview)) < 1800
    assert "source" not in overview and overview["kind"] == "platform_reference"
    assert overview["task"] == "regression"


@pytest.mark.parametrize("method", ["training_mean", "aac_ridge", "aac_random_forest", "aac_svr"])
def test_portable_adapter_reproduces_original_reference_predictions(method):
    item = reference_baseline(method, "seq", "score")
    assert hashlib.sha256(item["source"].encode()).hexdigest() == item["source_sha256"]
    module = {}; exec(compile(item["source"], "fixed_platform_reference", "exec"), module)
    train, query = observations(1, 20, "train"), observations(25, 30, "query")
    state = module["fit"]({"task": "regression", "train": train}, 0)
    x = original.features([{"sequence": r["inputs"]["seq"]} for r in train])
    y = np.asarray([r["target"] for r in train])
    expected = original.estimators()[method]
    expected.fit(x, (y-y.mean()) / y.std())
    q = original.features([{"sequence": r["inputs"]["seq"]} for r in query])
    truth = expected.predict(q) * y.std() + y.mean()
    actual = [module["predict"](state, row["inputs"])["score"] for row in query]
    np.testing.assert_allclose(actual, truth, rtol=1e-10, atol=1e-10)
    assert item["kind"] == "platform_reference"
    assert item["upstream_sha256"] in item["source"] and "# Platform reference: " + method in item["source"]
    with pytest.raises(ValueError, match="regression"):
        module["fit"]({"task": "classification", "train": train}, 0)
    with pytest.raises(ValueError, match="Canonical"):
        module["predict"](state, {"seq": "AB*"})


def test_column_names_are_literal_and_replacement_tokens_do_not_expand():
    column = "__OUTPUT__"
    item = reference_baseline("training_mean", column, "result")
    module = {}; exec(compile(item["source"], "fixed_platform_reference", "exec"), module)
    assert module["SEQUENCE_COLUMN"] == column
    state = module["fit"]({"task": "regression", "train": [
        {"inputs": {column: "AAA"}, "target": 2.0},
        {"inputs": {column: "GGG"}, "target": 4.0}]}, 0)
    assert module["predict"](state, {column: "AG"}) == {"result": 3.0}


@pytest.mark.parametrize("reference,column,output", [
    (None, "seq", "value"), (0, "seq", "value"), ("unknown", "seq", "value"), ("aac_ridge", "", "value"),
    ("aac_ridge", "seq", "x"*81), ("aac_ridge", "seq\nname", "value"),
])
def test_invalid_reference_or_fields_are_rejected(reference, column, output):
    with pytest.raises(ValueError):
        reference_baseline(reference, column, output)


@pytest.mark.asyncio
async def test_reference_tool_does_not_fit_register_or_write_candidate(tmp_path):
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps": []})
    tools = service.tools()
    assert "get_research_baseline" in {tool.name for tool in tools}
    context = {"plan": {"deliverable": "model"}, "model_development": {"next_action": {"stage": "evaluate_baseline"}}}
    focus = selected_tools(context, tools)
    assert set(focus) == {"prepare_research_dataset", "get_research_baseline", "train_research_model"}
    assert len(focus) == 3
    result = await service.get_research_baseline("aac_ridge", "seq")
    assert result["kind"] == "platform_reference"
    assert not list(service.workspace.iterdir())
    assert not service.development.records() and not service.proposals
    assert service.receipts[-1]["result"]["source_sha256"] == result["source_sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["aac_ridge", "aac_random_forest"])
async def test_reference_trains_and_replays_through_real_isolated_model_api(tmp_path, method):
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps": []})
    item = await service.get_research_baseline(method, "seq", output_file="src/reference.py")
    assert "source" not in item and item["source_file"] == "src/reference.py"
    assert hashlib.sha256((service.workspace/item["source_file"]).read_bytes()).hexdigest() == item["source_sha256"]
    assert not service.development.records() and not service.proposals
    data = {"task": "regression", "provenance": "Synthetic software fixture only",
            "grouping": "Disjoint fixture groups",
            "splits": {"train": observations(1, 20, "train"),
                       "validation": observations(21, 25, "validation"),
                       "test": observations(26, 30, "test")}}
    await service.write_research_file("evaluation.json", json.dumps(data))
    result = await service.train_research_model(
        method, "src/reference.py", "evaluation.json", "Fixed reference implementation; software fixture",
        variant="baseline", seed=0)
    experiment, training = result["experiment"], result["training"]
    assert experiment["fresh_process_replay"]
    assert not experiment["test_used_for_feedback"] and not training["validation_and_test_mounted"]
    assert training["training_rows"] == 19 and math.isfinite(experiment["validation"]["score"])
    assert len(service.development.records()) == 1
    assert not service.proposals


@pytest.mark.asyncio
async def test_reference_save_is_idempotent_and_never_replaces_revised_code(tmp_path):
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps":[]})
    first = await service.get_research_baseline("aac_ridge", output_file="src/reference.py")
    again = await service.get_research_baseline("aac_ridge", output_file="src/reference.py")
    assert first == again
    saved = (service.workspace/"src/reference.py").read_text()
    assert "Platform reference: aac_ridge" in saved and first["upstream_sha256"] in saved
    assert len(json.dumps(first)) < 1800 and "source" not in first
    revised = saved + "\n# Scientist revision in progress\n"
    await service.write_research_file("src/reference.py", revised)
    with pytest.raises(ValueError, match="different data"):
        await service.get_research_baseline("aac_ridge", output_file="src/reference.py")
    assert (service.workspace/"src/reference.py").read_text() == revised
    assert not service.development.records() and not service.proposals


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["../escape.py", "/tmp/escape.py", ".hidden/reference.py", "input.csv", ""])
async def test_reference_save_requires_a_selected_method_and_scoped_python_path(tmp_path, destination):
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps":[]})
    with pytest.raises((ValueError, OSError)):
        await service.get_research_baseline("" if not destination else "aac_ridge",
                                            output_file=destination or "src/no-reference.py")
    assert not list(service.workspace.rglob("*.py"))


@pytest.mark.asyncio
async def test_reference_save_cannot_follow_an_external_symlink(tmp_path):
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps":[]})
    outside = tmp_path/"outside.py";outside.write_text("preserve")
    (service.workspace/"linked.py").symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        await service.get_research_baseline("aac_ridge", output_file="linked.py")
    assert outside.read_text() == "preserve"
