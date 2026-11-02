"""Synthetic variable-length ABI checks, including actual isolated model and MCP calls."""

import copy
import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client

from agent.rsi import generated_sequence as api
from agent.rsi.experience import ModelExperienceStore, validate_projection
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.model_tasks import inference_inputs
from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result

ROOT = Path(__file__).resolve().parents[2]
SCOPE = {
    "protein": "synthetic variable-length fixture",
    "assay": "software test",
    "target": "synthetic value",
    "unit": "AU",
    "minimum_sequence_length": 1,
    "maximum_sequence_length": 100,
}
ROWS = [{"id": f"r{i}", "sequence": "C" + "A" * i, "target": float(2 * (i + 1) + 3)} for i in range(1, 31)]
TRAIN, VALID, FUTURE = ROWS[:20], ROWS[20:25], ROWS[25:]
CODE = """def fit(sequences, targets, seed):
    lengths = [len(s) for s in sequences]
    mx = sum(lengths) / len(lengths)
    my = sum(targets) / len(targets)
    slope = sum((x-mx)*(y-my) for x,y in zip(lengths,targets)) / sum((x-mx)**2 for x in lengths)
    return {'slope': slope, 'intercept': my-slope*mx, 'seen_lengths': lengths}
def predict(model, sequences):
    return [model['intercept'] + model['slope']*len(s) for s in sequences]
"""


def lifecycle(root):
    plugin = get_model_plugin(api.PLUGIN_ID)
    return ModelLifecycle(
        root,
        {
            "schema": 1,
            "scope": SCOPE,
            "plugin_id": plugin.id,
            "plugin_sha256": plugin.source_sha256,
            "training_sha256": digest(TRAIN),
            "validation_sha256": digest(VALID),
            "minimum_skill": 0.0,
            "max_candidates": 1,
        },
    )


def test_real_fit_assess_experience_and_reuse_preserve_variable_lengths(tmp_path):
    model = api.fit_candidate(TRAIN, SCOPE, {"source_code": CODE, "seed": 123})
    assert model["model_state"]["seen_lengths"] == [len(row["sequence"]) for row in TRAIN]
    lc = lifecycle(tmp_path / "controller")
    atomic_json(tmp_path / "candidate.json", model)
    mid = lc.submit(tmp_path / "candidate.json", session_id="development", mode="science_expert")
    report = lc.assess(mid, training=TRAIN, validation=VALID)
    assert report["passed"] and report["training_replayed"] and report["numerical_evidence"]
    assert report["metrics"]["rmse"] == 0
    store = ModelExperienceStore(tmp_path / "experience")
    identity = store.publish(lc, mid, summary="Synthetic variable-length regression test")
    projected = store.project("source", [identity])
    assert projected["entries"][0]["donor_scope"] == SCOPE
    assert "model_state" not in projected["entries"][0]
    changed = copy.deepcopy(projected)
    changed["entries"][0]["donor_scope"]["minimum_sequence_length"] = 200
    with pytest.raises(ValueError):
        validate_projection(changed)
    inputs = inference_inputs(FUTURE, api.PLUGIN_ID)
    assert all(set(row) == {"id", "sequence"} for row in inputs)
    predictions = api.predict(model, inputs, SCOPE)
    assert lc.record_reuse(
        mid, session_id="reuse", mode="science_agent", inputs=inputs, predictions=predictions, scope=SCOPE
    )["rows"] == len(FUTURE)
    lc.export(mid, tmp_path / "export")
    with pytest.raises(ValueError, match="overlap"):
        api.evaluate(model, TRAIN, SCOPE)
    other = {**SCOPE, "maximum_sequence_length": 99}
    with pytest.raises(ValueError, match="applicability"):
        api.predict(model, inputs, other)


@pytest.mark.parametrize("low,high", [(0, 100), (100, 1), (True, 100), (1, 2049), (1.0, 100)])
def test_invalid_length_scope_rejected(low, high):
    with pytest.raises(ValueError):
        api.validate_scope({**SCOPE, "minimum_sequence_length": low, "maximum_sequence_length": high})


@pytest.mark.parametrize(
    "attack", ["id", "sequence", "empty", "alphabet", "long", "label", "bool", "padding", "row_type"]
)
def test_invalid_or_repeated_records_rejected(attack):
    rows = copy.deepcopy(TRAIN)
    if attack == "id":
        rows[0]["id"] = rows[1]["id"]
    elif attack == "sequence":
        rows[0]["sequence"] = rows[1]["sequence"]
    elif attack in {"empty", "alphabet", "long", "padding"}:
        rows[0]["sequence"] = {"empty": "", "alphabet": "AX", "long": "A" * 101, "padding": "AA--"}[attack]
    elif attack in {"label", "bool"}:
        rows[0]["target"] = float("nan") if attack == "label" else True
    else:
        rows[0] = "invalid"
    with pytest.raises(ValueError):
        api.validate_rows(rows, SCOPE, labels=True)


def test_wrong_prediction_shape_and_source_change_are_rejected():
    bad = CODE.replace("return [model['intercept'] + model['slope']*len(s) for s in sequences]", "return []")
    model = api.fit_candidate(TRAIN, SCOPE, {"source_code": bad, "seed": 123})
    with pytest.raises(ValueError, match="finite numeric"):
        api.predict(model, FUTURE, SCOPE)
    model["recipe"]["source_code"] = CODE
    with pytest.raises(ValueError, match="source changed"):
        api.predict(model, FUTURE, SCOPE)


@pytest.mark.asyncio
async def test_existing_scientist_bundle_runs_variable_length_development_and_reuse(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts/rsi"))
    from run_paired_generated import make_bundle

    work = tmp_path / "workspace"
    work.mkdir()
    versions, version, caps = make_bundle(
        tmp_path / "develop_bundle",
        "develop",
        {"training.json": TRAIN},
        SCOPE,
        {"provider": "test_fixture"},
        plugin_id=api.PLUGIN_ID,
    )
    versions.verify(version)
    atomic_json(work / "training.json", TRAIN)
    config = {
        "mcpServers": {
            "variable": {
                "command": sys.executable,
                "args": ["-u", str(caps / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(work)},
            }
        }
    }

    async def call(client, name, args):
        result = (await client.call_tool(name, args)).data
        validate_generated_result(name, result, workspace=work, capability=caps)
        return result

    async with Client(config) as client:
        inspected = await call(client, "inspect_model_code_task", {})
        assert inspected["sequence_layout"] == {"kind": "variable_length", "minimum": 1, "maximum": 100}
        altered = {**inspected, "sequence_layout": {"kind": "aligned"}}
        with pytest.raises(ValueError):
            validate_generated_result("inspect_model_code_task", altered, workspace=work, capability=caps)
        tested = await call(client, "evaluate_model_code", {"source_code": CODE, "rationale": "Synthetic ABI test"})
        assert tested["training_only_feedback"]["rmse"] == 0
        await call(
            client, "submit_model_code", {"source_sha256": tested["source_sha256"], "rationale": "Tested fixture"}
        )
    lc = lifecycle(tmp_path / "controller")
    mid = lc.submit(work / "candidate.json", session_id="development", mode="science_expert")
    assert lc.assess(mid, training=TRAIN, validation=VALID)["passed"]
    inputs = inference_inputs(FUTURE, api.PLUGIN_ID)
    versions, version, caps = make_bundle(
        tmp_path / "reuse_bundle",
        "reuse",
        {"inputs.json": inputs},
        SCOPE,
        {"provider": "test_fixture"},
        lc,
        mid,
        plugin_id=api.PLUGIN_ID,
    )
    versions.verify(version)
    config["mcpServers"]["variable"]["args"] = ["-u", str(caps / "generated_tools.py")]
    atomic_json(work / "inputs.json", inputs)
    async with Client(config) as client:
        await call(client, "inspect_model_code_task", {})
        await call(client, "discover_generated_models", {})
        result = await call(client, "predict_with_generated_model", {"model_id": mid})
        assert result["rows"] == len(FUTURE)
    predictions = json.loads((work / "predictions.json").read_text())
    assert [len(row["sequence"]) for row in predictions] == [len(row["sequence"]) for row in FUTURE]
    lc.record_reuse(
        mid, session_id="fresh_reuse", mode="science_expert", inputs=inputs, predictions=predictions, scope=SCOPE
    )
