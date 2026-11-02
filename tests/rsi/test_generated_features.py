"""Real isolated feature-model and MCP lifecycle checks."""

import json
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from fastmcp import Client

from agent.rsi import generated_features as api
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json
from agent.rsi.engine.tool_contracts import validate_generated_result

ROOT = Path(__file__).resolve().parents[2]
AA = "ACDEFGHIKLMNPQRSTVWY"
SCOPE = {
    "protein": "fixture",
    "assay": "synthetic",
    "target": "activity",
    "unit": "AU",
    "sequence_length": 2,
    "representation": {"name": "fixture", "dimensions": 3, "encoder_sha256": "a" * 64},
}
ROWS = [
    {
        "id": f"s{i}_{j}",
        "sequence": a + b,
        "features": [1.0, i / 10.0, j / 10.0],
        "target": float(1 + 2 * i / 10 - j / 10),
    }
    for i, a in enumerate(AA[:10])
    for j, b in enumerate(AA[:10])
]
CODE = """import numpy as np
def fit(features, targets, seed):
    w=np.linalg.lstsq(np.asarray(features),np.asarray(targets),rcond=None)[0]
    return {'weights':w.tolist()}
def predict(model, features):
    return (np.asarray(features) @ np.asarray(model['weights'])).tolist()
"""


from agent.rsi.layout import source_relative

def lifecycle(root):
    p = get_model_plugin(api.PLUGIN_ID)
    return ModelLifecycle(
        root,
        {
            "schema": 1,
            "scope": SCOPE,
            "plugin_id": p.id,
            "plugin_sha256": p.source_sha256,
            "training_sha256": digest(ROWS[:60]),
            "validation_sha256": digest(ROWS[60:80]),
            "minimum_skill": 0.0,
            "max_candidates": 1,
        },
    )


def test_real_feature_acceptance_reuse_and_changed_encoder(tmp_path):
    model = api.fit_candidate(ROWS[:60], SCOPE, {"source_code": CODE, "seed": 11})
    assert api.evaluate(model, ROWS[60:80], SCOPE)["rmse"] < 1e-9
    lc = lifecycle(tmp_path / "registry")
    atomic_json(tmp_path / "candidate.json", model)
    mid = lc.submit(tmp_path / "candidate.json", session_id="develop", mode="science_agent")
    assert lc.assess(mid, training=ROWS[:60], validation=ROWS[60:80])["passed"]
    from agent.rsi.engine.experience import ModelExperienceStore

    store = ModelExperienceStore(tmp_path / "experience")
    entry = store.publish(lc, mid, summary="Verified feature model fixture")
    projection = store.project("source", [entry])
    assert projection["entries"][0]["source_code"] == CODE
    assert "model_state" not in projection["entries"][0]
    assert projection["entries"][0]["donor_scope"]["representation"] == SCOPE["representation"]
    inputs = [{k: v for k, v in r.items() if k != "target"} for r in ROWS[80:]]
    preds = api.predict(model, inputs, SCOPE)
    assert (
        lc.record_reuse(mid, session_id="reuse", mode="science_expert", inputs=inputs, predictions=preds, scope=SCOPE)[
            "rows"
        ]
        == 20
    )
    changed = deepcopy(inputs)
    changed[0]["features"][1] += 0.1
    with pytest.raises(ValueError, match="does not match"):
        lc.record_reuse(mid, session_id="other", mode="science_agent", inputs=changed, predictions=preds, scope=SCOPE)
    other = deepcopy(SCOPE)
    other["representation"]["encoder_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="encoder mismatch"):
        api.predict(model, inputs, other)


@pytest.mark.parametrize("features", [[1.0, math.nan, 0.0], [True, 1.0, 0.0], [1.0, 2.0], "wrong"])
def test_malformed_features_rejected(features):
    row = deepcopy(ROWS[0])
    row["features"] = features
    with pytest.raises(ValueError, match="finite vector"):
        api.validate_rows([row], SCOPE, labels=True)


def test_wrong_feature_prediction_abi_rejected():
    code = CODE.replace("return (np.asarray(features) @ np.asarray(model['weights'])).tolist()", "return []")
    model = api.fit_candidate(ROWS[:60], SCOPE, {"source_code": code, "seed": 11})
    with pytest.raises(ValueError, match="finite numeric prediction"):
        api.predict(model, ROWS[60:80], SCOPE)


@pytest.mark.asyncio
async def test_real_mcp_development_and_reuse(tmp_path):
    caps, work = tmp_path / "caps", tmp_path / "work"
    caps.mkdir()
    work.mkdir()
    for name in [
        "generated_tools.py",
        "generated_model.py",
        "generated_features.py",
        "generated_classifier.py",
        "code_sandbox.py",
        "code_worker.py",
        "numeric_runtime.py",
        "protein_references.py",
        "protein_model.py",
        "experience.py",
    ]:
        shutil.copy2(ROOT / source_relative(name), caps / name)
    task = {
        "phase": "develop",
        "plugin_id": api.PLUGIN_ID,
        "scope": SCOPE,
        "inputs": {"training.json": digest(ROWS[:60])},
    }
    atomic_json(caps / "task.json", task)
    atomic_json(work / "training.json", ROWS[:60])
    config = {
        "mcpServers": {
            "features": {
                "command": sys.executable,
                "args": ["-u", str(caps / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(work)},
            }
        }
    }

    async def call(client, name, args):
        result = await client.call_tool(name, args)
        validate_generated_result(name, result.data, workspace=work, capability=caps)
        return result.data

    async with Client(config) as client:
        inspected = await call(client, "inspect_model_code_task", {})
        assert "features: list[list[float]]" in inspected["abi"]["fit"]
        assert "features" not in inspected["training"]["examples"][0]
        tested = await call(client, "evaluate_model_code", {"source_code": CODE, "rationale": "numeric fixture"})
        assert tested["training_only_feedback"]["rmse"] < 1e-9
        await call(client, "submit_model_code", {"source_sha256": tested["source_sha256"], "rationale": "tested"})
    lc = lifecycle(tmp_path / "registry")
    mid = lc.submit(work / "candidate.json", session_id="develop", mode="science_agent")
    assert lc.assess(mid, training=ROWS[:60], validation=ROWS[60:80])["passed"]
    packet = lc.export(mid, caps / "registered")
    inputs = [{k: v for k, v in row.items() if k != "target"} for row in ROWS[80:]]
    task.update(phase="reuse", inputs={"inputs.json": digest(inputs)}, capability_sha256=digest(packet))
    atomic_json(caps / "task.json", task)
    atomic_json(work / "inputs.json", inputs)
    async with Client(config) as client:
        await call(client, "discover_generated_models", {})
        response = await call(client, "predict_with_generated_model", {"model_id": mid})
        assert response["rows"] == 20
    predictions = json.loads((work / "predictions.json").read_text())
    lc.record_reuse(
        mid, session_id="new_session", mode="science_expert", inputs=inputs, predictions=predictions, scope=SCOPE
    )
    predictions[0]["features_sha256"] = "0" * 64
    atomic_json(work / "predictions.json", predictions)
    with pytest.raises(ValueError):
        validate_generated_result(
            "predict_with_generated_model",
            {**response, "prediction_sha256": digest(predictions)},
            workspace=work,
            capability=caps,
        )
