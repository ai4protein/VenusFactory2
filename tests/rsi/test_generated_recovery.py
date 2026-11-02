"""Real MCP and networkless-worker tests for bounded generated-code recovery."""

import json
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastmcp import Client

from agent.rsi.code_sandbox import guard_source
from agent.rsi.langgraph_tools import build_tools
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result

ROOT = Path(__file__).resolve().parents[2]
SCOPE = {"protein": "repair_fixture", "assay": "synthetic", "target": "linear", "unit": "AU", "sequence_length": 2}
AA = "ACDEFGHIKLMNPQRSTVWY"
ROWS = [
    {"id": a + b, "sequence": a + b, "target": float(2 * i + j)}
    for i, a in enumerate(AA[:10])
    for j, b in enumerate(AA[:4])
]
VALID = [{"id": "FG", "sequence": "FG", "target": 13.0}, {"id": "GF", "sequence": "GF", "target": 14.0}]
GOOD = """import numpy as np
AA='ACDEFGHIKLMNPQRSTVWY'
def features(sequences):
    return np.array([[1.0]+[float(AA.index(a)) for a in seq] for seq in sequences])
def fit(sequences, targets, seed):
    weights=np.linalg.lstsq(features(sequences),np.asarray(targets),rcond=None)[0]
    return {'weights':weights.tolist()}
def predict(model, sequences):
    return (features(sequences) @ np.array(model['weights'])).tolist()
"""
MEMORY_FAILURE = """import numpy as np
def fit(sequences,targets,seed):
    too_large=np.empty((1000000,1024),dtype=np.float64)
    return {'mean':float(np.mean(targets)),'shape':list(too_large.shape)}
def predict(model,sequences):
    return [model['mean'] for s in sequences]
"""


def setup_bundle(tmp_path, *, classifier=False, corrupt_worker=False):
    caps, work = tmp_path / "caps", tmp_path / "work"
    caps.mkdir()
    work.mkdir()
    for name in [
        "generated_tools.py",
        "generated_model.py",
        "generated_classifier.py",
        "code_sandbox.py",
        "code_worker.py",
        "numeric_runtime.py",
        "protein_references.py",
        "protein_model.py",
        "experience.py",
    ]:
        shutil.copy2(ROOT / "src/agent/rsi" / name, caps / name)
    if corrupt_worker:
        (caps / "code_worker.py").write_text("this is invalid python !")
    scope = dict(SCOPE)
    rows = ROWS
    task = {"phase": "develop", "scope": scope, "inputs": {"training.json": digest(rows)}}
    if classifier:
        rows = [{**row, "target": int(row["target"] >= 10)} for row in ROWS]
        scope.update(target="class", unit="probability", class_labels=["low", "high"])
        task.update(plugin_id="generated_classifier.v1", inputs={"training.json": digest(rows)})
    atomic_json(caps / "task.json", task)
    atomic_json(work / "training.json", rows)
    config = {
        "mcpServers": {
            "generated": {
                "command": sys.executable,
                "args": ["-u", str(caps / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(work)},
            }
        }
    }
    return caps, work, config


async def wrapped(client, caps, work):
    counters = {"tool_calls": 0, "tool_errors": 0}
    events = []
    tools = await build_tools(
        client,
        allowed_tools={"evaluate_model_code", "submit_model_code"},
        workspace=work,
        capability=caps,
        deadline=time.monotonic() + 120,
        max_tool_calls=8,
        result=counters,
        event=events.append,
    )
    return {t.name: t for t in tools}, counters, events


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_first", [True, False])
async def test_real_resource_failure_repair_and_fallback_stay_within_two_attempts(tmp_path, bad_first):
    caps, work, config = setup_bundle(tmp_path)
    async with Client(config) as client:
        tools, counters, events = await wrapped(client, caps, work)
        outputs = []
        for source in [MEMORY_FAILURE, GOOD] if bad_first else [GOOD, MEMORY_FAILURE]:
            output = json.loads(
                await tools["evaluate_model_code"].ainvoke(
                    {"source_code": source, "rationale": "test bounded recovery"}
                )
            )
            outputs.append(output)
        failure = outputs[0 if bad_first else 1]
        assert failure["status"] == "development_failed" and failure["training_only_feedback"]["stage"] == "execution"
        assert "allocate" in failure["training_only_feedback"]["message"]
        good_sha = guard_source(GOOD)["source_sha256"]
        assert failure["submittable_sources"] == ([] if bad_first else [good_sha])
        assert outputs[-1]["attempts_remaining"] == 0
        # An additional generation attempt remains forbidden after either ordering.
        with pytest.raises(Exception, match="budget exhausted"):
            await client.call_tool("evaluate_model_code", {"source_code": GOOD, "rationale": "forbidden third attempt"})
        with pytest.raises(Exception, match="first pass"):
            await client.call_tool(
                "submit_model_code",
                {
                    "source_sha256": guard_source(MEMORY_FAILURE)["source_sha256"],
                    "rationale": "failed source cannot submit",
                },
            )
        submitted = json.loads(
            await tools["submit_model_code"].ainvoke(
                {"source_sha256": good_sha, "rationale": "submit successful source"}
            )
        )
        assert submitted["status"] == "candidate" and submitted["registered"] is False
        assert counters == {"tool_calls": 3, "tool_errors": 0}
        assert any(
            '"status": "development_failed"' in e.get("output", "")
            or '"status":"development_failed"' in e.get("output", "")
            for e in events
        )
    plugin = get_model_plugin("generated_python.v1")
    lc = ModelLifecycle(
        tmp_path / "registry",
        {
            "schema": 1,
            "scope": SCOPE,
            "plugin_id": plugin.id,
            "plugin_sha256": plugin.source_sha256,
            "training_sha256": digest(ROWS),
            "validation_sha256": digest(VALID),
            "minimum_skill": 0.0,
            "max_candidates": 1,
        },
    )
    model_id = lc.submit(work / "candidate.json", session_id="repaired", mode="science_agent")
    assert lc.assess(model_id, training=ROWS, validation=VALID)["passed"]
    attempts = [json.loads((work / f"code_attempt_{n}.json").read_text()) for n in [1, 2]]
    assert sum(a.get("status") == "development_failed" for a in attempts) == 1


@pytest.mark.asyncio
async def test_bad_probability_feedback_is_recoverable_but_not_submittable(tmp_path):
    caps, work, config = setup_bundle(tmp_path, classifier=True)
    code = "def fit(sequences,targets,seed):\n    return {}\ndef predict(model,sequences):\n    return [[0.2,0.2] for s in sequences]\n"
    async with Client(config) as client:
        tools, _, _ = await wrapped(client, caps, work)
        result = json.loads(
            await tools["evaluate_model_code"].ainvoke(
                {"source_code": code, "rationale": "invalid class probabilities"}
            )
        )
        assert result["status"] == "development_failed"
        assert result["training_only_feedback"]["stage"] == "prediction_validation"
        assert result["attempts_remaining"] == 1 and result["submittable_sources"] == []
        forged = dict(result, status="tested_not_published")
        with pytest.raises(ValueError):
            validate_generated_result("evaluate_model_code", forged, workspace=work, capability=caps)


@pytest.mark.asyncio
async def test_infrastructure_failure_is_not_candidate_feedback(tmp_path):
    caps, work, config = setup_bundle(tmp_path, corrupt_worker=True)
    async with Client(config) as client:
        tools, counters, events = await wrapped(client, caps, work)
        with pytest.raises(Exception, match="generated model failed"):
            await tools["evaluate_model_code"].ainvoke({"source_code": GOOD, "rationale": "worker cannot start"})
        assert counters["tool_errors"] == 1 and events[-1]["is_error"] is True
    assert not list(work.glob("code_attempt_*.json"))


@pytest.mark.asyncio
async def test_changed_training_input_remains_fatal(tmp_path):
    caps, work, config = setup_bundle(tmp_path)
    atomic_json(work / "training.json", ROWS[:-1])
    async with Client(config) as client:
        tools, counters, events = await wrapped(client, caps, work)
        with pytest.raises(Exception, match="frozen task input changed"):
            await tools["evaluate_model_code"].ainvoke({"source_code": GOOD, "rationale": "input must be bound"})
        assert counters["tool_errors"] == 1 and events[-1]["is_error"] is True
    assert not list(work.glob("code_attempt_*.json"))
