"""Real MCP/numerical-worker refits and independent source-handle verification."""

import copy
import hashlib
import json
import shutil

import pytest
from fastmcp import Client
from test_generated_durable_state import checked
from test_generated_recovery import GOOD, ROOT, ROWS, SCOPE, VALID, setup_bundle
from test_training_attempt_memory import episode

from agent.harness.contracts import file_hash
from agent.rsi import training_attempt_memory as memory
from agent.rsi.experience import LIMITATION
from agent.rsi.library_source import REFIT_TOOL, resolve_library_source
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result

IDENTITY = "a" * 64


def projection(source=GOOD, condition="source"):
    entry = {
        "experience_id": IDENTITY,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "donor_scope": {**SCOPE, "protein": "different synthetic donor"},
        "method_summary": "Synthetic fixture for transport and numerical verification only",
        "acceptance": {"passed": True, "training_replayed": True},
        "parents": [],
    }
    if condition == "source":
        entry["source_code"] = source
    return {
        "schema": 1,
        "condition": condition,
        "entries": [] if condition == "none" else [entry],
        "limitation": LIMITATION,
    }


def bind(caps, packet):
    atomic_json(caps / "library_experience.json", packet)
    task = json.loads((caps / "task.json").read_text())
    task["library_experience_sha256"] = digest(packet)
    atomic_json(caps / "task.json", task)
    return task


def setup(tmp_path, **kwargs):
    caps, work, config = setup_bundle(tmp_path)
    shutil.copy2(ROOT / "src/agent/rsi/library_source.py", caps / "library_source.py")
    task = bind(caps, projection(**kwargs))
    return caps, work, config, task


@pytest.mark.asyncio
async def test_refit_restart_shared_budget_independent_acceptance_and_fresh_reuse(tmp_path):
    caps, work, config, task = setup(tmp_path)
    async with Client(config) as client:
        await checked(client, "read_model_library_experience", {}, work, caps)
        result = await checked(
            client, REFIT_TOOL, {"experience_id": IDENTITY, "rationale": "refit current task"}, work, caps
        )
    assert result["attempts_remaining"] == 1
    assert result["source_origin"]["projection_sha256"] == task["library_experience_sha256"]
    async with Client(config) as client:
        # Failed authoring shares the same budget, while the prior refit remains submittable.
        bad = await checked(
            client, "evaluate_model_code", {"source_code": "def broken(!", "rationale": "repair fixture"}, work, caps
        )
        assert bad["attempts_remaining"] == 0 and bad["submittable_sources"] == [result["source_sha256"]]
        for name, args in [(REFIT_TOOL, {"experience_id": IDENTITY}), ("evaluate_model_code", {"source_code": GOOD})]:
            with pytest.raises(Exception, match="budget exhausted"):
                await client.call_tool(name, {**args, "rationale": "third forbidden"})
        receipt = await checked(
            client,
            "submit_model_code",
            {"source_sha256": result["source_sha256"], "rationale": "select refit"},
            work,
            caps,
        )
    assert receipt["registered"] is False
    plugin = get_model_plugin("generated_python.v1")
    model = json.loads((work / "candidate.json").read_text())
    assert model == plugin.fit(ROWS, SCOPE, {"source_code": GOOD, "seed": 314159})
    provenance = json.loads((work / "source_provenance.json").read_text())
    assert provenance["origin"] == "admitted_library_source_refit"
    assert provenance["source_origin"] == result["source_origin"]
    lifecycle = ModelLifecycle(
        tmp_path / "lifecycle",
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
    model_id = lifecycle.submit(work / "candidate.json", session_id="synthetic_develop", mode="science_expert")
    assert lifecycle.assess(model_id, training=ROWS, validation=VALID)["passed"]
    inputs = [{"id": "fresh", "sequence": "HI"}]
    packet = lifecycle.export(model_id, caps / "registered")
    atomic_json(
        caps / "task.json",
        {
            "phase": "reuse",
            "scope": SCOPE,
            "inputs": {"inputs.json": digest(inputs)},
            "capability_sha256": digest(packet),
        },
    )
    atomic_json(work / "inputs.json", inputs)
    async with Client(config) as client:
        with pytest.raises(Exception, match="No library source admitted"):
            await client.call_tool(REFIT_TOOL, {"experience_id": IDENTITY, "rationale": "reuse cannot train"})
        discovered = await checked(client, "discover_generated_models", {}, work, caps)
        assert discovered["models"][0]["model_id"] == model_id
        await checked(client, "predict_with_generated_model", {"model_id": model_id}, work, caps)
    assert json.loads((work / "predictions.json").read_text()) == plugin.predict(model, inputs, SCOPE)


@pytest.mark.asyncio
async def test_source_failure_consumes_shared_attempt_then_authoring_can_repair(tmp_path):
    caps, work, config, _ = setup(tmp_path, source="def broken(!")
    async with Client(config) as client:
        failed = await checked(
            client, REFIT_TOOL, {"experience_id": IDENTITY, "rationale": "invalid fixture source"}, work, caps
        )
        assert failed["status"] == "development_failed" and failed["attempts_remaining"] == 1
        assert failed["submittable_sources"] == []
        repaired = await checked(
            client, "evaluate_model_code", {"source_code": GOOD, "rationale": "actual replacement source"}, work, caps
        )
        assert repaired["attempts_remaining"] == 0
        await checked(
            client,
            "submit_model_code",
            {"source_sha256": repaired["source_sha256"], "rationale": "select repair"},
            work,
            caps,
        )
    provenance = json.loads((work / "source_provenance.json").read_text())
    assert provenance["origin"] == "agent_tool_call_source_code" and "source_origin" not in provenance


@pytest.mark.parametrize(
    "attack", ["none", "summary", "unadmitted", "path", "unknown", "changed", "weights", "symlink", "reuse"]
)
def test_handles_cannot_escape_frozen_source_admission(tmp_path, attack):
    caps, _, _, task = setup(tmp_path, condition=attack if attack in {"none", "summary"} else "source")
    identity = IDENTITY
    if attack == "unadmitted":
        task.pop("library_experience_sha256")
    elif attack == "path":
        identity = "../../candidate.json"
    elif attack == "unknown":
        identity = "b" * 64
    elif attack == "changed":
        atomic_json(caps / "library_experience.json", projection(GOOD + "\n# altered"))
    elif attack == "weights":
        packet = projection()
        packet["entries"][0]["weights"] = [12345]
        task = bind(caps, packet)
    elif attack == "symlink":
        path = caps / "library_experience.json"
        target = caps / "outside.json"
        path.rename(target)
        path.symlink_to(target)
    elif attack == "reuse":
        task["phase"] = "reuse"
    with pytest.raises(ValueError):
        resolve_library_source(caps, task, identity)


@pytest.mark.asyncio
async def test_refit_origin_cannot_be_forged_or_presented_as_new_source(tmp_path):
    caps, work, config, _ = setup(tmp_path)
    async with Client(config) as client:
        result = await checked(
            client, REFIT_TOOL, {"experience_id": IDENTITY, "rationale": "source binding"}, work, caps
        )
    for field in ["experience_id", "projection_sha256", "source_sha256"]:
        forged = copy.deepcopy(result)
        forged["source_origin"][field] = "f" * 64
        with pytest.raises(ValueError):
            validate_generated_result(REFIT_TOOL, forged, workspace=work, capability=caps)
    with pytest.raises(ValueError, match="authoring"):
        validate_generated_result("evaluate_model_code", result, workspace=work, capability=caps)
    artifact = json.loads((work / "code_attempt_1.json").read_text())
    artifact["source_origin"]["experience_id"] = "b" * 64
    atomic_json(work / "code_attempt_1.json", artifact)
    with pytest.raises(ValueError):
        validate_generated_result(REFIT_TOOL, result, workspace=work, capability=caps)


@pytest.mark.parametrize("attack", [None, "request", "response", "artifact", "projection"])
def test_repair_memory_pairs_handle_with_exact_source_and_training_feedback(tmp_path, attack):
    protocol, cycle, folder = episode(tmp_path, sources=[GOOD, "def broken(!"], reverse_results=True)
    caps = folder / "capability/bundles/versions/version"
    declaration = bind(caps, projection())
    _, origin = resolve_library_source(caps, declaration, IDENTITY)
    artifact_path = folder / "workspace/code_attempt_1.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["source_origin"] = origin
    atomic_json(artifact_path, artifact)
    path = folder / "execution/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        if event["tool_call_id"] != "1":
            continue
        if event["kind"] == "tool_call_start":
            event["tool_name"] = "mcp__venusfactory__" + REFIT_TOOL
            event["args"] = {
                "experience_id": "b" * 64 if attack == "request" else IDENTITY,
                "rationale": artifact["rationale"],
            }
        else:
            payload = json.loads(event["output"])
            payload["source_origin"] = {**origin, "experience_id": "b" * 64} if attack == "response" else origin
            event["output"] = json.dumps(payload)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    lock_path = folder / "execution/profile.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["task"]["tools"].append(REFIT_TOOL)
    lock["task"]["capabilities"] = {p.name: file_hash(p) for p in caps.iterdir()}
    lock["sha256"] = digest({k: v for k, v in lock.items() if k != "sha256"})
    atomic_json(lock_path, lock)
    atomic_json(folder / "execution/run.json", {"status": "incomplete", "profile_sha256": lock["sha256"]})
    if attack == "artifact":
        artifact["source_origin"] = {**origin, "experience_id": "b" * 64}
        atomic_json(artifact_path, artifact)
    elif attack == "projection":
        atomic_json(caps / "library_experience.json", projection(GOOD + "\n# changed"))
    if attack:
        with pytest.raises(ValueError):
            memory.capture(folder, protocol)
    else:
        receipt = memory.capture(folder, protocol)
        assert len(receipt["attempts"]) == 2
        atomic_json(folder / "attempt_evidence.json", receipt)
        cycle["attempt_evidence_sha256"] = digest(receipt)
        projected = memory.project(tmp_path, {"cycles": [cycle]}, protocol, "develop")
        assert projected["entries"][0]["source_code"] == GOOD
        assert projected["entries"][0]["source_origin"] == origin
