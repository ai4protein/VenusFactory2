"""Real MCP/isolated numerical tests on synthetic data; zero autonomous models."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from fastmcp import Client
from test_context_integration import make_bundle, make_task
from test_generated_context import CODE

from agent.rsi import model_tasks
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.scientific_episode import artifact_written, episode_contract
from agent.rsi.engine.shared_model_bundle import admit, prepare_bundle
from agent.rsi.engine.shared_model_contracts import _state
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.rsi.engine.tool_contracts import validate_generated_result


def read(path):
    return json.loads(path.read_text())


def task_pair(root):
    tasks = []
    for number in range(2):
        task, data, _ = make_task(root / f"data_{number}", protein=f"Synthetic plant protein {number}")
        task["id"] = f"synthetic_{number}"
        if number:
            for phase, rows in data.items():
                for row in rows:
                    row["target"] = 2.0 + row["target"]
                path = Path(task["data"][phase]["path"])
                atomic_json(path, rows)
                task["data"][phase]["sha256"] = sha256_file(path)
        tasks.append(task)
    return tasks


def prepare(root, tasks, **kwargs):
    # Existing contract fixtures explicitly retain the original wire format.
    kwargs.setdefault("tool_presentation", None)
    work = root / "workspace"
    versions, version, cap = prepare_bundle(
        root / "capability", work, tasks, {"provider": "synthetic"}, template_factory=make_bundle, **kwargs
    )
    return work, cap, versions, version


def client_for(work, cap):
    return Client(
        {
            "mcpServers": {
                "batch": {
                    "command": sys.executable,
                    "args": ["-u", str(cap / "generated_tools.py")],
                    "env": {"VENUS_RSI_WORKSPACE": str(work)},
                }
            }
        }
    )


async def call(client, work, cap, name, **arguments):
    result = (await client.call_tool(name, arguments)).data
    validate_generated_result(name, result, workspace=work, capability=cap)
    return result


def test_admission_and_projection(tmp_path):
    tasks = task_pair(tmp_path)
    work, cap, versions, version = prepare(tmp_path / "develop", tasks)
    versions.verify(version)
    assert not list((cap / "targets").rglob("*.py"))
    assert (cap / "single_model_tools.py").is_file()
    assert not list(work.rglob("*validation*")) and not list(work.rglob("*future*"))
    for task in tasks:
        assert read(work / "targets" / task["id"] / "training.json") == read(Path(task["data"]["training"]["path"]))
        assert set(read(cap / "targets" / task["id"] / "task.json")["inputs"]) == {"training.json"}
    admission = read(tmp_path / "develop/capability/shared_admission.json")
    assert admission["allocated_scopes"] == 2 and admission["shared_execution_count"] == 1
    assert admission["new_autonomous_models"] == 0


@pytest.mark.parametrize("change", ["id", "scope", "split", "interface", "study", "training_hash"])
def test_rejects_invalid_shared_admission(tmp_path, change):
    tasks = task_pair(tmp_path)
    if change == "id":
        tasks[1]["id"] = tasks[0]["id"]
    elif change == "scope":
        tasks[1] = deepcopy(tasks[0])
        tasks[1]["id"] = "duplicate_scope"
    elif change == "split":
        tasks[1], _, _ = make_task(
            tmp_path / "swapped", protein="Third synthetic protein", swaps={"g0": "g4", "g4": "g0"}
        )
        tasks[1]["id"] = "swapped"
    elif change == "interface":
        tasks[1]["plugin_id"] = "generated_sequence.v1"
    elif change == "study":
        tasks[1]["paper"]["doi"] = "test:different_study"
    else:
        tasks[1]["data"]["training"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        admit(tasks)


@pytest.mark.parametrize("layout", ["same", "work_inside", "cap_inside", "symlink"])
def test_rejects_output_overlap_or_symlinks(tmp_path, layout):
    tasks = task_pair(tmp_path)
    folder, work = tmp_path / "cap", tmp_path / "work"
    if layout == "same":
        work = folder
    elif layout == "work_inside":
        work = folder / "work"
    elif layout == "cap_inside":
        folder = work / "cap"
    else:
        (tmp_path / "target").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "target", target_is_directory=True)
        work = tmp_path / "link/work"
    with pytest.raises(ValueError):
        prepare_bundle(folder, work, tasks, {}, template_factory=make_bundle)


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin", ["generated_context.v1", "generated_context.v2", "generated_context.v3"])
async def test_shared_real_mcp_independent_acceptance_and_fresh_reuse(tmp_path, plugin):
    tasks = task_pair(tmp_path)
    for task in tasks:
        task["plugin_id"] = plugin
        if plugin in {"generated_context.v2", "generated_context.v3"}:
            task["scope"]["interface_version"] = plugin
        if plugin == "generated_context.v3":
            from agent.rsi.engine.storage import sha256_file

            group = task["scope"]["grouping"]
            reference = next(item for item in task["references"] if item["sha256"] == group["evidence_sha256"])
            evidence_path = Path(reference["path"])
            evidence = read(evidence_path)
            evidence["level"] = "treatment_condition"
            atomic_json(evidence_path, evidence)
            reference["sha256"] = sha256_file(evidence_path)
            group.update(level="treatment_condition", evidence_sha256=reference["sha256"])
    feedback = {"generated_context.v2": "group_baselines.v2", "generated_context.v3": "group_baselines.v3"}.get(
        plugin, "none"
    )
    work, cap, versions, version = prepare(tmp_path / "develop", tasks, training_baseline_feedback=feedback)
    async with client_for(work, cap) as client:
        inspection = await call(client, work, cap, "inspect_model_code_task")
        trial = await call(
            client, work, cap, "evaluate_model_code", source_code=CODE, rationale="shared linear program"
        )
        assert len(trial["results"]) == 2
        assert all(t["result"]["training_only_feedback"]["rmse"] < 1e-9 for t in trial["results"])
        args = {"source_sha256": trial["source_sha256"], "rationale": "submit independently fitted models"}
        submitted = await call(client, work, cap, "submit_model_code", **args)
        assert submitted["candidate_count"] == 2 and not submitted["registered"]
        assert await call(client, work, cap, "submit_model_code", **args) == submitted
        with pytest.raises(Exception, match="arguments differ"):
            await client.call_tool("submit_model_code", {**args, "rationale": "changed"})
    # Old observations remain verifiable after later operations.
    for name, value in [("inspect_model_code_task", inspection), ("evaluate_model_code", trial)]:
        validate_generated_result(name, value, workspace=work, capability=cap)
    registrations = {}
    for task in tasks:
        data = model_tasks.load_task(task)
        lifecycle = ModelLifecycle(tmp_path / "controllers" / task["id"], model_tasks.policy_for_task(task, data))
        mid = lifecycle.submit(
            work / "targets" / task["id"] / "candidate.json", session_id="shared_develop", mode="science_expert"
        )
        assert lifecycle.assess(mid, training=data["training"], validation=data["validation"])["passed"]
        registrations[task["id"]] = {"lifecycle": lifecycle, "model_id": mid}
    first, second = [read(work / "targets" / t["id"] / "candidate.json") for t in tasks]
    assert first["source_sha256"] == second["source_sha256"]
    assert first != second
    versions.verify(version)
    reuse_work, reuse_cap, reuse_versions, reuse_version = prepare(
        tmp_path / "reuse", tasks, phase="reuse", registrations=registrations
    )
    async with client_for(reuse_work, reuse_cap) as client:
        await call(client, reuse_work, reuse_cap, "inspect_model_code_task")
        await call(client, reuse_work, reuse_cap, "discover_generated_models")
        receipt = await call(client, reuse_work, reuse_cap, "predict_with_generated_model", model_id="all")
    assert len(receipt["targets"]) == 2
    for phase, workspace, capability in [("develop", work, cap), ("reuse", reuse_work, reuse_cap)]:
        contract = episode_contract(capability, phase, set())
        assert artifact_written(workspace, contract)
    for task in tasks:
        entry = registrations[task["id"]]
        inputs = read(reuse_work / "targets" / task["id"] / "inputs.json")
        assert all("target" not in row for row in inputs)
        predictions = read(reuse_work / "targets" / task["id"] / "predictions.json")
        result = entry["lifecycle"].record_reuse(
            entry["model_id"],
            session_id="fresh_shared_reuse",
            mode="science_agent",
            inputs=inputs,
            predictions=predictions,
            scope=task["scope"],
        )
        assert result["rows"] == 10
    reuse_versions.verify(reuse_version)
    # The aggregate file cannot disagree with individually replayable outputs.
    predictions = read(reuse_work / "predictions.json")
    predictions.pop()
    atomic_json(reuse_work / "predictions.json", predictions)
    with pytest.raises(ValueError, match="Shared predictions"):
        validate_generated_result(
            "predict_with_generated_model",
            {**receipt, "prediction_sha256": digest(predictions)},
            workspace=reuse_work,
            capability=reuse_cap,
        )


@pytest.mark.asyncio
async def test_failed_target_retained_and_repair_budget(tmp_path):
    tasks = task_pair(tmp_path)
    work, cap, _, _ = prepare(tmp_path / "develop", tasks)
    bad = CODE.replace(
        "    x = np.column_stack",
        "    if float(np.mean(targets)) > 2: raise ValueError('synthetic target failure')\n    x = np.column_stack",
        1,
    )
    async with client_for(work, cap) as client:
        await call(client, work, cap, "inspect_model_code_task")
        trial = await call(
            client, work, cap, "evaluate_model_code", source_code=bad, rationale="target-conditional test"
        )
        assert [x["result"]["status"] for x in trial["results"]] == ["tested_not_published", "development_failed"]
        repaired = await call(
            client, work, cap, "evaluate_model_code", source_code=CODE, rationale="repair both targets"
        )
        assert repaired["attempts_remaining"] == 0
        with pytest.raises(Exception, match="budget exhausted"):
            await client.call_tool("evaluate_model_code", {"source_code": CODE, "rationale": "third attempt"})
        # Choosing the first attempt must retain its failed target even though a later source worked.
        submitted = await call(
            client,
            work,
            cap,
            "submit_model_code",
            source_sha256=trial["source_sha256"],
            rationale="select first source for regression check",
        )
        assert submitted["candidate_count"] == 1 and submitted["allocated_targets"] == 2
    candidate = read(work / "candidate.json")
    assert [x["status"] for x in candidate["targets"]] == ["candidate", "training_failed"]


@pytest.mark.asyncio
async def test_fail_closed_reservation_and_invalid_snapshot(tmp_path):
    tasks = task_pair(tmp_path)
    work, cap, _, _ = prepare(tmp_path / "develop", tasks)
    async with client_for(work, cap) as client:
        await call(client, work, cap, "inspect_model_code_task")
        state = read(work / ".shared_model_development.json")
        state["attempts"].append(
            {
                "number": 1,
                "source_code": CODE,
                "source_sha256": hashlib.sha256(CODE.encode()).hexdigest(),
                "rationale": "interrupted",
                "status": "reserved",
            }
        )
        atomic_json(work / ("shared_ledger_" + digest(state) + ".json"), state)
        atomic_json(work / ".shared_model_development.json", state)
        with pytest.raises(Exception, match="Interrupted shared evaluation"):
            await client.call_tool("evaluate_model_code", {"source_code": CODE, "rationale": "cannot overwrite"})
    for bad in ["../outside", "", True, "A" * 64]:
        with pytest.raises(ValueError, match="history hash"):
            _state(work, read(cap / "task.json"), bad)


@pytest.mark.asyncio
async def test_host_rejects_mutated_batch_evidence(tmp_path):
    tasks = task_pair(tmp_path)
    work, cap, _, _ = prepare(tmp_path / "develop", tasks)
    async with client_for(work, cap) as client:
        inspection = await call(client, work, cap, "inspect_model_code_task")
        trial = await call(client, work, cap, "evaluate_model_code", source_code=CODE, rationale="test")
        submitted = await call(
            client, work, cap, "submit_model_code", source_sha256=trial["source_sha256"], rationale="submit"
        )
    attacks = [
        ("inspect_model_code_task", {**inspection, "targets": inspection["targets"][:1]}),
        ("submit_model_code", {**submitted, "candidate_count": 10000}),
        ("evaluate_model_code", {**trial, "results": trial["results"][::-1]}),
    ]
    for name, receipt in attacks:
        with pytest.raises(ValueError):
            validate_generated_result(name, receipt, workspace=work, capability=cap)
    for relative in [
        "candidate.json",
        "source_provenance.json",
        "shared_attempt_1.json",
        "targets/synthetic_0/candidate.json",
        "targets/synthetic_1/code_attempt_1.json",
        "targets/synthetic_0/.model_development.json",
    ]:
        path = work / relative
        original = path.read_bytes()
        value = read(path)
        value["tampered"] = True
        atomic_json(path, value)
        try:
            with pytest.raises(ValueError):
                validate_generated_result("submit_model_code", submitted, workspace=work, capability=cap)
        finally:
            path.write_bytes(original)
    old = read(work / "shared_attempt_1.json")
    atomic_json(work / "shared_attempt_9.json", old)
    with pytest.raises(ValueError, match="coverage"):
        validate_generated_result("submit_model_code", submitted, workspace=work, capability=cap)
