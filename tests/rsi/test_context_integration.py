"""Synthetic contextual tasks: real MCP computation, no acquired scientific models."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from fastmcp import Client
from test_generated_context import CODE, ROWS, SCOPE

from agent.harness import Profile
from agent.rsi import generated_context as api
from agent.rsi import model_tasks
from agent.rsi.engine.experience import ModelExperienceStore
from agent.rsi.engine.library_experience import IneligibleDonor, snapshot
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.protein_model import digest
from agent.rsi.datasets.research_project_catalog import SCHEMA, build_catalog
from agent.rsi.engine.scientist_model_library import _model
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.rsi.engine.tool_contracts import validate_generated_result

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
from run_paired_generated import make_bundle
from run_science_evolution import EvolvingModelGoal
from test_model_tasks import prepared


def save(path, value):
    atomic_json(path, value)
    return {"path": str(path), "sha256": sha256_file(path)}


def make_task(folder, protein=SCOPE["protein"], swaps=None):
    folder.mkdir(parents=True, exist_ok=True)
    source = save(folder / "source.json", {"kind": "synthetic six independent sample groups"})
    data, assignments, assets = {}, {}, {}
    for phase, start, end in [("training", 0, 40), ("validation", 40, 50), ("future", 50, 60)]:
        rows = deepcopy(ROWS[start:end])
        for row in rows:
            row["protein"] = protein
            if swaps:
                row["group"] = swaps.get(row["group"], row["group"])
            row["id"] = protein + "_" + row["id"]
            assignments[row["id"]] = {"group": row["group"], "phase": phase}
        data[phase] = rows
        assets[phase] = save(folder / (phase + ".json"), rows)
    evidence = save(
        folder / "group_evidence.json",
        {
            "schema": "context_groups.v1",
            "level": "biological_replicate",
            "study_doi": "test:plant_fixture",
            "sources": [source],
            "observations": assignments,
        },
    )
    scope = deepcopy(SCOPE)
    scope["protein"] = protein
    scope["grouping"]["evidence_sha256"] = evidence["sha256"]
    task = {
        "schema": 1,
        "id": "synthetic_context",
        "stage": "stability",
        "plugin_id": api.PLUGIN_ID,
        "scope": scope,
        "paper": {"title": "Synthetic plant context", "doi": "test:plant_fixture"},
        "references": [source, evidence],
        "data": assets,
        "minimum_skill": 0.0,
    }
    return task, data, assignments


@pytest.fixture
def contextual(tmp_path):
    return make_task(tmp_path / "task_assets")


def test_goal_preparation_and_group_evidence(contextual, tmp_path):
    task, data, _ = contextual
    assert model_tasks.load_task(task) == data
    store, driver = prepared(tmp_path, task)
    assert driver.model_plugin_id == api.PLUGIN_ID
    assert driver.inference_inputs() == api.inference_inputs(data["future"])
    assert all("target" not in r for r in driver.inference_inputs())
    _, split = model_tasks.source_and_split(task)
    assert split["observation_and_group_disjoint"]
    assert "identity_and_exact_sequence_disjoint" not in split
    assert store.root.is_dir()


@pytest.mark.parametrize(
    "attack", ["unbound", "assignment", "level", "study", "source", "extra_observation", "overlap"]
)
def test_context_task_rejects_unbound_independence(contextual, attack):
    task, data, _ = contextual
    if attack == "unbound":
        task["references"].pop()
    elif attack == "overlap":
        data["future"][0]["group"] = data["training"][0]["group"]
        task["data"]["future"] = save(Path(task["data"]["future"]["path"]), data["future"])
    else:
        path = Path(task["references"][1]["path"])
        evidence = json.loads(path.read_text())
        if attack == "assignment":
            evidence["observations"][data["future"][0]["id"]]["group"] = "unrecorded"
        elif attack == "level":
            evidence["level"] = "workflow_replicate"
        elif attack == "study":
            evidence["study_doi"] = "test:other"
        elif attack == "source":
            evidence["sources"] = []
        else:
            evidence["observations"]["invented"] = {"group": "fake", "phase": "training"}
        task["references"][1] = save(path, evidence)
        task["scope"]["grouping"]["evidence_sha256"] = task["references"][1]["sha256"]
    with pytest.raises(ValueError):
        model_tasks.load_task(task)


def catalog_manifest(folder, tasks):
    candidates = []
    for number, (task, _, assignments) in enumerate(tasks):
        candidates.append(
            {
                "id": f"candidate_{number}",
                "project_id": "plant_fixture",
                "task": save(folder / f"task_{number}.json", task),
                "protein_scope": {"kind": "protein", "identifier": task["scope"]["protein"]},
                "property": task["scope"]["target"],
                "scenario": task["scope"]["context"],
                "question": "Predict the response in an independent synthetic sample",
                "groups": save(folder / f"groups_{number}.json", assignments),
                "group_basis": "replicate_group",
                "scenario_evidence": task["references"][0],
            }
        )
    return {
        "schema": SCHEMA,
        "target_models": 10000,
        "projects": [
            {
                "id": "plant_fixture",
                "title": "Synthetic plant context",
                "url": "https://example.org/plant",
                "study_doi": "test:plant_fixture",
                "source_evidence": tasks[0][0]["references"][0],
                "license_evidence": tasks[0][0]["references"][0],
                "license": "Synthetic fixture",
            }
        ],
        "candidates": candidates,
    }


def test_catalog_counts_two_protein_scopes_and_rejects_cross_task_group_swap(tmp_path):
    first = make_task(tmp_path / "first")
    second = make_task(tmp_path / "second", protein="Second synthetic protein")
    report = build_catalog(catalog_manifest(tmp_path, [first, second]))
    assert report["summary"]["statuses"] == {"admitted": 2}
    assert report["summary"]["new_published_models"] == 0
    assert report["admission_policy"]["minimum_groups"] == {"training": 5, "validation": 3, "future": 3}
    assert report["entries"][0]["independence_profile"] == "context_response_groups.v1"
    assert len(report["entries"][0]["baseline_plan"]["implemented"]) == 6
    changed = make_task(tmp_path / "third", protein="Second synthetic protein", swaps={"g0": "g4", "g4": "g0"})
    report = build_catalog(catalog_manifest(tmp_path, [first, changed]))
    assert report["summary"]["statuses"] == {"admitted": 1, "deferred": 1}
    assert report["entries"][1]["reasons"] == ["cross_task_group_partition_conflict"]


def test_catalog_rejects_unbound_context_and_duplicate_repackaging(contextual, tmp_path):
    manifest = catalog_manifest(tmp_path, [contextual, contextual])
    assert build_catalog(manifest)["summary"]["statuses"] == {"admitted": 1, "duplicate": 1}
    manifest["candidates"][0]["scenario"] = {"material": "unrecorded condition"}
    assert build_catalog(manifest)["entries"][0]["status"] == "invalid"


def test_cross_task_context_source_requires_group_overlap_protocol(contextual, tmp_path):
    task, _, _ = contextual
    with pytest.raises(IneligibleDonor, match="study/group overlap protocol"):
        snapshot({"condition": "source", "library_root": str(tmp_path / "library"), "entry_ids": ["a" * 64]}, task)


@pytest.mark.asyncio
async def test_real_context_mcp_fit_register_reuse_and_publication_replay(contextual, tmp_path):
    task_definition, data, _ = contextual
    scope = task_definition["scope"]
    root = tmp_path / "goal"
    develop, reuse = root / "episodes/cycle_000", root / "episodes/cycle_001"
    work = develop / "workspace"
    work.mkdir(parents=True)
    versions, version, caps = make_bundle(
        develop / "capability",
        "develop",
        {"training.json": data["training"]},
        scope,
        {"provider": "synthetic"},
        plugin_id=api.PLUGIN_ID,
    )
    atomic_json(work / "training.json", data["training"])

    def config(workspace):
        return {
            "mcpServers": {
                "context": {
                    "command": sys.executable,
                    "args": ["-u", str(caps / "generated_tools.py")],
                    "env": {"VENUS_RSI_WORKSPACE": str(workspace)},
                }
            }
        }

    async def call(client, name, args, workspace=work):
        result = await client.call_tool(name, args)
        validate_generated_result(name, result.data, workspace=workspace, capability=caps)
        return result.data

    async with Client(config(work)) as client:
        inspection = await call(client, "inspect_model_code_task", {})
        assert inspection["covariates"] == scope["covariates"]
        assert "residue_reference_catalog" not in inspection
        trial = await call(
            client, "evaluate_model_code", {"source_code": CODE, "rationale": "synthetic numeric fixture"}
        )
        assert trial["training_only_feedback"]["independent_groups"] == 1
        assert trial["training_only_feedback"]["rmse"] < 1e-9
        await call(
            client, "submit_model_code", {"source_sha256": trial["source_sha256"], "rationale": "tested fixture"}
        )
    policy = model_tasks.policy_for_task(task_definition, data)
    lifecycle = ModelLifecycle(develop / "controller", policy)
    mid = lifecycle.submit(work / "candidate.json", session_id="development", mode="science_expert")
    assert lifecycle.assess(mid, training=data["training"], validation=data["validation"])["passed"]
    experience = ModelExperienceStore(tmp_path / "experience")
    entry = experience.publish(lifecycle, mid, summary="Synthetic context fixture")
    assert experience.project("source", [entry])["entries"][0]["donor_scope"] == scope
    versions.verify(version)
    inputs = model_tasks.inference_inputs(data["future"], api.PLUGIN_ID)
    reuse_versions, reuse_version, caps = make_bundle(
        reuse / "capability",
        "reuse",
        {"inputs.json": inputs},
        scope,
        {"provider": "synthetic"},
        lifecycle,
        mid,
        plugin_id=api.PLUGIN_ID,
    )
    reuse_work = reuse / "workspace"
    atomic_json(reuse_work / "inputs.json", inputs)
    async with Client(config(reuse_work)) as client:
        await call(client, "discover_generated_models", {}, reuse_work)
        response = await call(client, "predict_with_generated_model", {"model_id": mid}, reuse_work)
    reuse_versions.verify(reuse_version)
    predictions = json.loads((reuse_work / "predictions.json").read_text())
    receipt = lifecycle.record_reuse(
        mid, session_id="reuse", mode="science_expert", inputs=inputs, predictions=predictions, scope=scope
    )
    atomic_json(reuse / "reuse_receipt.json", receipt)
    record = {
        "model_id": mid,
        "controller": "episodes/cycle_000/controller",
        "reuse_receipt": "episodes/cycle_001/reuse_receipt.json",
    }
    episodes = {
        "episodes/cycle_000": {"phase": "develop", "session_id": "development"},
        "episodes/cycle_001": {"phase": "reuse", "session_id": "reuse"},
    }
    audited = _model(root, record, {"scope": scope, "model_plugin_id": api.PLUGIN_ID}, data, episodes, 0.0, True)
    assert audited["skill"] > 0.99
    predictions[0]["values_sha256"] = "0" * 64
    atomic_json(reuse_work / "predictions.json", predictions)
    with pytest.raises(ValueError, match="Context prediction identity"):
        validate_generated_result(
            "predict_with_generated_model",
            {**response, "prediction_sha256": digest(predictions)},
            workspace=reuse_work,
            capability=caps,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("evolving", [False, True])
async def test_context_goal_reaches_host_with_production_bundle(contextual, tmp_path, monkeypatch, evolving):
    import run_science_goal

    definition, data, _ = contextual
    store, driver = prepared(tmp_path, definition)
    if evolving:
        protocol = json.loads((store.root / "protocol.json").read_text())
        protocol["evolution"] = {"schema": 2, "generations": 2}
        atomic_json(store.root / "protocol.json", protocol)
        driver = EvolvingModelGoal(store, None)

    class HostReached(Exception):
        pass

    async def inspect_host(profile, task, runtimes, *, credential):
        assert task.phase == "develop"
        assert credential is None
        bundle = json.loads((task.capability / "task.json").read_text())
        assert bundle["plugin_id"] == api.PLUGIN_ID
        assert bundle["scope"] == definition["scope"]
        assert bundle["inputs"] == {"training.json": digest(data["training"])}
        assert (task.capability / "generated_context.py").is_file()
        assert sorted(p.name for p in task.workspace.iterdir()) == ["training.json"]
        assert ("read_model_experience" in task.tools) == evolving
        raise HostReached

    monkeypatch.setattr(run_science_goal, "execute", inspect_host)
    state = store.read()
    with pytest.raises(HostReached):
        await driver.cycle(0, Profile.from_dict(state["definition"]["profile"]), state, "Synthetic integration check")
