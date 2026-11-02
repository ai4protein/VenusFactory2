"""Native MCP and Agent boundary checks on synthetic assays; no LLM requests."""

import json
import sys

import pytest
from fastmcp import Client
from test_context_integration import make_bundle
from test_generated_antibody import CODE, make_task

from agent.rsi import generated_antibody as api
from agent.rsi import model_tasks
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result


def client_for(work, cap):
    return Client(
        {
            "mcpServers": {
                "assay": {
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


@pytest.mark.asyncio
async def test_native_development_and_reuse_roundtrip(tmp_path):
    task, parts = make_task(tmp_path / "data")
    work = tmp_path / "development_episode" / "workspace"
    work.mkdir(parents=True)
    atomic_json(work / "training.json", parts["training"])
    versions, version, cap = make_bundle(
        tmp_path / "develop_capability",
        "develop",
        {"training.json": parts["training"]},
        task["scope"],
        {"fixture": "synthetic_antibody"},
        plugin_id=api.PLUGIN_ID,
    )
    versions.verify(version)
    assert (cap / "generated_antibody.py").is_file()
    assert not list(cap.rglob("*validation*")) and not list(cap.rglob("*future*"))
    async with client_for(work, cap) as client:
        info = await call(client, work, cap, "inspect_model_code_task")
        assert info
        assert info["representation"] == task["scope"]["representation"]
        result = await call(
            client, work, cap, "evaluate_model_code", source_code=CODE, rationale="Synthetic numerical lifecycle test"
        )
        assert result["status"] == "tested_not_published"
        await call(
            client,
            work,
            cap,
            "submit_model_code",
            source_sha256=result["source_sha256"],
            rationale="Synthetic numerical lifecycle test",
        )
    controller = ModelLifecycle(
        tmp_path / "development_episode" / "controller", model_tasks.policy_for_task(task, parts)
    )
    mid = controller.submit(work / "candidate.json", session_id="synthetic_native_develop", mode="science_expert")
    report = controller.assess(mid, training=parts["training"], validation=parts["validation"])
    assert report["passed"]
    reuse = tmp_path / "reuse_episode" / "workspace"
    reuse.mkdir(parents=True)
    inputs = api.inference_inputs(parts["future"])
    atomic_json(reuse / "inputs.json", inputs)
    versions, version, cap = make_bundle(
        tmp_path / "reuse_capability",
        "reuse",
        {"inputs.json": inputs},
        task["scope"],
        {"fixture": "synthetic_antibody"},
        lifecycle=controller,
        model_id=mid,
        plugin_id=api.PLUGIN_ID,
    )
    async with client_for(reuse, cap) as client:
        result = await call(client, reuse, cap, "predict_with_generated_model", model_id=mid)
        assert result["rows"] == len(inputs)
    predictions = json.loads((reuse / "predictions.json").read_text())
    assert (
        controller.record_reuse(
            mid,
            session_id="synthetic_native_reuse",
            mode="science_expert",
            inputs=inputs,
            predictions=predictions,
            scope=task["scope"],
        )["event"]
        == "reused"
    )

    from agent.rsi.scientist_model_library import _model

    receipt = controller._state()["events"][-1]
    atomic_json(reuse.parent / "reuse_receipt.json", receipt)
    record = {
        "model_id": mid,
        "controller": "development_episode/controller",
        "reuse_receipt": "reuse_episode/reuse_receipt.json",
    }
    episodes = {
        "development_episode": {"phase": "develop", "session_id": "synthetic_native_develop"},
        "reuse_episode": {"phase": "reuse", "session_id": "synthetic_native_reuse"},
    }
    audited = _model(
        tmp_path, record, {"scope": task["scope"], "model_plugin_id": api.PLUGIN_ID}, parts, episodes, 0.0, True
    )
    assert audited["skill"] > 0.999999
    assert audited["packet"]["model"]["plugin_id"] == api.PLUGIN_ID


def test_agent_describes_full_assay_inputs_and_uses_plugin(tmp_path):
    from mcp_tools.models.evolved import service

    task, parts = make_task(tmp_path / "data")
    plugin = get_model_plugin(api.PLUGIN_ID)
    model = plugin.fit(parts["training"], task["scope"], {"source_code": CODE, "seed": 19})

    class Library:
        def resolve(self, entry_id):
            return {
                "key": {"plugin_id": api.PLUGIN_ID, "scope": task["scope"]},
                "origin": {"stage": "kinetics", "paper": task["paper"]},
                "packet": {"assessment": {"metrics": {"fixture": True}}},
            }

        def predict(self, entry_id, inputs, scope, session_id):
            return {"predictions": plugin.predict(model, inputs, scope), "entry_id": entry_id}

    library = Library()
    card = service.describe_model(library, "synthetic_entry")
    assert card["input_fields"] == list(api.INPUT_FIELDS)
    assert card["input_contract"]["kind"] == "paired_antibody_measurement"
    assert card["input_contract"]["representation"] == task["scope"]["representation"]
    assert card["input_contract"]["target_required"] is False
    inputs = api.inference_inputs(parts["future"])
    result = service.predict_model(library, "synthetic_entry", inputs, task["scope"])
    assert len(result["predictions"]) == len(inputs)


@pytest.mark.asyncio
async def test_shared_paired_antibody_program_develops_and_reuses_two_targets(tmp_path):
    from test_shared_model_program import prepare

    tasks = [make_task(tmp_path / "egfr", "P00533")[0], make_task(tmp_path / "erbb2", "P04626")[0]]
    work, cap, versions, version = prepare(tmp_path / "shared_develop", tasks)
    async with client_for(work, cap) as client:
        info = await call(client, work, cap, "inspect_model_code_task")
        assert info
        result = await call(
            client,
            work,
            cap,
            "evaluate_model_code",
            source_code=CODE,
            rationale="Synthetic paired antibody numerical integration",
        )
        assert all(r["result"]["status"] == "tested_not_published" for r in result["results"])
        submitted = await call(
            client,
            work,
            cap,
            "submit_model_code",
            source_sha256=result["source_sha256"],
            rationale="Synthetic source tested",
        )
        assert submitted["candidate_count"] == 2
    versions.verify(version)
    registrations = {}
    for task in tasks:
        parts = model_tasks.load_task(task)
        lifecycle = ModelLifecycle(tmp_path / "controllers" / task["id"], model_tasks.policy_for_task(task, parts))
        mid = lifecycle.submit(
            work / "targets" / task["id"] / "candidate.json",
            session_id="synthetic_paired_develop",
            mode="science_expert",
        )
        assert lifecycle.assess(mid, training=parts["training"], validation=parts["validation"])["passed"]
        registrations[task["id"]] = {"lifecycle": lifecycle, "model_id": mid}
    work, cap, versions, version = prepare(tmp_path / "shared_reuse", tasks, phase="reuse", registrations=registrations)
    async with client_for(work, cap) as client:
        result = await call(client, work, cap, "predict_with_generated_model", model_id="all")
        assert len(result["targets"]) == 2
    for task in tasks:
        row = registrations[task["id"]]
        folder = work / "targets" / task["id"]
        inputs, predictions = (json.loads((folder / name).read_text()) for name in ["inputs.json", "predictions.json"])
        assert all("target" not in r for r in inputs)
        receipt = row["lifecycle"].record_reuse(
            row["model_id"],
            session_id="synthetic_paired_reuse",
            mode="science_agent",
            inputs=inputs,
            predictions=predictions,
            scope=task["scope"],
        )
        assert receipt["event"] == "reused" and receipt["rows"] == 24
    versions.verify(version)


@pytest.mark.asyncio
@pytest.mark.parametrize("evolving", [False, True])
async def test_antibody_goal_reaches_harness_with_training_only_bundle(tmp_path, monkeypatch, evolving):
    import run_science_goal
    from run_science_evolution import EvolvingModelGoal
    from test_model_tasks import prepared

    from agent.harness import Profile
    from agent.rsi.protein_model import digest

    definition, data = make_task(tmp_path / "data")
    store, driver = prepared(tmp_path, definition)
    if evolving:
        protocol = json.loads((store.root / "protocol.json").read_text())
        protocol["evolution"] = {"schema": 2, "generations": 2}
        atomic_json(store.root / "protocol.json", protocol)
        driver = EvolvingModelGoal(store, None)

    class HostReached(Exception):
        pass

    async def inspect_host(profile, task, runtimes, *, credential):
        assert task.phase == "develop" and credential is None
        bundle = json.loads((task.capability / "task.json").read_text())
        assert bundle["plugin_id"] == api.PLUGIN_ID
        assert bundle["scope"] == definition["scope"]
        assert bundle["inputs"] == {"training.json": digest(data["training"])}
        assert (task.capability / "generated_antibody.py").is_file()
        assert sorted(p.name for p in task.workspace.iterdir()) == ["training.json"]
        assert ("read_model_experience" in task.tools) == evolving
        raise HostReached

    monkeypatch.setattr(run_science_goal, "execute", inspect_host)
    state = store.read()
    with pytest.raises(HostReached):
        await driver.cycle(
            0, Profile.from_dict(state["definition"]["profile"]), state, "Synthetic antibody integration"
        )


def test_antibody_catalog_admission_uses_observed_scope_and_group_counts(tmp_path):
    from pathlib import Path

    from agent.rsi.research_project_catalog import inspect_candidate
    from agent.rsi.storage import sha256_file

    task, parts = make_task(tmp_path / "data")
    groups_path = tmp_path / "catalog_groups.json"
    atomic_json(
        groups_path,
        {row["id"]: {"group": row["group"], "phase": phase} for phase, rows in parts.items() for row in rows},
    )
    task_path = Path(task["data"]["training"]["path"]).parent / "task.json"
    candidate = {
        "id": "synthetic_antibody",
        "project_id": "synthetic_project",
        "task": {"path": str(task_path), "sha256": sha256_file(task_path)},
        "protein_scope": {"kind": "project", "identifier": task["scope"]["protein"]},
        "property": task["scope"]["target"],
        "scenario": task["scope"]["context"],
        "question": "Synthetic endpoint regression",
        "groups": {"path": str(groups_path), "sha256": sha256_file(groups_path)},
        "group_basis": task["scope"]["grouping"]["level"],
        "scenario_evidence": task["references"][0],
    }
    record, loaded = inspect_candidate(candidate, {"study_doi": task["paper"]["doi"]})
    assert loaded == parts
    assert record["status"] == "admitted"
    assert record["rows"] == {"training": 64, "validation": 24, "future": 24}
    assert record["groups"] == {"training": 8, "validation": 3, "future": 3}
    assert record["independence_profile"] == "antibody_patent_groups.v1"
    assert len(record["baseline_plan"]["implemented"]) == 5
    assert not record["homology_independence_established"]
    assert not record["baseline_plan"]["domain_comparison_complete"]


def test_shared_paired_tasks_reject_cross_task_leakage_even_with_renamed_groups(tmp_path):
    from pathlib import Path

    from agent.rsi.shared_model_bundle import admit
    from agent.rsi.storage import sha256_file

    tasks = [make_task(tmp_path / "egfr", "P00533")[0], make_task(tmp_path / "erbb2", "P04626")[0]]
    task = tasks[1]
    parts = model_tasks.load_task(task)
    parts["training"], parts["validation"] = parts["validation"], parts["training"]
    for phase, rows in parts.items():
        for row in rows:
            row["group"] += "_renamed"
        path = Path(task["data"][phase]["path"])
        atomic_json(path, rows)
        task["data"][phase]["sha256"] = sha256_file(path)
    grouping = next(ref for ref in task["references"] if ref["sha256"] == task["scope"]["grouping"]["evidence_sha256"])
    path = Path(grouping["path"])
    evidence = json.loads(path.read_text())
    evidence["observations"] = {
        r["id"]: {"group": r["group"], "phase": phase, "input_sha256": api.input_digest(r)}
        for phase, rows in parts.items()
        for r in rows
    }
    atomic_json(path, evidence)
    grouping["sha256"] = sha256_file(path)
    task["scope"]["grouping"]["evidence_sha256"] = grouping["sha256"]
    model_tasks.load_task(task)
    with pytest.raises(ValueError, match="Cross-task partition conflict"):
        admit(tasks)


def test_full_catalog_admits_native_paired_antibody_scopes(tmp_path):
    from pathlib import Path

    from agent.rsi.poisk_tasks import binding
    from agent.rsi.research_project_catalog import build_catalog

    tasks = [make_task(tmp_path / "egfr", "P00533")[0], make_task(tmp_path / "erbb2", "P04626")[0]]
    candidates = []
    for task in tasks:
        parts = model_tasks.load_task(task)
        path = tmp_path / (task["id"] + "_groups.json")
        atomic_json(
            path, {row["id"]: {"group": row["group"], "phase": phase} for phase, rows in parts.items() for row in rows}
        )
        candidates.append(
            {
                "id": task["id"],
                "project_id": "synthetic_project",
                "task": binding(Path(task["data"]["training"]["path"]).parent / "task.json"),
                "protein_scope": {"kind": "protein", "identifier": task["scope"]["protein"]},
                "property": task["scope"]["target"],
                "scenario": task["scope"]["context"],
                "question": "Synthetic paired-antibody regression",
                "groups": binding(path),
                "group_basis": task["scope"]["grouping"]["level"],
                "scenario_evidence": task["references"][0],
            }
        )
    manifest = {
        "schema": "research_project_catalog.v1",
        "target_models": 10000,
        "projects": [
            {
                "id": "synthetic_project",
                "title": "Synthetic fixture",
                "url": "https://example.org/fixture",
                "study_doi": tasks[0]["paper"]["doi"],
                "source_evidence": tasks[0]["references"][0],
                "license_evidence": tasks[0]["references"][0],
                "license": "Synthetic fixture",
            }
        ],
        "candidates": candidates,
    }
    report = build_catalog(manifest)
    assert report["summary"]["statuses"] == {"admitted": 2}
    assert report["summary"]["new_published_models"] == 0
    assert all(row["independence_profile"] == "antibody_patent_groups.v1" for row in report["entries"])
