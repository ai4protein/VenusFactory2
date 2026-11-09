"""Training-only comparisons, frozen intervention binding and real MCP execution."""

import importlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastmcp import Client
from test_context_integration import make_task
from test_generated_context import CODE, ROWS, SCOPE

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi import context_feedback as feedback
from agent.rsi import generated_context as api
from agent.rsi.campaigns.scientist_model_campaign import admit_allocations, bind_goal
from agent.rsi.engine.context_controls import score_rows
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.rsi.engine.tool_contracts import validate_generated_result

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
from run_paired_generated import make_bundle
from run_science_evolution import EvolvingModelGoal
from run_science_goal import GeneratedModelGoal, prepare


def candidate(rows):
    train, dev = api.development_split(rows)
    truth = np.array([r["target"] for r in dev])
    mean = np.mean([r["target"] for r in train])
    return {
        "rows": len(dev),
        "independent_groups": len({r["group"] for r in dev}),
        "rmse": float(np.sqrt(np.mean((truth - mean) ** 2))),
    }


def test_shared_split_and_independent_mean_computation():
    rows = deepcopy(ROWS[:40])
    result = feedback.comparison(rows, SCOPE, candidate(rows))
    train, dev = api.development_split(rows)
    assert result == feedback.comparison(rows[::-1], SCOPE, candidate(rows))
    assert set(result["split"]["fit_groups"]).isdisjoint(result["split"]["score_groups"])
    assert result["attempted_controls"] == result["completed_controls"] == 6
    assert result["new_autonomous_models"] == 0
    controls = {x["method"]: x for x in result["controls"]}
    expected = np.sqrt(np.mean((np.array([r["target"] for r in dev]) - np.mean([r["target"] for r in train])) ** 2))
    assert controls["training_mean"]["metrics"]["rmse"] == pytest.approx(expected)
    assert all("predictions" not in x and "seconds" not in x for x in result["controls"])
    assert result["candidate_minus_best_rmse"] >= 0


def test_scoring_labels_do_not_influence_control_predictions_or_scaling():
    train, dev = api.development_split(deepcopy(ROWS[:40]))
    before = score_rows(train, dev, SCOPE, timings=False)
    altered = deepcopy(dev)
    for row in altered:
        row["target"] += 1000
    after = score_rows(train, altered, SCOPE, timings=False)
    assert [x["predictions"] for x in before] == [x["predictions"] for x in after]
    assert before[0]["metrics"] != after[0]["metrics"]
    with pytest.raises(ValueError, match="independent"):
        score_rows(train, train[:2], SCOPE)


def test_reference_anchor_excluded_metrics_and_failed_controls_remain_visible():
    scope = deepcopy(SCOPE)
    scope["unit"] = "ratio_to_within_replicate_reference"
    scope["covariates"].append({"name": "reference_temperature", "unit": "degC", "minimum": 20, "maximum": 90})
    rows = deepcopy(ROWS[:40])
    for row in rows:
        row["values"].append(30)
        row["target"] /= 1.2
    got = feedback.comparison(rows, scope, candidate(rows))
    assert all(x["non_reference_metrics"]["rows"] == 9 for x in got["controls"] if x["status"] == "completed")
    for row in rows:
        row["target"] = -1.0
    got = feedback.comparison(rows, scope, candidate(rows))
    assert len(got["controls"]) == 6
    assert got["completed_controls"] == 4
    assert {x["status"] for x in got["controls"] if x["method"].startswith("thermal_")} == {"failed"}


@pytest.mark.parametrize("bad", ["future", "group_baselines.v2", None, {}, False])
def test_unknown_feedback_policy_fails_closed(bad):
    with pytest.raises(ValueError):
        feedback.policy_for({feedback.FIELD: bad}, plugin_id=api.PLUGIN_ID)


def test_policy_requires_context_and_is_absent_from_reuse(tmp_path):
    with pytest.raises(ValueError, match="contextual"):
        feedback.policy_for({feedback.FIELD: feedback.POLICY}, plugin_id="generated_sequence.v1")
    with pytest.raises(ValueError, match="only during development"):
        make_bundle(
            tmp_path, "reuse", {}, SCOPE, {}, plugin_id=api.PLUGIN_ID, training_baseline_feedback=feedback.POLICY
        )
    protocol = {"model_plugin_id": api.PLUGIN_ID, feedback.FIELD: feedback.POLICY}
    feedback.verify_bundle({"phase": "reuse", "plugin_id": api.PLUGIN_ID}, protocol)
    with pytest.raises(ValueError, match="differs"):
        feedback.verify_bundle({"phase": "develop", "plugin_id": api.PLUGIN_ID}, protocol)


def prepare_goal(tmp_path, policy):
    task, data, _ = make_task(tmp_path / "assets")
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("dmx", "unisound-u2-flash", "https://www.dmxapi.cn/v1"),
        Budget(wall_seconds=10, model_calls=4, tool_calls=3, thinking=None),
    )
    atomic_json(tmp_path / "task.json", task)
    atomic_json(tmp_path / "profile.json", profile.public())
    budget = {"max_cycles": 4, "model_calls": 16, "tool_calls": 12, "wall_seconds": 300}
    allocation = {
        "id": "synthetic_feedback",
        "task": task,
        "profile": profile.public(),
        "generations": 1,
        "goal_budget": budget,
        feedback.FIELD: policy,
    }
    args = SimpleNamespace(
        profile=tmp_path / "profile.json",
        objective="Synthetic feedback test",
        task_definition=tmp_path / "task.json",
        output=tmp_path / "goal",
        case="synthetic_context",
        source_cache=None,
        training_baseline_feedback=policy,
        **budget,
    )
    return prepare(args), allocation, data


def test_goal_and_campaign_bind_the_intervention_and_reject_later_changes(tmp_path):
    store, allocation, _ = prepare_goal(tmp_path, feedback.POLICY)
    assert admit_allocations([allocation])[0][feedback.FIELD] == feedback.POLICY
    driver = GeneratedModelGoal(store, None)
    assert driver.protocol[feedback.FIELD] == store.read()["definition"][feedback.FIELD] == feedback.POLICY
    bind_goal(allocation, store.root)
    with pytest.raises(ValueError, match="allocation changed"):
        bind_goal({**allocation, feedback.FIELD: "none"}, store.root)
    protocol = deepcopy(driver.protocol)
    protocol[feedback.FIELD] = "none"
    atomic_json(store.root / "protocol.json", protocol)
    with pytest.raises(ValueError, match="binding changed"):
        GeneratedModelGoal(store, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("evolving", [False, True])
async def test_production_goal_exposes_only_frozen_training_feedback(tmp_path, monkeypatch, evolving):
    import run_science_goal

    store, _, data = prepare_goal(tmp_path, feedback.POLICY)
    if evolving:
        protocol = json.loads((store.root / "protocol.json").read_text())
        protocol["evolution"] = {"schema": 2, "generations": 2}
        atomic_json(store.root / "protocol.json", protocol)
    driver = (EvolvingModelGoal if evolving else GeneratedModelGoal)(store, None)

    class HostReached(Exception):
        pass

    async def inspect_host(profile, task, runtimes, *, credential):
        bundle = json.loads((task.capability / "task.json").read_text())
        assert bundle[feedback.FIELD] == feedback.POLICY
        assert bundle["inputs"] == {"training.json": digest(data["training"])}
        assert sorted(p.name for p in task.workspace.iterdir()) == ["training.json"]
        for name in ["context_controls.py", "context_feedback.py", "reference_thermal_baselines.py"]:
            assert (task.capability / name).is_file()
        raise HostReached

    monkeypatch.setattr(run_science_goal, "execute", inspect_host)
    with pytest.raises(HostReached):
        await driver.cycle(0, Profile.from_dict(store.read()["definition"]["profile"]), store.read(), "Synthetic")


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["none", feedback.POLICY])
async def test_real_mcp_feedback_submission_and_host_numeric_replay(tmp_path, policy):
    work = tmp_path / "workspace"
    work.mkdir()
    atomic_json(work / "training.json", ROWS[:40])
    versions, version, caps = make_bundle(
        tmp_path / "capability",
        "develop",
        {"training.json": ROWS[:40]},
        SCOPE,
        {"provider": "synthetic"},
        plugin_id=api.PLUGIN_ID,
        training_baseline_feedback=policy,
    )
    config = {
        "mcpServers": {
            "context": {
                "command": sys.executable,
                "args": ["-u", str(caps / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(work)},
            }
        }
    }
    async with Client(config) as client:

        async def call(name, args):
            result = await client.call_tool(name, args)
            validate_generated_result(name, result.data, workspace=work, capability=caps)
            return result.data

        info = await call("inspect_model_code_task", {})
        assert (feedback.FIELD in info) == (policy != "none")
        failed = await call("evaluate_model_code", {"source_code": "invalid syntax!", "rationale": "test failed code"})
        assert failed["status"] == "development_failed" and failed["attempts_remaining"] == 1
        trial = await call("evaluate_model_code", {"source_code": CODE, "rationale": "test valid code"})
        assert trial["attempts_remaining"] == 0
        assert ("baseline_comparison" in trial["training_only_feedback"]) == (policy != "none")
        submitted = await call("submit_model_code", {"source_sha256": trial["source_sha256"], "rationale": "tested"})
        assert submitted["training_rows"] == 40 and not submitted["registered"]
    versions.verify(version)
    if policy != "none":
        # Even a jointly altered result, artifact and ledger must fail numerical replay.
        changed = deepcopy(trial)
        changed["training_only_feedback"]["baseline_comparison"]["controls"][0]["metrics"]["rmse"] += 1
        path = work / "code_attempt_2.json"
        record = json.loads(path.read_text())
        record["training_feedback"] = changed["training_only_feedback"]
        atomic_json(path, record)
        ledger = json.loads((work / ".model_development.json").read_text())
        ledger["attempts"][1]["artifact_sha256"] = sha256_file(path)
        provenance_path = work / "source_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["training_feedback"] = changed["training_only_feedback"]
        atomic_json(provenance_path, provenance)
        ledger["submission"]["artifact_sha256"]["source_provenance.json"] = sha256_file(provenance_path)
        atomic_json(work / ".model_development.json", ledger)
        atomic_json(work / ("code_ledger_" + digest(ledger) + ".json"), ledger)
        with pytest.raises(ValueError, match="numerical replay"):
            validate_generated_result("evaluate_model_code", changed, workspace=work, capability=caps)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_standalone_feedback_binds_versioned_controls_and_rejects_modified_source(tmp_path, version):
    from test_context_baseline_refit import fixture

    scope, training, scoring = fixture("numeric")
    if version > 1:
        scope["interface_version"] = f"generated_context.v{version}"
    suffix = "" if version == 1 else f"_v{version}"
    module = importlib.import_module(f"agent.rsi.engine.context{suffix}_feedback")
    rows = training + scoring
    metrics = candidate(rows)
    expected = module.comparison(rows, scope, metrics)
    _, _, bundle = make_bundle(
        tmp_path / "capability",
        "develop",
        {"training.json": rows},
        scope,
        {"provider": "synthetic"},
        plugin_id=f"generated_context.v{version}",
        training_baseline_feedback=module.POLICY,
    )
    # Alter a disposable copy; leave the version store immutable.
    standalone = tmp_path / "standalone"
    shutil.copytree(bundle, standalone)
    script = r"""
import importlib, json, sys
from pathlib import Path
bundle, module_name = sys.argv[1:]
sys.path.insert(0, bundle)
module = importlib.import_module(module_name)
rows, scope, metrics = json.load(sys.stdin)
first = module.comparison(rows, scope, metrics)
control_module = module_name.replace("_feedback", "_controls")
core = Path(bundle) / (control_module + ".py")
assert Path(importlib.import_module(control_module).__file__).resolve() == core.resolve()
assert not any(name == "agent" or name.startswith("agent.") for name in sys.modules)
core.chmod(core.stat().st_mode | 0o200)
core.write_text(core.read_text() + "\n# synthetic source alteration\n")
try:
    module.verify_feedback(
        {"phase": "develop", "scope": scope, "plugin_id": scope.get("interface_version", "generated_context.v1"),
         module.FIELD: module.POLICY}, rows, {**metrics, "baseline_comparison": first},
    )
except ValueError as error:
    assert "numerical replay" in str(error)
else:
    raise AssertionError("Changed versioned control source accepted")
print(json.dumps(first, allow_nan=False))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(standalone), f"context{suffix}_feedback"],
        cwd=tmp_path,
        input=json.dumps([rows, scope, metrics]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    assert f"context{suffix}_controls.py" in expected["implementation"]


@pytest.mark.parametrize("version", [1, 2, 3])
def test_released_feedback_records_remain_verifiable(version):
    from test_context_baseline_refit import fixture

    scope, training, scoring = fixture("numeric")
    if version > 1:
        scope["interface_version"] = f"generated_context.v{version}"
    suffix = "" if version == 1 else f"_v{version}"
    module = importlib.import_module(f"agent.rsi.engine.context{suffix}_feedback")
    rows = training + scoring
    metrics = candidate(rows)
    recorded = module.comparison(rows, scope, metrics)
    # Numerical replay alone misses this regression: the stored source identity
    # is also part of the equality contract used by publication and inference.
    bindings = json.loads((Path(__file__).parent / "fixtures/context_feedback_abi.json").read_text())
    recorded["implementation"] = bindings["policies"][str(version)]
    persisted = {**metrics, "baseline_comparison": recorded}
    original = deepcopy(persisted)
    task = {
        "phase": "develop",
        "scope": scope,
        "plugin_id": f"generated_context.v{version}",
        module.FIELD: module.POLICY,
    }
    module.verify_feedback(task, rows, json.loads(json.dumps(persisted)))
    assert persisted == original
    changed = deepcopy(persisted)
    changed["baseline_comparison"]["controls"][0]["metrics"]["rmse"] += 1
    with pytest.raises(ValueError, match="numerical replay"):
        module.verify_feedback(task, rows, changed)
