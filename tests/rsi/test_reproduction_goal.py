"""Goal/registry evidence integration with explicitly synthetic jobs, no LLM or GPU."""

import importlib.util
import json
from pathlib import Path

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi.engine.reproduction_goal_contract import CRITERIA, allocation
from agent.science_goal import GoalSpec, GoalStore, run_goal


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/rsi"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("goal_reproduction_fixture", scripts / "run_reproduction_goal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "verify_job_report", lambda *args: {"synthetic_fixture": True})
    monkeypatch.setattr(mod, "canonical_claims", lambda *args: {"synthetic_fixture": True})
    return mod


def setup(module, tmp_path, case="phl7_wt_hrex_cuda"):
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=2, model_calls=12, tool_calls=12),
    )
    store = GoalStore.create(
        tmp_path / "goal",
        GoalSpec("Reproduce, register and verify reuse", CRITERIA, model_calls=48, tool_calls=48),
        profile,
    )
    module.write_json(
        store.root / "protocol.json",
        {
            "case": case,
            "allocation": allocation(case),
            "implementation": {},
            "cycle_wall_seconds": 4,
        },
    )
    return store, module.ReproductionGoal(store, "unused-fixture")


def run_fixture(module, output, recipe=None, session_prefix=None, declaration=None):
    output.mkdir(parents=True)
    declaration = declaration or allocation("phl7_wt_hrex_cuda")
    jid = "case_" + digest(str(output))[:20]
    domain = output / "controller" / jid / "report.json"
    domain.parent.mkdir(parents=True)
    module.write_json(domain, {"status": "completed"})
    protocol = {
        "plugin": declaration["plugin"],
        "parameters": declaration["parameters"],
        "registered_recipe_sha256": digest(recipe) if recipe else None,
    }
    module.write_json(output / "protocol.json", protocol)
    job = {
        "job_id": jid,
        "plugin_id": declaration["plugin"],
        "parameters": declaration["parameters"],
        "status": "completed",
        "report_sha256": file_hash(domain),
    }
    (output / "workspace").mkdir()
    module.write_json(
        output / "workspace/case_submission.json",
        {
            "schema": 1,
            "job_id": jid,
            "plugin_id": job["plugin_id"],
            "parameters": job["parameters"],
            "submission_only": True,
        },
    )
    (output / "review_workspace").mkdir()
    module.write_json(output / "review_workspace/review.json", {"claims": {"synthetic_fixture": True}})
    report = {
        "status": "completed",
        "protocol_sha256": digest(protocol),
        "end_to_end_verified": True,
        "agent_requested": True,
        "new_computation": True,
        "implementation_unchanged": True,
        "submission_receipt_verified": True,
        "jobs": [job],
    }
    for phase in ["launch", "review"]:
        path = output / phase
        path.mkdir()
        tools = (
            ["discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"]
            if phase == "launch"
            else ["inspect_case_evidence", "submit_evidence_review"]
        )
        if recipe is not None and phase == "launch":
            tools.append("inspect_registered_reproduction")
        events = []
        for i, name in enumerate(tools):
            events.extend(
                [
                    {"kind": "tool_call_start", "tool_call_id": str(i), "tool_name": name},
                    {"kind": "tool_result", "tool_call_id": str(i), "is_error": False},
                ]
            )
        (path / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
        (path / "profile.lock.json").write_text("{}")
        (path / "model_requests.jsonl").write_text("{}\n")
        run = {
            "status": "completed",
            "session_id": (session_prefix or str(output)) + phase,
            "audit": {"passed": True, "requests": {"model_calls": 1}, "events": {"tool_calls": len(tools)}},
        }
        module.write_json(path / "run.json", run)
        report[phase] = {"run": run}
    report["review"]["source_report_sha256"] = file_hash(domain)
    module.write_json(output / "report.json", report)
    return report


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["phl7_wt_hrex_cuda", "clean_full_training"])
async def test_register_then_retrieve_recompute_and_verify_new_sessions(module, tmp_path, monkeypatch, case):
    store, goal = setup(module, tmp_path, case)
    seen = []

    async def reproduce(args, **kwargs):
        seen.append(kwargs)
        return run_fixture(module, args.output, kwargs["registered_recipe"], declaration=allocation(case))

    monkeypatch.setattr(module, "reproduce", reproduce)
    state = await run_goal(store, goal.cycle, goal.evaluate, cycle_wall_seconds=4)
    assert state["status"] == "completed" and len(state["cycles"]) == 2
    assert seen[0]["registered_recipe"] is None
    assert seen[1]["registered_recipe"]["novel_model_claim"] is False
    assert "registered_recipe" in seen[1]["goal_context"]
    assert state["usage"]["model_calls"] == 4
    assert all(c["passed"] for c in state["assessment"]["criteria"].values())
    # Changing a previously verified source after registration is detected.
    source = seen[1]["registered_recipe"]["source_run"]
    (store.root / source / "launch/events.jsonl").write_text("")
    with pytest.raises(ValueError, match="tool chain"):
        goal.evaluate(state)


@pytest.mark.asyncio
async def test_failed_job_is_retained_and_does_not_register(module, tmp_path, monkeypatch):
    store, goal = setup(module, tmp_path)

    async def reproduce(args, **kwargs):
        report = run_fixture(module, args.output)
        report.update(status="launch_or_job_failed", end_to_end_verified=False)
        module.write_json(args.output / "report.json", report)
        return report

    monkeypatch.setattr(module, "reproduce", reproduce)
    state = store.read()
    result = await goal.cycle(0, Profile.from_dict(state["definition"]["profile"]), state, "fixture objective")
    assert result["status"] == "failed" and not result["progress"]
    assert not list((store.root / "recipes/versions").iterdir())
    assert (store.root / result["run_record"]).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "allocation",
        "domain_hash",
        "receipt",
        "new_computation",
        "review_claim",
        "session",
        "symlink",
        "missing_recipe_read",
    ],
)
def test_forged_or_incomplete_reproduction_never_passes(module, tmp_path, mutation):
    output = tmp_path / "run"
    recipe = (
        {"plugin": "phl7.hrex_validation.v1", "parameters": {"backend": "cuda"}}
        if mutation == "missing_recipe_read"
        else None
    )
    report = run_fixture(module, output, recipe)
    if mutation == "allocation":
        report["jobs"][0]["parameters"] = {"backend": "cpu"}
    elif mutation == "domain_hash":
        report["jobs"][0]["report_sha256"] = "0" * 64
    elif mutation == "receipt":
        (output / "workspace/case_submission.json").write_text("{}")
    elif mutation == "new_computation":
        report["new_computation"] = False
    elif mutation == "review_claim":
        (output / "review_workspace/review.json").write_text('{"claims":{"synthetic_fixture":false}}')
    elif mutation == "session":
        report["review"]["run"]["session_id"] = report["launch"]["run"]["session_id"]
        module.write_json(output / "review/run.json", report["review"]["run"])
    elif mutation == "symlink":
        target = output / "review_workspace/review.json"
        target.rename(output / "elsewhere.json")
        target.symlink_to(output / "elsewhere.json")
    elif mutation == "missing_recipe_read":
        path = output / "launch/events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(e) + "\n" for e in events if e["tool_call_id"] != "3"))
    module.write_json(output / "report.json", report)
    with pytest.raises(ValueError):
        module.verify_run(output, allocation("phl7_wt_hrex_cuda"), registered_recipe=recipe)


def test_allocations_do_not_share_mutable_parameters():
    first = allocation("vhh_ha_64x4")
    first["parameters"]["backbones"] = 1
    assert allocation("vhh_ha_64x4")["parameters"]["backbones"] == 64
    assert allocation("vhh_ha_64x4")["job_timeout"] > 6 * 3600


@pytest.mark.asyncio
async def test_snapshot_interruption_retains_orphan_and_allows_verified_retry(module, tmp_path, monkeypatch):
    store, goal = setup(module, tmp_path)
    original = goal.versions.snapshot
    attempts = []

    def interrupted_snapshot(*args, **kwargs):
        manifest = original(*args, **kwargs)
        attempts.append(manifest["version_id"])
        if len(attempts) == 1:
            raise RuntimeError("simulated worker interruption before goal checkpoint")
        return manifest

    async def reproduce(args, **kwargs):
        return run_fixture(module, args.output, kwargs["registered_recipe"])

    monkeypatch.setattr(goal.versions, "snapshot", interrupted_snapshot)
    monkeypatch.setattr(module, "reproduce", reproduce)
    state = await run_goal(store, goal.cycle, goal.evaluate, cycle_wall_seconds=4)
    assert state["status"] == "completed" and len(state["cycles"]) == 3
    assert state["cycles"][0]["status"] == "failed"
    assert state["progress"]["recipe_version"] == attempts[1]
    assert (goal.versions.root / "versions" / attempts[0]).exists()
    assert state["usage"]["model_calls"] == 12 + 4
