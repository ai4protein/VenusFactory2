"""Synthetic lifecycle/admission tests; scientific results come from separate runs."""

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

from agent.rsi import clean_training_plugin as c
from agent.rsi.engine.case_evidence import canonical_claims, render_evidence_report
from agent.rsi.campaigns.case_jobs import verify_job_report
from agent.rsi.campaigns.case_plugins import invocation, list_case_plugins, maximum_case_job_seconds, resource_for
from agent.rsi.campaigns.paper_campaign import campaign_cases, paper_case
from agent.rsi.engine.reproduction_goal_contract import allocation


def report():
    values = {}
    for a in c.evaluation.SCHEDULES:
        for name, n in c.evaluation.COHORTS.items():
            values[a + "." + name] = {
                "status": "passed",
                "rows": n,
                "label_mismatches": [],
                "rounded_distances_match": True,
                "metrics_match": True,
                "metadata_match": True,
                "full_union_metrics": {"micro_f1": 0.2, "weighted_f1": 0.3, "exact_match": 0.1},
            }
    return {
        "status": "completed",
        "stage": "audited",
        "plugin": c.PLUGIN,
        "parameters": {},
        "execution_origin": c.ORIGIN,
        "allocation": copy.deepcopy(c.ALLOCATION),
        "external_model_calls": 0,
        **c.LIMITS,
        "audit_evidence": {
            "status": "passed",
            "limits": c.evaluation.LIMITS,
            "training": {k: {"status": "passed", "epochs": v["epochs"]} for k, v in c.evaluation.SCHEDULES.items()},
            "allocations": values,
        },
    }


def test_scope_is_visible_to_both_existing_goal_catalogs(tmp_path):
    paper = paper_case("clean_full_training")
    assert paper["paper_id"] == "clean"
    assert paper["allocation"] == allocation("clean_full_training")
    assert paper["full_paper_reproduced"] is False
    assert campaign_cases("enzyme_annotation_training") == [paper]
    assert len(campaign_cases("eight_papers")) == 8
    assert campaign_cases("enzyme_annotation")[0]["case"] == "clean_all_cohorts"
    plugin = next(p for p in list_case_plugins() if p["id"] == c.PLUGIN)
    assert plugin["parameters"]["properties"] == {}
    assert plugin["parameters"]["additionalProperties"] is False
    assert resource_for(c.PLUGIN, {}) == "cpu"
    assert maximum_case_job_seconds(c.PLUGIN) == c.JOB_TIMEOUT == 259200
    args = invocation(
        c.PLUGIN, {}, root=tmp_path, output=tmp_path / "research_runs/rsi/new", controller_python="python"
    )
    assert args[0].endswith("envs/clean/bin/python")
    assert args[2].endswith("run_clean_training_plugin.py") and "--verify" not in args


@pytest.mark.parametrize(
    "parameters",
    [
        {"epochs": 1},
        {"algorithm": "triplet"},
        {"resume": True},
        {"inputs": "/tmp/old"},
        {"operation": "verify_existing"},
    ],
)
def test_no_scope_reduction_or_prior_output_shortcut(parameters, tmp_path):
    with pytest.raises(ValueError, match="22525"):
        invocation(
            c.PLUGIN, parameters, root=tmp_path, output=tmp_path / "research_runs/rsi/new", controller_python="python"
        )


def test_fixed_stage_arguments_do_not_select_old_evidence(tmp_path, monkeypatch):
    output = tmp_path / "research_runs/rsi/run"
    commands = c.commands(tmp_path, output)
    assert list(commands) == list(c.STAGES)
    assert "--resume" not in commands["inputs"]
    assert "--epochs" not in commands["triplet"]
    for name in ("triplet", "supconH"):
        assert commands[name][commands[name].index("--inputs") + 1] == str(output / "inputs")
        assert commands[name][commands[name].index("--algorithm") + 1] == name
    monkeypatch.setenv("FAKE_SECRET_FOR_TEST", "must_not_cross")
    assert "FAKE_SECRET_FOR_TEST" not in c.environment(tmp_path, output)
    assert c.environment(tmp_path, output)["PYTHONHASHSEED"] == "1234"
    with pytest.raises(ValueError):
        c.output_path(tmp_path, tmp_path / "outside")


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "running"),
        ("stage", "trained"),
        ("external_model_calls", True),
        ("execution_origin", "verified_existing"),
        ("parameters", {"resume": True}),
        *[(key, True) for key in c.LIMITS],
    ],
)
def test_cannot_promote_incomplete_or_invented_scientific_claim(key, value):
    value_report = report()
    value_report[key] = value
    with pytest.raises(ValueError):
        canonical_claims(c.PLUGIN, value_report)


def test_every_training_schedule_and_cohort_is_required():
    for mutation in [
        lambda a: a["training"].pop("supconH"),
        lambda a: a["training"]["triplet"].update(epochs=1999),
        lambda a: a["allocations"].pop("supconH.price"),
        lambda a: a["allocations"]["triplet.new"].update(rows=391),
        lambda a: a["allocations"]["triplet.new"].update(label_mismatches=["bad"]),
        lambda a: a["allocations"]["triplet.new"]["full_union_metrics"].update(micro_f1=float("nan")),
    ]:
        value = report()
        mutation(value["audit_evidence"])
        with pytest.raises(ValueError):
            canonical_claims(c.PLUGIN, value)
    claims = canonical_claims(c.PLUGIN, report())
    assert claims["prediction_records"] == 1156 and claims["new_embedding_records"] == 22525
    assert not claims["new_model_developed"] and not claims["recursive_improvement_established"]
    assert "2,000 triplet" in render_evidence_report("fixture", claims)


def test_host_cannot_skip_the_pinned_numerical_gate(tmp_path, monkeypatch):
    calls = []

    def reject(*args):
        calls.append(args)
        raise ValueError("isolated verification failed")

    monkeypatch.setattr(c, "verify_artifacts", reject)
    with pytest.raises(ValueError, match="isolated"):
        verify_job_report(tmp_path, {"plugin_id": c.PLUGIN, "parameters": {}, "output": str(tmp_path)}, report())
    assert len(calls) == 1


def journal_fixture(tmp_path, monkeypatch):
    (tmp_path / "inputs").mkdir()
    c.inputs.save(tmp_path / "inputs/report.json", {"resumes": 0, "attempts_started": 2, "completed_records": 2})
    c.inputs.save(tmp_path / "inputs/plan.json", {"records": [{"key": "a"}, {"key": "b"}]})
    monkeypatch.setattr(c.inputs, "TOTAL", 2)
    # Canonical scientific input admission is tested in test_clean_training_inputs.
    monkeypatch.setattr(c.inputs, "validate_plan", lambda _: None)
    events = [
        {"event": event, "key": key, "at": 11 + i}
        for i, (event, key) in enumerate(
            [
                ("started", "a"),
                ("completed", "a"),
                ("started", "b"),
                ("completed", "b"),
            ]
        )
    ]
    path = tmp_path / "inputs/attempts.jsonl"
    path.write_text("".join(json.dumps(v) + "\n" for v in events))
    stages = [{"name": "inputs", "started_unix": 10, "finished_unix": 20}]
    for name in ("triplet", "supconH"):
        p = tmp_path / "training" / name
        p.mkdir(parents=True)
        (p / "epochs.jsonl").write_text('{"epoch": 1, "loss": 1, "at": 25}\n')
        stages.append({"name": name, "started_unix": 21, "finished_unix": 30})
    return path, events, stages


@pytest.mark.parametrize("attack", ["truncated", "reordered", "copied_old", "extra", "cached"])
def test_fresh_execution_journal_rejects_reuse_and_incompleteness(tmp_path, monkeypatch, attack):
    path, events, stages = journal_fixture(tmp_path, monkeypatch)
    c.fresh_journals(tmp_path, stages)
    if attack == "truncated":
        events.pop()
    elif attack == "reordered":
        events.reverse()
    elif attack == "copied_old":
        events[0]["at"] = 1
    elif attack == "extra":
        events.append(events[-1])
    else:
        c.inputs.save(tmp_path / "inputs/report.json", {"resumes": 1, "attempts_started": 2, "completed_records": 2})
    path.write_text("".join(json.dumps(v) + "\n" for v in events))
    with pytest.raises(ValueError):
        c.fresh_journals(tmp_path, stages)


def test_progress_never_grants_acceptance_and_rejects_invalid_counts(tmp_path):
    (tmp_path / "inputs").mkdir()
    c.inputs.save(tmp_path / "report.json", {"stage": "inputs"})
    c.inputs.save(tmp_path / "inputs/report.json", {"completed_records": 100})
    assert c.public_progress(tmp_path) == {
        "available": True,
        "stage": "inputs",
        "unverified_progress": True,
        "completed_input_records": 100,
        "total_input_records": 22525,
    }
    c.inputs.save(tmp_path / "inputs/report.json", {"completed_records": 999999})
    assert "completed_input_records" not in c.public_progress(tmp_path)
    assert "status" not in c.public_progress(tmp_path)


@pytest.mark.asyncio
async def test_verification_subprocess_is_responsive_and_rejects_forged_response(tmp_path, monkeypatch):
    expected = {"status": "passed", "claims": {"fixture": True}}
    argv = [sys.executable, "-c", "import time,json;time.sleep(0.2);print(" + repr(json.dumps(expected)) + ")"]
    monkeypatch.setattr(c, "verification_request", lambda *args: (argv, expected))
    task = asyncio.create_task(c.verify_artifacts_async({}, tmp_path, tmp_path))
    ticks = 0
    while not task.done():
        await asyncio.sleep(0.02)
        ticks += 1
    assert await task == {"fixture": True} and ticks >= 3
    expected["claims"] = {"fixture": False}
    with pytest.raises(ValueError, match="verification failed"):
        await c.verify_artifacts_async({}, tmp_path, tmp_path)


@pytest.mark.asyncio
async def test_cancelled_verification_leaves_no_running_process(tmp_path, monkeypatch):
    pid_path = tmp_path / "pid"
    code = (
        "import os,pathlib,time;pathlib.Path(" + repr(str(pid_path)) + ").write_text(str(os.getpid()));time.sleep(60)"
    )
    monkeypatch.setattr(c, "verification_request", lambda *args: ([sys.executable, "-c", code], {}))
    task = asyncio.create_task(c.verify_artifacts_async({}, tmp_path, tmp_path))
    async with asyncio.timeout(5):
        while not pid_path.exists():
            await asyncio.sleep(0.02)
    pid = int(pid_path.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not Path("/proc", str(pid)).exists()


def test_failed_stage_is_retained_and_output_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "research_runs/rsi/run"
    monkeypatch.setattr(c, "SOURCES", ())
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    frozen = {"commands": {"inputs": command}}
    monkeypatch.setattr(c, "protocol", lambda *args: frozen)
    with pytest.raises(ValueError, match="stage failed"):
        c.run(tmp_path, output)
    result = c.read(output / "report.json")
    assert result["status"] == "failed" and len(result["stages"]) == 1
    assert result["stages"][0]["exit_code"] == 7
    previous = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        c.run(tmp_path, output)
    assert (output / "report.json").read_bytes() == previous


def bound_fixture(tmp_path, monkeypatch):
    output = tmp_path / "research_runs/rsi/run"
    output.mkdir(parents=True)
    for name in c.SOURCES:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# synthetic source fixture\n")
    for name in c.artifact_names():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    frozen = c.protocol(tmp_path, output)
    c.inputs.save(output / "protocol.json", frozen)
    value = report()
    value["protocol_sha256"] = c.inputs.digest(frozen)
    value["stages"] = [
        {
            "name": name,
            "status": "completed",
            "exit_code": 0,
            "argv": frozen["commands"][name],
            "timeout_seconds": c.TIMEOUTS[name],
            "started_unix": i + 1,
            "finished_unix": i + 2,
        }
        for i, name in enumerate(c.STAGES)
    ]
    for name in c.SOURCES:
        (output / "implementation" / name).write_bytes((tmp_path / name).read_bytes())
    c.inputs.save(output / "audit.json", value["audit_evidence"])
    value["artifact_sha256"] = {name: c.inputs.sha(output / name) for name in c.artifact_names()}
    c.inputs.save(output / "report.json", value)
    # Journal behavior is exercised separately with actual file streams.
    monkeypatch.setattr(c, "fresh_journals", lambda *args: None)
    assert c.validate_artifacts(value, output, tmp_path)["prediction_records"] == 1156
    return output, value


@pytest.mark.parametrize(
    "attack",
    ["omission", "archive_substitution", "live_source_drift", "wrong_arguments", "stage_failure", "audit_substitution"],
)
def test_artifact_or_pipeline_substitution_is_rejected_before_numerical_replay(tmp_path, monkeypatch, attack):
    output, value = bound_fixture(tmp_path, monkeypatch)
    if attack == "omission":
        value["artifact_sha256"].pop("audit.json")
    elif attack == "archive_substitution":
        name = "implementation/" + c.SOURCES[0]
        (output / name).write_text("different source")
        value["artifact_sha256"][name] = c.inputs.sha(output / name)
    elif attack == "live_source_drift":
        (tmp_path / c.SOURCES[0]).write_text("new implementation")
    elif attack == "wrong_arguments":
        value["stages"][1]["argv"] = ["python", "unregistered.py"]
    elif attack == "stage_failure":
        value["stages"][2]["status"] = "failed"
    else:
        (output / "audit.json").write_text("{}")
        value["artifact_sha256"]["audit.json"] = c.inputs.sha(output / "audit.json")
    with pytest.raises(ValueError):
        c.validate_artifacts(value, output, tmp_path)


def test_isolated_verifier_receives_the_actual_persisted_report(tmp_path, monkeypatch):
    output, value = bound_fixture(tmp_path, monkeypatch)
    argv, _ = c.verification_request(value, output, tmp_path)
    assert argv[0].endswith("/envs/clean/bin/python") and argv[-1] == "--verify"
    c.inputs.save(output / "report.json", {"status": "completed"})
    with pytest.raises(ValueError, match="persisted report"):
        c.verification_request(value, output, tmp_path)


def test_full_numeric_replay_cannot_be_replaced_by_receipt_acceptance(tmp_path, monkeypatch):
    output, value = bound_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(c.importlib.metadata, "version", lambda key: c.inputs.ENVIRONMENT[key])
    calls = []

    def incorrect(*args):
        calls.append(args)
        return {"status": "failed"}

    monkeypatch.setattr(c.evaluation, "verify_output", incorrect)
    with pytest.raises(ValueError, match="numerical replay"):
        c.verify_in_process(value, output, tmp_path)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_broker_wrapper_receives_seeded_credential_free_environment(tmp_path, monkeypatch):
    from agent.rsi import case_jobs
    from agent.rsi import clean_training_plugin as plugin

    root = tmp_path / "repo"
    root.mkdir()
    script = root / "environment_fixture.py"
    script.write_text(
        "import json,os,sys,pathlib\n"
        "p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True)\n"
        "(p/'report.json').write_text(json.dumps({'status':'completed','environment':dict(os.environ)}))\n"
    )
    monkeypatch.setenv("DMX_API_KEY", "synthetic-secret-never-sent")
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    monkeypatch.setattr(
        case_jobs,
        "invocation",
        lambda plugin_id, parameters, **kw: [sys.executable, "-u", str(script), str(kw["output"])],
    )
    checked = []

    async def inspect_environment(report, output, workspace):
        env = report["environment"]
        assert env["PYTHONHASHSEED"] == "1234"
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4"
        assert env["OPENBLAS_NUM_THREADS"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["HOME"] == str(output / "job_home")
        assert "DMX_API_KEY" not in env
        checked.append(True)
        raise ValueError("Synthetic environment-only execution cannot pass scientific verification")

    monkeypatch.setattr(plugin, "verify_artifacts_async", inspect_environment)
    broker = case_jobs.CaseJobBroker(
        root=root,
        workspace=root / "workspace",
        output=root / "research_runs/rsi/jobs",
        allowed_plugins={plugin.PLUGIN},
        timeout_s=10,
    )
    async with broker:
        job = broker.submit(plugin.PLUGIN, {}, "environment-fixture")
        await broker.wait_jobs()
        assert checked == [True]
        status = broker.status(job["job_id"])
        assert status["status"] == "failed"
        assert "environment-only" in status["error"]
