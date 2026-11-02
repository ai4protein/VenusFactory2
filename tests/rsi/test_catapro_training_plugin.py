import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent.rsi import catapro_training_evidence as evidence
from agent.rsi.case_evidence import canonical_claims, render_evidence_report
from agent.rsi.case_plugins import invocation, resource_for
from agent.rsi.catapro_case import file_sha
from agent.rsi.catapro_training import PARAMETERS, STAGES, gpu_lease, stop_training_process
from agent.rsi.reproduction_goal_contract import allocation


def call(root, params):
    return invocation(
        "catapro.training.v1",
        params,
        root=root,
        output=root / "research_runs/rsi/new",
        controller_python=Path(sys.executable),
    )


def test_training_recipe_does_not_accept_preflight_or_partial_overrides(tmp_path):
    command = call(tmp_path, {"operation": "reproduce"})
    assert command[0].endswith("20261017_catapro_runtime_setup_v1/venv/bin/python")
    assert command[-2:] == ["--operation", "reproduce"]
    assert resource_for("catapro.training.v1", {"operation": "reproduce"}) == "gpu"
    params = {"operation": "verify_existing", "campaign": "catapro_training_20261017"}
    assert call(tmp_path, params)[-4:] == ["--operation", "verify_existing", "--campaign", "catapro_training_20261017"]
    assert resource_for("catapro.training.v1", params) == "cpu"
    for wrong in [
        {},
        {"operation": "reproduce", "epochs": 1},
        {"operation": "reproduce", "stages": ["kcat"]},
        {"operation": "verify_existing", "campaign": "../other"},
        {"operation": "reproduce", "campaign": "catapro_training_20261017"},
    ]:
        with pytest.raises(ValueError):
            call(tmp_path, wrong)
    assert allocation("catapro_full_training")["parameters"] == {"operation": "reproduce"}


def facts_report():
    stages = {}
    for stage, (cohort, _, _) in STAGES.items():
        count = evidence.COUNTS[cohort]
        base, extra = divmod(count, 10)
        stages[stage] = {
            "status": "passed",
            "epochs_cap": 150,
            "rows": count,
            "folds": 10,
            "checkpoints": [{"fold": i, "validation_rows": base + (i < extra), "sha256": "0" * 64} for i in range(10)],
            "metrics": {"PCC": -0.2, "SCC": -0.1, "R2": -0.3, "RMSE": 1.5},
        }
    return {
        "status": "completed",
        "execution_origin": "verified_existing",
        "source_campaign": "research_runs/rsi/fixture",
        "audit_evidence": {
            "source_commit": evidence.COMMIT,
            "epochs_cap": 150,
            "training_fits": 50,
            "cohort_rows": evidence.COUNTS.copy(),
            "all_rows_independently_replayed": True,
            "stages": stages,
            "reconstruction_decisions": ["Explicit fixture decision"],
            "alpha_decision": {"value": 0.4, "basis": "Released inference default, differing from training default"},
        },
    }


def test_training_claims_preserve_origin_negative_scores_and_reconstruction_scope():
    facts = canonical_claims("catapro.training.v1", facts_report())
    assert facts["training_reproduced"] is True
    assert facts["metrics"]["kcat"]["R2"] == -0.3
    assert facts["full_paper_reproduced"] is False
    assert facts["new_model_or_self_evolution_established"] is False
    assert facts["exact_author_stochastic_equivalence"] is False
    assert facts["execution_origin"] == "verified_existing"
    rendered = render_evidence_report("fixture", facts)
    assert "fifty" not in rendered.lower() or "training" in rendered
    assert "reconstructed earlier" in rendered
    assert "grid search" in rendered


@pytest.mark.parametrize(
    "problem",
    [
        "short_epochs",
        "missing_stage",
        "missing_fold",
        "missing_rows",
        "nan",
        "alpha",
        "decisions",
        "positive_r2",
        "origin",
    ],
)
def test_incomplete_training_cannot_support_registration(problem):
    report = facts_report()
    audit = report["audit_evidence"]
    if problem == "short_epochs":
        audit["epochs_cap"] = 1
    elif problem == "missing_stage":
        del audit["stages"]["activity_km"]
    elif problem == "missing_fold":
        audit["stages"]["km"]["checkpoints"].pop()
    elif problem == "missing_rows":
        audit["stages"]["activity"]["rows"] -= 1
    elif problem == "nan":
        audit["stages"]["kcat"]["metrics"]["RMSE"] = float("nan")
    elif problem == "alpha":
        audit["alpha_decision"]["value"] = 0.5
    elif problem == "decisions":
        audit["reconstruction_decisions"] = []
    elif problem == "positive_r2":
        audit["stages"]["km"]["metrics"]["R2"] = 1.5
    else:
        report["execution_origin"] = "new_computation_from_old_receipt"
    with pytest.raises(ValueError):
        canonical_claims("catapro.training.v1", report)


def terminal_fixture(root):
    protocol = {
        "source_commit": evidence.COMMIT,
        "epochs": 150,
        "training_fits": 50,
        "stages": list(STAGES),
        "parameters": PARAMETERS,
        "alpha": {"value": 0.4},
        "tables": dict.fromkeys(STAGES, {}),
        "author_source": {"author_source/" + n: "0" * 64 for n in evidence.AUTHOR_FILES},
        "implementation": {"repository/fixture.py": "0" * 64},
        "reconstruction_decisions": ["Explicit fixture"],
    }
    (root / "protocol.json").write_text(json.dumps(protocol))
    report = {
        "status": "completed",
        "protocol_sha256": file_sha(root / "protocol.json"),
        "stages": [{"stage": s, "status": "completed", "returncode": 0} for s in STAGES],
    }
    (root / "report.json").write_text(json.dumps(report))
    return protocol, report


@pytest.mark.parametrize(
    "problem",
    [
        "running",
        "preflight",
        "missing_stage",
        "failed_stage",
        "changed_protocol",
        "changed_source",
        "changed_parameter",
    ],
)
def test_terminal_admission_rejects_invalid_campaigns_before_loading_models(tmp_path, problem):
    protocol, report = terminal_fixture(tmp_path)
    if problem == "running":
        report["status"] = "running"
    elif problem == "preflight":
        protocol["epochs"] = 1
    elif problem == "missing_stage":
        report["stages"].pop()
    elif problem == "failed_stage":
        report["stages"][0]["returncode"] = 1
    elif problem == "changed_protocol":
        report["protocol_sha256"] = "different"
    elif problem == "changed_source":
        protocol["author_source"].pop(next(iter(protocol["author_source"])))
    else:
        protocol["parameters"] = copy.deepcopy(PARAMETERS)
        protocol["parameters"]["kcat"]["lr"] = 0.1
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    if problem != "changed_protocol":
        report["protocol_sha256"] = file_sha(tmp_path / "protocol.json")
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        evidence.validate_terminal_protocol(tmp_path)


def test_nested_evidence_cannot_escape_root_or_accept_changes(tmp_path):
    source = tmp_path / "root"
    source.mkdir()
    p = source / "weights"
    p.write_bytes(b"initial")
    bindings = {"weights": file_sha(p)}
    evidence.verify_bound_files(source, bindings)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        evidence.verify_bound_files(source, bindings)
    outside = tmp_path / "outside"
    outside.write_bytes(b"initial")
    with pytest.raises(ValueError, match="escapes"):
        evidence.verify_bound_files(source, {"../outside": file_sha(outside)})
    p.unlink()
    p.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        evidence.verify_bound_files(source, {"weights": file_sha(outside)})


def test_broker_lease_stays_locked_after_training_returns(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    with path.open("a") as parent:
        fcntl.flock(parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("VENUS_RSI_CASE_LOCK_FD", str(parent.fileno()))
        with gpu_lease(path) as child:
            assert child.fileno() != parent.fileno()
        with path.open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wrong = tmp_path / "wrong"
        wrong.touch()
        with pytest.raises(ValueError, match="different"):
            with gpu_lease(wrong):
                pass


def live(pid):
    p = Path("/proc") / str(pid) / "stat"
    try:
        return p.read_text().rsplit(")", 1)[1].split()[0] not in ["Z", "X"]
    except (FileNotFoundError, ProcessLookupError):
        return False


def test_training_cancellation_reaps_stubborn_parent_and_descendant(tmp_path):
    marker = tmp_path / "pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"p=Path({str(marker)!r}); q=p.with_suffix('.ready'); "
        "q.write_text(str(os.getpid())); q.replace(p); time.sleep(60)"
    )
    parent_code = (
        'import subprocess,signal,time,sys; signal.signal(signal.SIGTERM,signal.SIG_IGN); subprocess.Popen([sys.executable,"-c",'
        + repr(child_code)
        + "]); time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        child = int(marker.read_text())
        started = time.monotonic()
        stop_training_process(process, grace=0.2)
        assert time.monotonic() - started < 2
        deadline = time.monotonic() + 2
        while live(child) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not live(process.pid) and not live(child)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def test_wrapper_runs_controller_in_its_own_process_for_broker_cleanup(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    import run_catapro_training_plugin as plugin

    calls = []

    def fake_entry(path):
        return {
            "prepare": lambda args: calls.append(("prepare", os.getpid(), args.epochs)),
            "run": lambda campaign: calls.append(("run", os.getpid(), campaign)),
        }

    monkeypatch.setattr(plugin.runpy, "run_path", fake_entry)
    plugin.reproduce(tmp_path / "campaign", tmp_path)
    assert calls == [("prepare", os.getpid(), 150), ("run", os.getpid(), tmp_path / "campaign")]


def test_broker_cancels_real_training_controller_and_worker_group(tmp_path, monkeypatch):
    import asyncio

    from agent.rsi import case_jobs

    repo = Path(__file__).resolve().parents[2]
    marker = tmp_path / "descendant_pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"p=Path({str(marker)!r}); q=p.with_suffix('.ready'); "
        "q.write_text(str(os.getpid())); q.replace(p); time.sleep(60)"
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        'import subprocess,signal,time,sys\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nsubprocess.Popen([sys.executable,"-c",'
        + repr(child_code)
        + "])\ntime.sleep(60)\n"
    )
    wrapper = tmp_path / "controller.py"
    wrapper.write_text(
        "import runpy,sys,json\nfrom pathlib import Path\nsys.path.insert(0,"
        + repr(str(repo / "src"))
        + ")\n"
        + "entry=runpy.run_path("
        + repr(str(repo / "scripts/rsi/run_catapro_training.py"))
        + ")\n"
        + 'run=entry["run"]\nrun.__globals__["__file__"]='
        + repr(str(worker))
        + "\n"
        + 'out=Path(sys.argv[-1]);out.mkdir()\n(out/"stages").mkdir();(out/"home").mkdir()\n'
        + 'protocol={"author_source":{},"implementation":{},"gpu_lock":'
        + repr(str(tmp_path / "research_runs/rsi/.case_gpu.lock"))
        + ',"stages":["kcat"],"stage_timeout_seconds":60}\n'
        + '(out/"protocol.json").write_text(json.dumps(protocol))\n'
        + '(out/"report.json").write_text(json.dumps({"status":"prepared","protocol_sha256":entry["file_sha"](out/"protocol.json"),"stages":[]}))\n'
        + "run(out)\n"
    )

    def invocation(plugin_id, parameters, *, root, output, controller_python):
        return [str(controller_python), "-u", str(wrapper), "--output", str(output)]

    monkeypatch.setattr(case_jobs, "invocation", invocation)
    (tmp_path / "research_runs/rsi").mkdir(parents=True)

    async def scenario():
        async with case_jobs.CaseJobBroker(
            root=tmp_path,
            workspace=tmp_path / "workspace",
            output=tmp_path / "research_runs/rsi/broker",
            allowed_plugins=["catapro.training.v1"],
            timeout_s=30,
        ) as broker:
            job = broker.submit("catapro.training.v1", {"operation": "reproduce"}, "cancel_probe")
            destination = Path(broker.jobs[job["job_id"]]["output"])
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert marker.exists()
            record = json.loads((destination / "report.json").read_text())
            worker_pid = record["stages"][0]["process"]["pid"]
            controller_pid = record["controller"]["pid"]
            child_pid = int(marker.read_text())
        assert broker.jobs[job["job_id"]]["status"] == "cancelled"
        final = json.loads((destination / "report.json").read_text())
        assert final["status"] == "failed" and final["error_type"] == "KeyboardInterrupt"
        for pid in [controller_pid, worker_pid, child_pid]:
            deadline = time.monotonic() + 2
            while live(pid) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert not live(pid)

    asyncio.run(scenario())
