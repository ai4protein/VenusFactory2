"""Full notebook admission and explicit negative scientific outcomes."""

import asyncio
import os
import signal
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from agent.rsi import case_jobs
from agent.rsi.case_evidence import canonical_claims, render_evidence_report, validate_review
from agent.rsi.case_plugins import invocation, list_case_plugins, resource_for
from agent.rsi.mcba_evidence import COMMIT, TASK_IDS, summarize_campaign


def complete_report():
    tasks, checked = [], []
    for tid in TASK_IDS:
        rank = tid.startswith("rank_")
        reference = True if rank or tid.endswith("_full") else None
        tasks.append({"id": tid, "status": "completed", "reference_passed": reference, "top_exact_order_match": rank})
        checked.append(
            {
                "id": tid,
                "architectures": 1 if rank else 40,
                "prediction_rows_per_architecture": 160000,
                "numerical_prediction_replay_passed": True,
                "test_metric_discrepancies": [],
                "historical_reference_passed": reference,
            }
        )
    # Preserve a failed numerical agreement check in a completed computation.
    checked[9]["numerical_prediction_replay_passed"] = False
    original = {"status": "campaign_completed", "tasks": tasks}
    audit = {
        "campaign_completed": True,
        "source_integrity_passed": True,
        "all_selected_tasks_completed": True,
        "independent_prediction_replay_passed": False,
        "tasks": checked,
    }
    return {
        "status": "completed",
        "plugin_id": "mcba.multisubstrate.v1",
        "operation": "reproduce",
        "requested_campaign": None,
        "execution_origin": "new_computation",
        "source_commit": COMMIT,
        "source_campaign": "fixture/campaign",
        "campaign_evidence": original,
        "audit_evidence": audit,
        "summary": summarize_campaign(original, audit),
    }


def test_plugin_only_admits_complete_fixed_scope(tmp_path):
    def recipe(parameters):
        return invocation(
            "mcba.multisubstrate.v1",
            parameters,
            root=tmp_path,
            output=tmp_path / "research_runs/rsi/run",
            controller_python=Path(sys.executable),
        )

    assert recipe({"operation": "reproduce"})[-2:] == ["--operation", "reproduce"]
    assert recipe({"operation": "verify_existing", "campaign": "author_multisubstrate_20261012"})[-2:] == [
        "--campaign",
        "author_multisubstrate_20261012",
    ]
    for invalid in [
        {},
        {"operation": "reproduce", "tasks": ["rank_itopride"]},
        {"operation": "verify_existing", "campaign": "../../other"},
        {"operation": "reproduce", "source": "/private"},
    ]:
        with pytest.raises(ValueError):
            recipe(invalid)
    assert resource_for("mcba.multisubstrate.v1", {}) == "cpu"
    assert any(p["id"] == "mcba.multisubstrate.v1" for p in list_case_plugins())


def test_completed_execution_does_not_erase_negative_replay():
    report = complete_report()
    claims = canonical_claims("mcba.multisubstrate.v1", report)
    assert claims["completed_tasks"] == 25
    assert claims["independent_prediction_replay_passed"] is False
    assert claims["computational_agreement_passed"] is False
    assert claims["historical_reference_counts"] == {"passed": 11, "failed": 0, "unavailable": 14}
    assert claims["protein_scaffolds"] == 1 and claims["full_paper_reproduced"] is False
    assert "agreement: False" in render_evidence_report("test", claims)
    with pytest.raises(ValueError, match="disagree"):
        validate_review({**claims, "computational_agreement_passed": True}, claims)
    report["execution_origin"] = "verified_existing"
    assert (
        canonical_claims("mcba.multisubstrate.v1", report)["repeat_computation_is_independent_efficacy_evidence"]
        is False
    )


@pytest.mark.parametrize(
    "mutation",
    ["partial", "duplicate", "missing_architectures", "non_boolean", "false_aggregate", "false_reference_availability"],
)
def test_partial_or_inconsistent_evidence_is_rejected(mutation):
    report = complete_report()
    audit = report["audit_evidence"]
    if mutation == "partial":
        audit["tasks"].pop()
    if mutation == "duplicate":
        audit["tasks"][-1] = deepcopy(audit["tasks"][0])
    if mutation == "missing_architectures":
        audit["tasks"][9]["architectures"] = 1
    if mutation == "non_boolean":
        audit["tasks"][9]["numerical_prediction_replay_passed"] = 0
    if mutation == "false_aggregate":
        audit["independent_prediction_replay_passed"] = True
    if mutation == "false_reference_availability":
        report["campaign_evidence"]["tasks"][-1]["reference_passed"] = True
    with pytest.raises(ValueError):
        canonical_claims("mcba.multisubstrate.v1", report)


@pytest.mark.asyncio
async def test_broker_returns_discrepancies_and_checks_audit_source(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "research_runs/rsi").mkdir(parents=True)
    script = root / "fixture.py"
    report = complete_report()
    script.write_text(
        "import pathlib,sys,json,hashlib\nr="
        + repr(report)
        + "\np=pathlib.Path(sys.argv[1]);(p/'audit').mkdir(parents=True)\ns=json.dumps(r['audit_evidence']);(p/'audit/audit.json').write_text(s);r['audit_report_sha256']=hashlib.sha256(s.encode()).hexdigest();(p/'report.json').write_text(json.dumps(r))\n"
    )
    monkeypatch.setattr(
        case_jobs,
        "invocation",
        lambda _plugin, _params, **kwargs: [sys.executable, "-u", str(script), str(kwargs["output"])],
    )
    broker = case_jobs.CaseJobBroker(
        root=root,
        workspace=root / "workspace",
        output=root / "research_runs/rsi/job",
        allowed_plugins={"mcba.multisubstrate.v1"},
        timeout_s=10,
    )
    async with broker:
        job = broker.submit("mcba.multisubstrate.v1", {"operation": "reproduce"}, "test")
        await broker.wait_jobs()
        result = broker.status(job["job_id"])
        assert result["status"] == "completed", result
        assert result["evidence"]["computational_agreement_passed"] is False


@pytest.mark.asyncio
async def test_broker_timeout_reaps_term_ignoring_descendant(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "research_runs/rsi").mkdir(parents=True)
    marker = root / "pid"
    script = root / "fixture.py"
    child = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open("
        + repr(str(marker))
        + ",'w').write(str(os.getpid())); time.sleep(120)"
    )
    script.write_text(
        "import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c'," + repr(child) + "])\ntime.sleep(120)\n"
    )
    monkeypatch.setattr(case_jobs, "invocation", lambda _plugin, _params, **kwargs: [sys.executable, "-u", str(script)])
    broker = case_jobs.CaseJobBroker(
        root=root,
        workspace=root / "workspace",
        output=root / "research_runs/rsi/job",
        allowed_plugins={"mcba.multisubstrate.v1"},
        timeout_s=1,
    )
    child_pid = None
    try:
        async with broker:
            job = broker.submit("mcba.multisubstrate.v1", {"operation": "reproduce"}, "test")
            await broker.wait_jobs()
            assert broker.status(job["job_id"])["status"] == "timed_out"
            child_pid = int(marker.read_text())
            for _ in range(100):
                try:
                    stopped = Path(f"/proc/{child_pid}/stat").read_text().split()[2] == "Z"
                except (FileNotFoundError, ProcessLookupError):
                    stopped = True
                if stopped:
                    break
                await asyncio.sleep(0.02)
            assert stopped
    finally:
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
