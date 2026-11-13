"""Fixed worker dispatch checks; kernel proof is tested separately without a cluster."""

import os
from types import SimpleNamespace

import pytest
from test_pbs_goal_worker import context as context

from agent.harness import cluster_allocation, pbs_goal, pbs_goal_worker
from agent.harness.session_sandbox import SandboxUnavailable


def configure_cpu(context, monkeypatch):
    context.record["scheduler"] = {
        "backend": "slurm",
        "cluster": "fixture",
        "submitted_after": 0,
        "submitted_before": 60,
    }
    context.record["resources"]["ngpus"] = 0
    context.record["job_id"] = "123"
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "fixture")
    monkeypatch.setenv("PBS_JOBID", "stale.foreign")
    monkeypatch.setenv("VENUS_SCIENTIST_PBS_RECORD", "/stale/foreign")

    def capture(record):
        context.calls.append("allocation")
        assert os.environ["SLURM_JOB_ID"] == "123"
        assert os.environ["SLURM_CLUSTER_NAME"] == "fixture"
        assert "PBS_JOBID" not in os.environ
        assert "VENUS_SCIENTIST_PBS_RECORD" not in os.environ
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["VENUS_RESEARCH_GPU_BACKEND"] != "pbs"
        assert os.environ["VENUS_SCIENTIST_CLUSTER_RECORD"] == str(context.path)
        assert "UNRELATED_HOST_SECRET" not in os.environ
        return SimpleNamespace(job_id="123", receipt=lambda: {"backend": "slurm", "job_id": "123", "device": "cpu"})

    def ready(path, receipt):
        assert receipt == {"backend": "slurm", "job_id": "123", "device": "cpu"}
        context.calls.append("ready")

    monkeypatch.setattr(cluster_allocation, "capture_allocation", capture)
    monkeypatch.setattr(pbs_goal, "worker_ready", ready)


def test_cpu_worker_reaches_existing_research_only_after_allocation_receipt(context, monkeypatch):
    configure_cpu(context, monkeypatch)
    result = pbs_goal_worker.run_record(context.path)
    assert result["status"] == "waiting_for_input"
    assert context.calls == ["claim", "source", "allocation", "ready", "free", "runtime", "run", "source"]
    assert context.finished == [("waiting_for_input", None)]
    assert context.store.read()["usage"]["model_calls"] == 0


def test_cpu_proof_failure_never_reads_credentials_or_runs_research(context, monkeypatch):
    configure_cpu(context, monkeypatch)

    def denied(record):
        raise SandboxUnavailable("Synthetic missing job cgroup")

    monkeypatch.setattr(cluster_allocation, "capture_allocation", denied)
    monkeypatch.setattr(
        pbs_goal_worker, "_selected_credential", lambda *_: pytest.fail("unverified allocation read credential")
    )
    with pytest.raises(SandboxUnavailable):
        pbs_goal_worker.run_record(context.path)
    assert context.finished == [("interrupted", "SandboxUnavailable")]
    assert "runtime" not in context.calls and "ready" not in context.calls
