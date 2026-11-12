"""Real queue admission and CPU inference for reviewed synthetic RSI models."""

import json
from functools import partial
from pathlib import Path

import pytest
from fastapi import HTTPException
from test_capability_admission import model_candidate, reviewer

from agent.harness import research_workspace as runtime
from agent.harness.capability_review import review_proposal
from agent.harness.curated_publication import publish_candidate
from backend.web_v2 import compute_access
from backend.web_v2.job_queue import JobQueue
from mcp_tools.scheduler import catalog, jobs, worker
from mcp_tools.tools import runtime_tool_policy
from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS


@pytest.mark.asyncio
@pytest.mark.parametrize("online", [False, True], ids=["local", "online"])
async def test_reviewed_generated_model_runs_in_real_cpu_queue_with_no_gpu_available(tmp_path, monkeypatch, online):
    # Admission, persisted dispatch, claim, ownership and quotas are real. Only
    # the independent reviewer is deterministic; model inference uses bubblewrap.
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.delenv("VENUS_SCIENTIST_ONLINE", raising=False)
    for name in ("VENUS_QUEUE_JOB_ID", "VENUS_QUEUE_OWNER", "VENUS_QUEUE_LEASE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VENUS_WORKER_AUTOSTART", "0")
    monkeypatch.setenv("VENUS_QUEUE_CPU_SLOTS", "1")
    monkeypatch.setenv("VENUS_QUEUE_GPU_SLOTS", "0")
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("VENUS_ACCESS_DB", str(tmp_path / "access.sqlite3"))
    monkeypatch.setenv("VENUS_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "sandbox"))
    monkeypatch.setenv("VENUS_SESSION_SANDBOX", "bwrap")
    monkeypatch.setenv("VENUS_ONLINE_IP_COMPUTE_STARTS_PER_MINUTE", "1000")
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: False)
    monkeypatch.setattr(compute_access, "online", lambda: False)

    service = await model_candidate(tmp_path / "study")
    bundle = service.proposals[0]
    review = await review_proposal(bundle, reviewer=reviewer, control=tmp_path / "review", runner=runtime.run_python)
    assert review["status"] == "approved" and review["automatic"]["evaluation"]["training_reproduced"]
    shared = tmp_path / "shared"
    monkeypatch.setattr(runtime, "LIBRARY", shared)
    card = publish_candidate(bundle, "queue-fixture", library=shared, author_email="fixture@example.org", curator="fixture operator")
    # The runtime default argument predates the test's isolated public library.
    monkeypatch.setattr(runtime, "run_capability", partial(runtime.run_capability, library=shared))
    mapping = {tool.name: tool for tool in RESEARCH_CAPABILITY_TOOLS}
    monkeypatch.setattr(jobs, "native_tools", lambda: mapping)
    monkeypatch.setattr(catalog, "native_tools", lambda: mapping)
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: online)
    monkeypatch.setattr(compute_access, "online", lambda: online)
    monkeypatch.setenv("WEBUI_V2_MODE", "online" if online else "local")

    allowance = compute_access.ComputeStore()
    allowance.bind("alice", ("shared-lab-network", "alice-visitor"))
    allowance.bind("bob", ("shared-lab-network", "bob-visitor"))
    if online:
        allowance.admit("previous-model-plaza-runs", "alice", {"prediction": 19})
    queue = JobQueue()
    scheduler = jobs.ToolJobs(queue)
    # capacity currently clamps zero to one. Occupy its actual GPU capacity as
    # well, so this regression does not mistake GPU_SLOTS=0 for a disabled GPU.
    blockers = []
    claimed = None
    submitted = None
    try:
        for _ in range(queue.capacity("gpu")):
            identity = queue.submit("gpu-fixture-owner", "fixture", "gpu")
            queue.bind(identity)
            assert queue.claim(identity)
            blockers.append(identity)
        submitted = scheduler.submit(
            "run_generated_capability", {"capability_id": card["id"], "inputs": {"sequence": "ACDE"}},
            owner="alice", session_id="alice-session",
        )
        row = queue.get(submitted["id"], "alice")
        assert row["status"] == "queued" and row["resource"] == "cpu"
        assert row["kind"] == "mcp:model_prediction" and row["metadata"]["session_id"] == "alice-session"
        assert queue.claim_next(row["namespace"], "gpu") is None
        claimed = queue.claim_next(row["namespace"], "cpu")
        assert claimed and claimed["id"] == submitted["id"] and claimed["lease"]
        assert scheduler.list("bob") == []
        with pytest.raises(FileNotFoundError):
            scheduler.get(submitted["id"], "bob")
        with pytest.raises(FileNotFoundError):
            scheduler.get(submitted["job_number"], "bob")

        # Execute the actual worker entrypoint only after a real queue claim.
        # Its CPU sandbox reuses this lease instead of waiting on its own slot.
        monkeypatch.setenv("VENUS_QUEUE_JOB_ID", claimed["id"])
        monkeypatch.setenv("VENUS_QUEUE_OWNER", "alice")
        monkeypatch.setenv("VENUS_QUEUE_LEASE_TOKEN", claimed["lease"])
        spec_path = Path(json.loads(row["dispatch"])["command"][-1])
        result = await worker.execute(json.loads(spec_path.read_text()))
        assert not result["is_error"] and json.loads(result["value"]) == {"value": 4}
        jobs.save_json(queue.files(claimed["id"]) / "mcp-result.json", result)
        queue.finish(claimed["id"], "completed", lease=claimed["lease"])
        spec_path.unlink()
        assert scheduler.get(submitted["id"], "alice")["status"] == "completed"
        assert all(queue.get(identity)["status"] == "running" for identity in blockers)
        assert allowance.status("alice")["prediction"]["used"] == (20 if online else 0)
        assert allowance.status("bob")["prediction"]["used"] == 0
        if online:
            with pytest.raises(HTTPException) as error:
                scheduler.submit(
                    "run_generated_capability", {"capability_id": card["id"], "inputs": {"sequence": "ACDEF"}},
                    owner="alice", session_id="alice-session",
                )
            assert error.value.status_code == 429
            assert len([item for item in queue.list("alice") if item["kind"].startswith("mcp:")]) == 1
            assert not list((queue.root / "mcp-inputs").glob("*.json"))
    finally:
        if submitted and queue.get(submitted["id"])["status"] in {"running", "queued"}:
            queue.finish(submitted["id"], "cancelled", lease=claimed["lease"] if claimed else None)
        for identity in blockers:
            queue.finish(identity, "completed")


def test_generated_inference_remains_metered_and_plugins_can_override_resources():
    policy = catalog.classify("run_generated_capability")
    assert policy["resource"] == "cpu" and policy["execution"] == "queued"
    assert compute_access.demands_for("mcp:" + policy["category"], {"tool": "run_generated_capability"}) == {"prediction": 1}
    assert catalog.classify("run_generated_capability", {"queue_resource": "gpu"})["resource"] == "gpu"
    assert catalog.classify("proteinmpnn_sequence_design_from_structure")["resource"] == "gpu"
    assert catalog.classify("train_protein_model")["resource"] == "gpu"
