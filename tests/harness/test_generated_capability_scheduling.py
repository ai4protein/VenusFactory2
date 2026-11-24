"""Queue admission and historical resource policies for reviewed RSI models."""

from functools import partial

import pytest
from test_capability_admission import model_candidate, reviewer

from agent.harness import research_workspace as runtime
from agent.harness.capability_review import review_proposal
from agent.harness.curated_publication import publish_candidate
from backend.web_v2 import compute_access
from backend.web_v2.job_queue import JobQueue
from mcp_tools.scheduler import catalog, jobs
from mcp_tools.tools import runtime_tool_policy
from mcp_tools.tools.discovery.research_capabilities import RESEARCH_CAPABILITY_TOOLS


@pytest.mark.asyncio
@pytest.mark.parametrize("online", [False, True], ids=["local", "online"])
async def test_reviewed_generated_model_cannot_be_queued_or_charge_a_visitor(tmp_path, monkeypatch, online):
    # Prepare a reviewed and curated case, then exercise its public admission policy.
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
    # Publication in RSI Plaza never grants an execution entry point. Exercise
    # the real admission boundary before it can create a job or debit a visitor.
    before = allowance.status("alice")["prediction"]["used"]
    with pytest.raises(RuntimeError, match="RSI Plaza"):
        scheduler.submit("run_generated_capability",
                         {"capability_id": card["id"], "inputs": {"sequence": "ACDE"}},
                         owner="alice", session_id="alice-session")
    assert scheduler.list("alice") == []
    assert scheduler.list("bob") == []
    assert allowance.status("alice")["prediction"]["used"] == before
    assert allowance.status("bob")["prediction"]["used"] == 0
    assert not list((queue.root / "mcp-inputs").glob("*.json"))


def test_generated_inference_remains_metered_and_plugins_can_override_resources():
    policy = catalog.classify("run_generated_capability")
    assert policy["resource"] == "cpu" and policy["execution"] == "queued"
    assert compute_access.demands_for("mcp:" + policy["category"], {"tool": "run_generated_capability"}) == {"prediction": 1}
    assert catalog.classify("run_generated_capability", {"queue_resource": "gpu"})["resource"] == "gpu"
    assert catalog.classify("proteinmpnn_sequence_design_from_structure")["resource"] == "gpu"
    assert catalog.classify("train_protein_model")["resource"] == "gpu"
