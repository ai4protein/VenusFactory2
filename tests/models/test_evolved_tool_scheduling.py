"""Queue and transport contracts; synthetic calls do not train scientific models."""

import asyncio
import hashlib
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("busy_resource,can_claim", [("gpu", True), ("cpu", False)])
def test_evolved_prediction_uses_cpu_slots_while_preserving_owner(tmp_path, monkeypatch, busy_resource, can_claim):
    from backend.web_v2.job_queue import JobQueue
    from mcp_tools.scheduler.jobs import ToolJobs
    from mcp_tools.models.evolved.tools_agent import predict_with_evolved_model_tool

    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_WORKER_AUTOSTART", "0")
    monkeypatch.setenv("VENUS_QUEUE_CPU_SLOTS", "1")
    monkeypatch.setenv("VENUS_QUEUE_GPU_SLOTS", "1")
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setattr("mcp_tools.scheduler.jobs.native_tools", lambda: {"predict_with_evolved_model": predict_with_evolved_model_tool})
    monkeypatch.setattr("mcp_tools.scheduler.jobs.assert_tool_call_allowed", lambda *args: None)
    queue = JobQueue()
    jobs = ToolJobs(queue)
    blocker = queue.submit("other-owner", "fixture", busy_resource)
    queue.bind(blocker)
    assert queue.claim(blocker)
    job = claimed = None
    try:
        job = jobs.submit("predict_with_evolved_model", {
            "entry_id": "a" * 64, "inputs": [{"id": "x", "sequence": "AA"}], "scope": {"protein": "fixture"},
        }, owner="alice", session_id="alice-session")
        row = queue.get(job["id"], "alice")
        assert row["resource"] == "cpu"
        assert row["kind"] == "mcp:model_prediction"
        assert row["metadata"]["session_id"] == "alice-session"
        claimed = queue.claim_next(row["namespace"], "cpu")
        assert (claimed is not None) is can_claim
        if claimed is not None:
            assert claimed["id"] == job["id"] and claimed["lease"]
        assert jobs.list("bob") == []
        with pytest.raises(FileNotFoundError):
            jobs.get(job["id"], "bob")
    finally:
        if job is not None:
            queue.finish(job["id"], "cancelled", lease=claimed["lease"] if claimed else None)
        queue.finish(blocker, "completed")


def test_real_gpu_models_and_explicit_plugin_resources_keep_their_queue():
    from mcp_tools.scheduler.catalog import classify

    assert classify("predict_with_evolved_model")["resource"] == "cpu"
    assert classify("predict_with_evolved_model")["execution"] == "queued"
    for name in ["predict_structure_esmfold", "proteinmpnn_sequence_design_from_structure", "train_protein_model"]:
        assert classify(name)["resource"] == "gpu"
    assert classify("predict_with_evolved_model", {"queue_resource": "gpu"})["resource"] == "gpu"


@pytest.mark.asyncio
async def test_mcp_gateway_persists_its_transport_session_in_the_job(monkeypatch):
    from mcp_tools.scheduler import mcp_service

    monkeypatch.delenv("VENUS_MCP_QUEUE_WORKER", raising=False)
    monkeypatch.setattr("mcp_tools.tools.runtime_tool_policy.assert_tool_call_allowed", lambda *args: None)
    saved = []

    class Hub:
        async def list_tools(self):
            return [SimpleNamespace(name="predict_with_evolved_model")]

    class Jobs:
        def submit(self, name, arguments, **metadata):
            saved.append(metadata)
            return {"id": "fixture-job"}

        async def wait(self, identity, owner):
            return {"status": "completed", "result": {"content": [{"type": "text", "text": "{}"}]}}

    monkeypatch.setitem(sys.modules, "mcp_tools.mcp_server", SimpleNamespace(mcp=Hub()))
    monkeypatch.setattr(mcp_service, "ToolJobs", Jobs)

    async def no_inline(context):
        pytest.fail("Prediction must go through the queue")

    for session in ["transport-alice", "transport-bob"]:
        context = SimpleNamespace(message=SimpleNamespace(name="predict_with_evolved_model", arguments={}),
                                  fastmcp_context=SimpleNamespace(session_id=session))
        await mcp_service.QueueMiddleware().on_call_tool(context, no_inline)
    assert [row.get("session_id") for row in saved] == ["transport-alice", "transport-bob"]
    assert [row["owner"] for row in saved] == ["mcp:" + hashlib.sha256(s.encode()).hexdigest() for s in ["transport-alice", "transport-bob"]]
    assert all(row["route"] == "mcp" for row in saved)


@pytest.mark.asyncio
async def test_queued_mcp_calls_use_original_sessions_without_cross_request_leakage(monkeypatch):
    from mcp_tools.models.evolved import tools_mcp
    from mcp_tools.scheduler.worker import execute

    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_MCP_QUEUE_WORKER", "1")
    monkeypatch.setitem(sys.modules, "mcp_tools.mcp_server", SimpleNamespace(mcp=tools_mcp.mcp))
    monkeypatch.setattr("mcp_tools.tools.runtime_tool_policy.assert_tool_call_allowed", lambda *args: None)
    monkeypatch.setattr(tools_mcp, "dispatch", lambda action, **kw: {"success": True, "session": kw["session_id"], "binding": kw["session_binding"]})
    arguments = {"entry_id": "a" * 64, "inputs": [{"id": "x", "sequence": "AA"}], "scope": {"protein": "fixture"}}

    async def invoke(session):
        result = await execute({"name": "predict_with_evolved_model", "route": "mcp", "arguments": arguments, "session_id": session})
        assert not result["is_error"]
        return result["structured_content"]

    results = await asyncio.gather(*(invoke(session) for session in ["alice-original", "bob-original", "alice-original"]))
    assert [row["session"] for row in results] == ["alice-original", "bob-original", "alice-original"]
    assert all(row["binding"] == "mcp_transport_session" for row in results)
    # A later direct connection must use its own transport, not a prior job's ID.
    from fastmcp import Client
    async with Client(tools_mcp.mcp) as client:
        direct = (await client.call_tool("predict_with_evolved_model", arguments)).data
    assert direct["session"] and direct["session"] not in {"alice-original", "bob-original"}


def test_queued_session_restores_after_failure_and_supports_legacy_jobs():
    from mcp_tools.scheduler.context import bind_mcp_session, mcp_session_id

    assert mcp_session_id("live") == "live"
    with bind_mcp_session("outer"):
        with pytest.raises(RuntimeError, match="fixture failure"):
            with bind_mcp_session("inner"):
                assert mcp_session_id("worker") == "inner"
                raise RuntimeError("fixture failure")
        assert mcp_session_id("worker") == "outer"
        with bind_mcp_session(None):
            assert mcp_session_id("legacy-worker") == "legacy-worker"
        assert mcp_session_id("worker") == "outer"
    assert mcp_session_id("live") == "live"


@pytest.mark.parametrize("value", ["", " ", 42, {}, "x" * 257])
def test_malformed_private_session_does_not_replace_current_context(value):
    from mcp_tools.scheduler.context import bind_mcp_session, mcp_session_id

    with bind_mcp_session("outer"):
        with pytest.raises(ValueError, match="session"):
            with bind_mcp_session(value):
                pytest.fail("Invalid job identity was accepted")
        assert mcp_session_id("live") == "outer"
