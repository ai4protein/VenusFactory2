"""Real queued CPU analysis through native HTTP tools; no LLM or GPU calls."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import pbs_goal_tools, web_native_worker
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.research_plan_evidence import checkpoint_steps
from agent.harness.research_workspace import ResearchWorkspace
from backend.web_v2.job_queue import TERMINAL, JobQueue, process_token
from mcp_tools.scheduler import client as scheduler_client
from mcp_tools.scheduler.catalog import native_tools


@pytest.mark.asyncio
async def test_native_queued_analysis_binds_real_execution_and_rejects_changed_output(tmp_path, monkeypatch):
    private = tmp_path / "attempt/runtime"
    private.mkdir(parents=True, mode=0o700)
    env = {"VENUS_RUNTIME_ROOT": private, "VENUS_JOB_QUEUE_ROOT": private / "queue",
           "TEMP_OUTPUTS_DIR": private / "outputs", "VENUS_USAGE_DB": private / "usage.sqlite",
           "VENUS_ACCESS_DB": private / "access.sqlite", "PYTHON_DOTENV_DISABLED": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "WEBUI_V2_MODE": "local", "VENUS_SCIENTIST_ONLINE": "0",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VENUS_WORKER_AUTOSTART": "1",
           "VENUS_QUEUE_LEASE_SECONDS": "120"}
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    for family in ("CPU", "GPU", "API", "RESEARCH"):
        monkeypatch.setenv("VENUS_WORKERS_" + family, "1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    for name in ("VENUS_QUEUE_JOB_ID", "VENUS_QUEUE_OWNER", "VENUS_QUEUE_LEASE_TOKEN", "MCP_HTTP_PORT",
                 "VENUS_CAPABILITY_POLICY_SNAPSHOT", "VENUS_MODEL_RUNTIMES_CONFIG", "VENUS_SCIENTIST_PBS_RECORD"):
        monkeypatch.delenv(name, raising=False)
    # This fixture exercises an ordinary Web Scientist parent; only the unrelated
    # PBS allocation gate is replaced. HTTP, queue leases and evidence are real.
    monkeypatch.setattr(pbs_goal_tools, "_verified_root", lambda record: private)
    record = {"nonce": "a" * 32, "settings": {"tool_credential_refs": []}}
    session, owner = "queued-analysis-fixture", "visitor:queued-analysis-fixture"
    goal = tmp_path / "goals" / hashlib.sha256(session.encode()).hexdigest() / ("b" * 32)
    workspace = goal / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "protein.fasta").write_text(">synthetic-sequence\nMKWVTFISLLFLFSSAYS\n")
    plan = {"deliverable": "analysis", "summary": "Compute actual sequence properties", "needs_input": [],
            "steps": [{"id": kind, "title": kind, "acceptance": "Provide measured supporting evidence",
                       "verification": kind} for kind in ("execution", "data", "model")]}
    research = ResearchWorkspace(workspace, goal / "control/tools", plan)
    queue = JobQueue(private / "queue")
    identity, created = queue.enqueue(owner, "scientist", "research", {"kind": "integration_fixture"}, "fixture-parent",
                                      metadata={"goal_folder": str(goal), "goal_id": goal.name, "session_id": session},
                                      session_key=session)
    assert created
    parent = queue.claim_next("fixture-parent", "research")
    assert parent["id"] == identity and parent["lease"]
    for key, value in {"VENUS_QUEUE_JOB_ID": identity, "VENUS_QUEUE_OWNER": owner,
                       "VENUS_QUEUE_LEASE_TOKEN": parent["lease"]}.items():
        monkeypatch.setenv(key, value)
    stopping = asyncio.Event()
    beats = []

    async def heartbeat():
        while not stopping.is_set():
            await asyncio.to_thread(queue.heartbeat, identity, parent["lease"])
            beats.append(time.monotonic())
            try:
                await asyncio.wait_for(stopping.wait(), .2)
            except TimeoutError:
                pass

    keeper = asyncio.create_task(heartbeat())
    observed, evidence_links, sidecar = [], [], {}

    async def native_adapter(context):
        connection = json.loads((context.task.capability / "connection.json").read_text())
        async with httpx.AsyncClient(trust_env=False, timeout=60) as http:
            async def call(name, arguments):
                response = await http.post(connection["url"],
                    headers={"Authorization": "Bearer " + connection["token"]},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})
                response.raise_for_status()
                body = response.json()
                assert "error" not in body, body
                assert body["result"].get("isError") is not True, body
                return body["result"]

            await call("describe_tool", {"name": "calculate_physchem_from_fasta"})
            result = await call("call_tool", {"name": "calculate_physchem_from_fasta", "arguments": {
                "fasta_file": "/workspace/protein.fasta", "output_file": "/workspace/results/physchem.json"}})
            for item in result["content"]:
                if item.get("type") != "text":
                    continue
                try:
                    value = json.loads(item["text"])
                except ValueError:
                    continue
                if isinstance(value, dict) and "analysis_evidence" in value:
                    evidence_links.append(value["analysis_evidence"])
            assert len(evidence_links) == 1
            await call("submit_research_checkpoint", {
                "summary": "Sequence calculations completed; host must distinguish data/model evidence.",
                "completed_steps": ["execution", "data", "model"],
                "artifacts": [evidence_links[0]["path"], "results/physchem.json"], "needs_input": []})
        return {"status": "completed", "text": "Measured sequence properties and saved the checkpoint."}

    monkeypatch.setattr(web_native_worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=native_adapter), None)))
    passed = False
    try:
        async with pbs_goal_tools.tool_service(record) as current_sidecar:
            sidecar.update(current_sidecar)
            tool = native_tools()["calculate_physchem_from_fasta"]
            profile = Profile("science_agent", "kimi-code", ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                              Budget(tool_calls=4, wall_seconds=120))
            await web_native_worker.run({"profile": profile.public(), "workspace": str(workspace),
                "control": str(goal / "control"), "session_id": session, "runtimes": {},
                "base_url": "http://127.0.0.1:1", "token": "synthetic-unused-relay",
                "text": "Compute the uploaded sequence properties and checkpoint the measured result."},
                tools=[tool, *research.tools()], event_sink=observed.append,
                analysis_recorder=research.record_queued_analysis)
            records = [row["result"] for row in research.receipts if row["tool"] == "queued_analysis"]
            assert len(records) == 1
            saved = records[0]
            job = await scheduler_client.rpc("tool_job", {"identity": saved["task_id"], "owner": owner})
            assert job["status"] == "completed" and job["parent_job"] == identity
            assert job["session_id"] == session and job["result"]["is_error"] is False
            assert saved["owner_sha256"] == hashlib.sha256(owner.encode()).hexdigest()
            assert saved["evidence"] == job["result"]["analysis_evidence"]
            assert set(saved["evidence"]["input_files"]) == {"protein.fasta"}
            assert set(saved["evidence"]["output_files"]) == {"results/physchem.json"}
            output = workspace / "results/physchem.json"
            measured = json.loads(output.read_text())
            assert measured["sequence_length"] == 18 and measured["molecular_weight"] > 0
            assert (workspace / saved["artifact"]["path"]).is_file()
            with pytest.raises(scheduler_client.ToolJobError) as denied:
                await scheduler_client.rpc("tool_job", {"identity": saved["task_id"], "owner": "another-visitor"})
            assert denied.value.code == 404
            checks = checkpoint_steps(plan, research.checkpoint, workspace=workspace,
                                      receipts=research.receipts, experiments=[], progress={})
            by_id = {row["id"]: row for row in checks}
            assert by_id["execution"]["passed"] is True
            assert by_id["execution"]["analysis_job_ids"] == [saved["task_id"]]
            assert by_id["data"]["passed"] is False and by_id["model"]["passed"] is False
            measured["molecular_weight"] += 1
            output.write_text(json.dumps(measured))
            invalidated = checkpoint_steps(plan, research.checkpoint, workspace=workspace,
                                           receipts=research.receipts, experiments=[], progress={})
            assert not any(row["passed"] for row in invalidated)
            assert not next(row for row in invalidated if row["id"] == "execution")["analysis_job_ids"]
            assert queue.get(identity, owner)["status"] == "running"
            assert len(beats) > 1
        passed = True
    finally:
        stopping.set()
        await keeper
        for row in queue.list(owner):
            if row["id"] != identity and row["status"] not in TERMINAL:
                queue.cancel(row["id"], owner)
        queue.finish(identity, "completed" if passed else "failed", lease=parent["lease"])
    assert process_token(sidecar["pid"]) is None
    assert json.loads((private / "mcp-cleanup.json").read_text())["status"] == "stopped"
    assert not any(row["status"] in {"queued", "running"} for row in queue.list(owner))
    assert any(row["kind"] == "completed" for row in observed)
