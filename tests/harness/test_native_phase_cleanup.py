"""Native phase teardown must finish already admitted MCP sandbox cleanup."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.tools import StructuredTool

from agent.harness import session_sandbox, web_native_worker
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.sandbox_resources import Reservations


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["completed", "cancelled"])
async def test_native_phase_exit_reaps_inflight_sandbox_before_return(tmp_path, monkeypatch, ending):
    from mcp_tools.scheduler import catalog

    monkeypatch.setattr(catalog, "native_tools", lambda: {})
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.delenv("VENUS_SCIENTIST_ONLINE", raising=False)
    monkeypatch.delenv("VENUS_QUEUE_JOB_ID", raising=False)
    monkeypatch.setenv("VENUS_SESSION_SANDBOX", "bwrap")
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "reservations"))
    monkeypatch.setenv("VENUS_SANDBOX_LOCAL_MEMORY_MB", "512")
    monkeypatch.setenv("VENUS_SANDBOX_TOTAL_MEMORY_MB", "1024")
    monkeypatch.setenv("VENUS_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    events = []
    work, control = tmp_path / "work", tmp_path / "control"
    work.mkdir()
    control.mkdir()
    (work / "wait.py").write_text("import time\ntime.sleep(120)\n")
    started = asyncio.Event()

    async def run_script(path: str) -> dict:
        """Execute saved Python in this session."""

        def progress(value):
            if value.get("status") == "running":
                started.set()

        return await session_sandbox.execute(work, tmp_path / "sandbox", "python", path, 120, on_progress=progress)

    tool = StructuredTool.from_function(
        coroutine=run_script, name="run_session_python", metadata={"workspace_relative_paths": True}
    )
    clients, requests = [], []

    async def adapter(ctx):
        config = json.loads((ctx.task.capability / "connection.json").read_text())
        client = httpx.AsyncClient(timeout=130, trust_env=False)
        clients.append(client)
        request = asyncio.create_task(
            client.post(
                config["url"],
                headers={"Authorization": "Bearer " + config["token"]},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "run_session_python", "arguments": {"path": "wait.py"}},
                },
            )
        )
        requests.append(request)
        await asyncio.wait_for(started.wait(), 15)
        if ending == "cancelled":
            await asyncio.Event().wait()
        return {"status": "completed", "text": "Phase ended while a tool was active."}

    monkeypatch.setattr(
        web_native_worker,
        "builtin_registry",
        lambda: SimpleNamespace(resolve=lambda *args: (SimpleNamespace(run=adapter), None)),
    )
    profile = Profile(
        "science_agent", "codex", ModelBinding("fixture", "unused", "http://127.0.0.1:1"), Budget(tool_calls=2)
    )
    run = asyncio.create_task(
        web_native_worker.run(
            {
                "profile": profile.public(),
                "workspace": str(work),
                "control": str(control),
                "session_id": "fixture",
                "runtimes": {},
                "base_url": "http://127.0.0.1:1",
                "token": "fixture",
                "text": "Run analysis",
            },
            tools=[tool],
            event_sink=events.append,
        )
    )
    try:
        if ending == "cancelled":
            await asyncio.wait_for(started.wait(), 15)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run, 10)
        else:
            await asyncio.wait_for(run, 20)
        receipts = list((tmp_path / "sandbox").glob("*.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert receipt["status"] == "cancelled" and receipt["temporary_files_removed"]
        assert not list((tmp_path / "sandbox").glob("sandbox-*"))
        with Reservations().locked() as rows:
            assert not rows
        assert not (control / "native/capability/connection.json").exists()
        with sqlite3.connect(tmp_path / "usage.sqlite3") as database:
            statuses = database.execute(
                "SELECT status,latency_ms FROM usage_events WHERE kind='tool' AND name='run_session_python'"
            ).fetchall()
        assert len(statuses) == 1 and statuses[0][0] == "cancelled" and statuses[0][1] is not None
        assert any(row.get("kind") == "research_tool_status" and row.get("status") == "cancelled" for row in events)
    finally:
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        for request in requests:
            if not request.done():
                request.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        await asyncio.gather(*(client.aclose() for client in clients))
