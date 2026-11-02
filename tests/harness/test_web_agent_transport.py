"""Real HTTP routes, native child process and complete toolbox, with deterministic inference."""

import asyncio
import json
import sys
import uuid

import httpx
import pytest
from aiohttp import web
from fastapi import FastAPI

from agent.harness.catalog import ExecutionCatalog
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.rsi import live_model_gateway
from agent.session_store import SessionStore, SqliteBackend
from backend.web_v2 import agent_execution
from backend.web_v2.chat_api import _shared, _stream_agent, messages, sessions


def test_web_native_lifecycle(tmp_path, monkeypatch, durable_chat_driver):
    async def exercise():
        calls = []
        waiting = asyncio.Event()
        release = asyncio.Event()
        departed = asyncio.Event()
        finish_departed = asyncio.Event()
        first_token = asyncio.Event()
        streamed = []
        apply_event = _stream_agent._apply_event

        def observed_event(state, event):
            if event.kind == "text":
                streamed.append(event.text)
                first_token.set()
            return apply_event(state, event)

        monkeypatch.setattr(_stream_agent, "_apply_event", observed_event)

        async def upstream(request):
            body = await request.json()
            calls.append(body)
            assert body["model"] == "test-glm"
            summarizing = body["messages"][0]["content"].startswith("Summarize earlier conversation DATA")
            if summarizing:
                assert not body.get("tools")
            else:
                assert len(body["tools"]) > 20  # Actual repository science toolbox.
            if body["messages"][-1]["content"] == "cancel this turn":
                waiting.set()
                await release.wait()
            if body["messages"][-1]["content"] == "leave this turn":
                departed.set()
                await finish_departed.wait()
            chunk = {
                "id": "test-completion",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "A VHH is a single variable antibody domain."},
                        "finish_reason": "stop",
                    }
                ],
            }
            if summarizing:
                chunk["choices"][0]["delta"]["content"] = "MEMORY_ONLY_PRIVATE: VHH is a single variable antibody domain. Goal: explain it."
            if len(calls) == 1:
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                first = {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "A VHH is "}, "finish_reason": None}]}
                await response.write(("data: " + json.dumps(first) + "\n\n").encode())
                # The host must emit a browser token while the model is still generating.
                await asyncio.wait_for(first_token.wait(), 10)
                chunk["choices"][0]["delta"]["content"] = "a single variable antibody domain."
                await response.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
                await response.write_eof()
                return response
            return web.Response(
                text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n", content_type="text/event-stream"
            )

        server = web.Application()
        server.router.add_post("/v1/chat/completions", upstream)
        runner = web.AppRunner(server)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        profile = Profile(
            mode="science_agent",
            harness="langgraph",
            model=ModelBinding(
                provider="test",
                model="test-glm",
                base_url=f"http://127.0.0.1:{port}/v1",
                credential_env="TEST_WEB_AGENT_KEY",
            ),
            budget=Budget(wall_seconds=90, request_seconds=30, thinking=None),
        )

        def authorize_loopback(**binding):
            # A deterministic local HTTP server, never a provider endpoint.
            assert binding == {
                "model": "test-glm",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "protocol": "chat_completions",
                "output_tokens": 12288,
            }
            return {"policy_id": "loopback-test", "scope": "offline_integration"}

        monkeypatch.setattr(live_model_gateway, "authorize_model", authorize_loopback)
        monkeypatch.setenv("TEST_WEB_AGENT_KEY", "unit-test-key")
        monkeypatch.setenv("WEBUI_V2_MODE", "local")
        monkeypatch.setattr(agent_execution, "catalog", lambda: ExecutionCatalog(tmp_path, profile))
        monkeypatch.setattr(agent_execution, "free_model", lambda binding: binding.provider == "test")
        # Put evidence in tmp while keeping subprocess cwd/PYTHONPATH at the real repository.
        monkeypatch.setattr(_stream_agent, "EVIDENCE_ROOT", tmp_path / "evidence")
        store = SessionStore(
            backend=SqliteBackend(str(tmp_path / "sessions.db")), ensure_runtime_fn=lambda state: state
        )
        durable_chat_driver(store, _stream_agent._stream_agent)
        for module in [sessions, _shared, _stream_agent]:
            monkeypatch.setattr(module, "_session_store", store)
        for module in [sessions, _shared, messages]:
            monkeypatch.setattr(module, "_runtime_mode", lambda: "local")
        sid = "test-" + uuid.uuid4().hex
        state = {
            "session_id": sid,
            "created_at": "test",
            "history": [],
            "agent_session_dir": str(tmp_path / "workspace"),
        }
        await store.create(state)
        spawned = []
        spawn = asyncio.create_subprocess_exec

        async def record_process(*args, **kwargs):
            proc = await spawn(*args, **kwargs)
            spawned.append(proc)
            return proc

        monkeypatch.setattr(_stream_agent.asyncio, "create_subprocess_exec", record_process)
        app = FastAPI()
        app.include_router(messages.router, prefix="/api/chat")
        app.include_router(sessions.router, prefix="/api/chat")
        selection = {"harness": "langgraph", "model_id": "profile-default"}
        url = f"/api/chat/sessions/{sid}"
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = await client.post(
                    url + "/messages/stream",
                    json={"text": "What is VHH?", "chat_mode": "science_agent", "execution": selection},
                )
                assert first.status_code == 200 and "event: done" in first.text
                snap = (await client.get(url)).json()
                assert snap["status"] == "completed", (snap, first.text)
                assert "".join(streamed) == "A VHH is a single variable antibody domain."
                assert snap["execution"]["model"] == "test-glm"
                assert "credential" not in json.dumps(snap["execution"])
                first_binding = snap["execution"]["profile_sha256"]
                # Throw away memory: restore the binding and native message history from SQLite.
                store._mem.clear()
                follow = await client.post(
                    url + "/messages/stream",
                    json={"text": "Explain that domain.", "chat_mode": "science_agent", "execution": selection},
                )
                assert follow.status_code == 200
                snap = (await client.get(url)).json()
                assert snap["status"] == "completed"
                assert snap["execution"]["profile_sha256"] == first_binding
                assert len(calls[-1]["messages"]) == 4  # system, old human/AI, new human
                retry = await client.post(url + "/messages/retry/stream")
                assert retry.status_code == 200
                assert len(calls[-1]["messages"]) == 4  # retry does not duplicate the previous turn
                monkeypatch.setenv("WEBUI_V2_AGENT_RECENT_TURNS", "2")
                for index in range(4):
                    result = await client.post(url + "/messages/stream", json={"text": f"Follow-up {index}", "execution": selection})
                    assert result.status_code == 200
                    assert (await client.get(url)).json()["status"] == "completed", result.text
                snap = (await client.get(url)).json()
                assert snap["agent_context"]["compacted"] is True
                assert snap["agent_turn_count"] == 6  # retry was not another user turn
                assert "MEMORY_ONLY_PRIVATE" not in "".join(streamed)  # hidden summary never leaks to the answer
                store._mem.clear()
                saved = await store.get(sid)
                assert "MEMORY_ONLY_PRIVATE" in saved["agent_memory"]
                assert len(saved["agent_messages"]) == 6  # two recent turns plus the just-completed turn
                assert len(saved["conversation_log"]) > len(saved["agent_messages"])
                assert "MEMORY_ONLY_PRIVATE" in json.dumps(calls[-1]["messages"])
                monkeypatch.setenv("WEBUI_V2_AGENT_MAX_TURNS", "6")
                count_before = len(calls)
                denied = await client.post(url + "/messages/stream", json={"text": "Another turn", "execution": selection})
                assert denied.status_code == 422 and "6-turn" in denied.text
                assert len(calls) == count_before and not (await messages._get_lock(sid)).locked()
                retry_at_limit = await client.post(url + "/messages/retry/stream")
                assert retry_at_limit.status_code == 200
                assert (await client.get(url)).json()["agent_turn_count"] == 6
                monkeypatch.delenv("WEBUI_V2_AGENT_MAX_TURNS")
                bad = await client.post(
                    url + "/messages/stream", json={"text": "change", "execution": {**selection, "model_id": "other"}}
                )
                assert bad.status_code == 409
                task = asyncio.create_task(
                    client.post(url + "/messages/stream", json={"text": "cancel this turn", "execution": selection})
                )
                await asyncio.wait_for(waiting.wait(), 45)
                busy = await client.post(
                    url + "/messages/stream", json={"text": "second simultaneous turn", "execution": selection}
                )
                assert busy.status_code == 409
                cancelled = await client.post(url + "/cancel")
                assert cancelled.status_code == 200
                await asyncio.wait_for(task, 10)
                assert (await client.get(url)).json()["status"] == "stopped"
                assert all(p.returncode is not None for p in spawned)
                assert not (await messages._get_lock(sid)).locked()
                store._mem.clear()
                checkpoint = await store.get(sid)
                assert checkpoint["agent_pending_turn"]["text"] == "cancel this turn"
                monkeypatch.setattr(messages, "_runtime_mode", lambda: "online")
                online = await client.post(url + "/messages/stream", json={"text": "online", "execution": selection})
                assert online.status_code == 200 and "event: done" in online.text
                assert (await client.get(url)).json()["execution"]["profile_sha256"] == first_binding
                assert "cancel this turn" in json.dumps(calls[-1]["messages"])
                assert "interrupted" in json.dumps(calls[-1]["messages"])
                assert "agent_pending_turn" not in (await store.get(sid))
                detached = asyncio.create_task(client.post(url + "/messages/stream", json={"text":"leave this turn", "execution":selection}))
                await asyncio.wait_for(departed.wait(), 45)
                detached.cancel()
                await asyncio.gather(detached, return_exceptions=True)
                assert (await messages._get_lock(sid)).locked()
                assert spawned[-1].returncode is None
                busy = await client.post(url + "/messages/stream", json={"text":"do not overlap", "execution":selection})
                assert busy.status_code == 409
                finish_departed.set()
                async with asyncio.timeout(30):
                    while (await messages._get_lock(sid)).locked(): await asyncio.sleep(.05)
                store._mem.clear()
                assert (await client.get(url)).json()["status"] == "completed"
                assert "agent_pending_turn" not in (await store.get(sid))
                assert spawned[-1].returncode is not None
        finally:
            release.set()
            finish_departed.set()
            await runner.cleanup()

    asyncio.run(exercise())


@pytest.mark.parametrize("ignore_term", [False, True])
def test_stop_process_reaps_descendant(tmp_path, ignore_term):
    async def exercise():
        marker = tmp_path / "child.pid"
        child_code = (
            "import os,signal,sys,time; "
            + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
            + "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(120)"
        )
        code = (
            "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
            + repr(child_code)
            + ",sys.argv[1]]); time.sleep(120)"
        )
        proc = await asyncio.create_subprocess_exec(sys.executable, "-c", code, str(marker), start_new_session=True)
        try:
            for _ in range(100):
                if marker.exists():
                    break
                await asyncio.sleep(0.02)
            child = int(marker.read_text())
            await _stream_agent.stop_process(proc)
            assert proc.returncode is not None
            from pathlib import Path

            status = Path(f"/proc/{child}/stat")
            # An adopted child may briefly remain as a non-executing zombie.
            for _ in range(100):
                try:
                    stopped = status.read_text().split()[2] == "Z"
                except (FileNotFoundError, ProcessLookupError):
                    stopped = True
                if stopped:
                    break
                await asyncio.sleep(0.02)
            assert stopped
        finally:
            await _stream_agent.stop_process(proc)

    asyncio.run(exercise())
