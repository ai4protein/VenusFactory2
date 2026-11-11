"""Slow legacy computations must not block unrelated HTTP requests."""
import asyncio
import importlib
import threading

import httpx
import pytest
from fastapi import FastAPI, HTTPException


@pytest.mark.parametrize("mode", ["online", "local"])
@pytest.mark.parametrize("kind", ["advanced", "quick", "download", "stream"])
def test_waiting_prediction_does_not_freeze_http(monkeypatch, tmp_path, mode, kind):
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("VENUS_ACCESS_DB", str(tmp_path / "access.sqlite3"))
    from backend.web_v2 import compute_access
    # No inference, allowance changes or remote downloads in this regression.
    monkeypatch.setattr(compute_access, "inline_prediction", lambda *a: None)
    from backend.web_v2.compute_access import actor_context, inline_request_context
    from mcp_tools.tools.capability_policy import CapabilityStore
    monkeypatch.setattr(CapabilityStore, "assert_call", lambda *a: None)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    contexts = []

    def blocked(*args, **kwargs):
        contexts.append(actor_context.get())
        started.set()
        release.wait(0.8)  # Also prevents a broken implementation hanging the suite.
        finished.set()
        raise HTTPException(418, "Fixture computation completed")

    if kind in {"advanced", "stream"}:
        module = importlib.import_module("backend.web_v2.advanced_tools_api")
        monkeypatch.setattr(module, "_resolve_upload_file", lambda path: "/fixture.pdb")
        monkeypatch.setattr(module, "_run_generator_collect_last", blocked)
        path = "/api/advanced-tools/directed-evolution/run" + ("/stream" if kind == "stream" else "")
        body = {"input_mode": "structure", "file_path": "fixture.pdb", "model_name": "ProSST-2048"}
    elif kind == "quick":
        module = importlib.import_module("backend.web_v2.quick_tools_api")
        pdb = tmp_path / "fixture.pdb"
        pdb.write_text("HEADER fixture\n")
        monkeypatch.setattr(module, "_resolve_upload_input", lambda path: str(pdb))
        monkeypatch.setattr(module, "run_zero_shot_prediction", blocked)
        path, body = "/api/quick-tools/run/mutation", {"structure_file": "fixture.pdb"}
    else:
        module = importlib.import_module("backend.web_v2.download_api")
        monkeypatch.setattr(module, "_task_folder", lambda *a: tmp_path)
        monkeypatch.setattr(module, "_run_download_script", blocked)
        path, body = "/api/download/uniprot", {"id_value": "P12345"}

    app = FastAPI()
    app.include_router(module.router)

    @app.get("/health")
    async def health():
        return {"ready": True}

    async def exercise():
        actor = actor_context.set(("fixture-ip", "fixture-visitor"))
        request = inline_request_context.set({})
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                work = asyncio.create_task(client.post(path, json=body))
                try:
                    assert await asyncio.to_thread(started.wait, 2), "Fixture did not reach computation"
                    assert not finished.is_set(), "Computation blocked the HTTP event loop"
                    response = await asyncio.wait_for(client.get("/health"), 0.3)
                    assert response.status_code == 200 and response.json() == {"ready": True}
                    assert not finished.is_set(), "Health only returned after the computation finished"
                finally:
                    release.set()
                    result = await work
                if kind == "stream":
                    assert result.status_code == 200 and "Fixture computation completed" in result.text
                else:
                    assert result.status_code == 418 and result.json()["detail"] == "Fixture computation completed"
                assert contexts == [("fixture-ip", "fixture-visitor")]
        finally:
            inline_request_context.reset(request)
            actor_context.reset(actor)

    asyncio.run(exercise())
