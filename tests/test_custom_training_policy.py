import json
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from backend.web_v2 import custom_model_api as api
from backend.web_v2.job_queue import JobQueue

MODELS = {"ESM2-8M": api.ONLINE_TRAINING_MODEL, "ESM2-35M": "facebook/esm2_t12_35M_UR50D", "ProtBert": "Rostlab/prot_bert"}
CONSTANT = {"plm_models": MODELS, "dataset_configs": {}}


@pytest.fixture
def online(monkeypatch, tmp_path):
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    monkeypatch.setattr(api, "is_online_mode", lambda: True)
    monkeypatch.setattr(api, "_safe_read_json", lambda path: CONSTANT)
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setattr(api, "get_web_v2_area_dir", lambda *args, **kwargs: tmp_path / "results")
    return tmp_path


def payload(model="ESM2-8M", **extra):
    return {"plm_model": model, "dataset_custom": "public/example", **extra}


def test_online_meta_and_preview_accept_only_8m(online, monkeypatch):
    app = FastAPI(); app.include_router(api.router)
    monkeypatch.setattr(api, "_queued_stream", lambda task, script, args, owner, response: JSONResponse({"model": args["plm_model"]}))
    with TestClient(app) as client:
        meta = client.get("/api/custom-model/meta").json()
        assert meta["training_plm_models"] == {"ESM2-8M": MODELS["ESM2-8M"]}
        assert len(meta["plm_models"]) == 3  # Other pages retain their full registry.
        for endpoint in ("preview", "start"):
            assert client.post("/api/custom-model/training/" + endpoint, json={"args": payload()}).status_code == 200
            for model in ("ESM2-35M", "ProtBert"):
                response = client.post("/api/custom-model/training/" + endpoint, json={"args": payload(model)})
                assert response.status_code == 403
                assert "ESM2-8M" in response.json()["detail"]


def test_local_training_retains_larger_models(online, monkeypatch):
    monkeypatch.setattr(api, "is_online_mode", lambda: False)
    assert api._normalize_training_args(payload("ESM2-35M"), CONSTANT)["plm_model"] == MODELS["ESM2-35M"]


def test_continue_training_passes_checkpoint_and_checks_backbone(online, monkeypatch):
    model = online / "best_model.pt"; model.touch()
    monkeypatch.setattr(api, "_resolve_client_file_path", lambda path, **kw: model)
    monkeypatch.setattr(api, "_load_model_config", lambda path: {"plm_model": MODELS["ESM2-8M"]})
    args = api._normalize_training_args(payload(training_mode="Continue Training", transfer_model_dropdown=str(model)), CONSTANT)
    assert args["initial_model_path"] == str(model)
    assert "--initial_model_path" in api._build_exec_cmd("src/train.py", args)
    monkeypatch.setattr(api, "_load_model_config", lambda path: {"plm_model": MODELS["ESM2-35M"]})
    with pytest.raises(HTTPException):
        api._normalize_training_args(payload(training_mode="Continue Training", transfer_model_dropdown=str(model)), CONSTANT)
    with pytest.raises(HTTPException):
        api._normalize_training_args(payload(training_mode="Continue Training"), CONSTANT)


@pytest.mark.parametrize("extra", [{"num_epochs": 0}, {"num_epochs": 1.5}, {"learning_rate": -1}, {"learning_rate": "nan"}, {"batch_size": 0}, {"dataset_custom": ""}])
def test_invalid_training_inputs_are_rejected(online, extra):
    with pytest.raises(HTTPException): api._normalize_training_args(payload(**extra), CONSTANT)


def test_retry_cannot_bypass_8m_policy(online):
    queue = JobQueue(); identity = queue.submit("alice", "custom:training", "gpu", metadata={"task": "training", "args": {"plm_model": MODELS["ESM2-35M"]}})
    queue.cancel(identity, "alice")
    import asyncio
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.custom_job_retry(identity, "alice"))
    assert exc.value.status_code == 403
    assert len(queue.list("alice")) == 1


def test_training_waits_for_gpu_slot_and_outputs_are_owned(online, monkeypatch):
    monkeypatch.setenv("VENUS_QUEUE_GPU_SLOTS", "1")
    marker = online / "executed.txt"
    monkeypatch.setattr(api, "_build_exec_cmd", lambda *a: [sys.executable, "-c", "from pathlib import Path;Path(" + repr(str(marker)) + ").write_text('ok')"])
    queue = JobQueue(); holder = queue.submit("holder", "test", "gpu"); queue.bind(holder); assert queue.claim(holder)
    identity = api._create_task("training", "src/train.py", {"plm_model": MODELS["ESM2-8M"], "output_dir": "/ignored/online"}, "alice")
    try:
        assert queue.get(identity, "alice")["status"] == "queued"
        time.sleep(.3)
        assert not marker.exists()
        with pytest.raises(HTTPException): api._owned_task(identity, "bob")
        queue.finish(holder, "completed")
        for _ in range(120):
            if queue.get(identity)["status"] in {"completed", "failed"}: break
            time.sleep(.05)
        assert queue.get(identity)["status"] == "completed"
        assert marker.read_text() == "ok"
        job = queue.get(identity)
        assert Path(job["metadata"]["args"]["output_dir"]).is_relative_to(online / "results")
    finally:
        queue.cancel(identity); queue.finish(holder, "completed")


def test_online_application_exposes_training_routes(monkeypatch):
    """Exercise production registration, not a synthetic router-only app."""
    import importlib
    from types import SimpleNamespace
    import config
    real = config.get_config()
    server = SimpleNamespace(**{k: getattr(real.server, k) for k in real.server.model_fields}) if hasattr(real.server, 'model_fields') else SimpleNamespace(**vars(real.server))
    server.mode = 'online'
    import copy
    settings = copy.copy(real)
    object.__setattr__(settings, 'server', server)
    monkeypatch.setattr(config, 'get_config', lambda: settings)
    import backend.api_server as api_server
    application = importlib.reload(api_server).app
    client = TestClient(application)  # No lifespan: no background services.
    response = client.get('/api/custom-model/meta')
    assert response.status_code == 200
    assert 'training_plm_models' in response.json()
    assert '/api/custom-model/training/start' in application.openapi()['paths']


def test_online_custom_files_require_explicit_browser_ownership(online, monkeypatch):
    from backend.web_v2.job_records import JobRecords
    import hashlib
    root = online / 'files'; root.mkdir()
    public = root / 'server-config.json'; public.write_text('{"not_public": true}')
    own = root / 'own.csv'; own.write_text('aa_seq,label\nACDE,1\n')
    monkeypatch.setattr(api, '_ALLOWED_FILE_ROOTS', [root])
    app = FastAPI(); app.include_router(api.router)
    with TestClient(app) as alice, TestClient(app) as bob:
        alice.get('/api/custom-model/session')
        owner = hashlib.sha256(alice.cookies.get('vf_custom_jobs').encode()).hexdigest()
        JobRecords().grant(own, 'custom', owner)
        for client in (alice, bob):
            assert client.get('/api/custom-model/artifacts/download', params={'path': str(root)}).status_code == 404
        assert alice.get('/api/custom-model/artifacts/download', params={'path': str(own)}).status_code == 200
        assert bob.get('/api/custom-model/artifacts/download', params={'path': str(own)}).status_code == 404
        assert alice.get('/api/custom-model/artifacts/download', params={'path': str(public)}).status_code == 404
        secret = root / 'unclaimed.json'; secret.write_text('{}')
        assert alice.get('/api/custom-model/artifacts/download', params={'path': str(secret)}).status_code == 404
        for value in ('../private', '/etc/passwd', 'file:///tmp/private', str(online)):
            assert alice.post('/api/custom-model/training/preview', json={'args': payload(dataset_custom=value)}).status_code == 400


def test_training_csv_download_is_stable_and_owner_scoped(online, monkeypatch):
    from backend.web_v2.job_records import JobRecords
    import hashlib
    output = online / 'checkpoints'; output.mkdir()
    monkeypatch.setattr(api, '_ALLOWED_FILE_ROOTS', [online])
    app = FastAPI(); app.include_router(api.router)
    with TestClient(app) as alice, TestClient(app) as bob:
        alice.get('/api/custom-model/session')
        owner = hashlib.sha256(alice.cookies.get('vf_custom_jobs').encode()).hexdigest()
        JobRecords().grant(output, 'custom', owner)
        queue = JobQueue()
        identity = queue.submit(owner, 'custom:training', 'cpu', metadata={
            'task': 'training', 'args': {'output_dir': str(output), 'num_epochs': 1},
            'output_root': str(output),
        })
        files = queue.files(identity)
        files.mkdir(parents=True, exist_ok=True)
        (files / 'command.json').write_text(json.dumps({'command': ['training']}))
        (files / 'worker.log').write_text(
            'Epoch 0 Train Loss: 0.4\nTest Loss: 0.5\nTest Accuracy: 0.75\n'
        )
        queue.finish(identity, 'completed')
        stream_url = '/api/custom-model/jobs/' + identity + '/stream'
        frames = alice.post(stream_url)
        assert frames.status_code == 200
        events = [json.loads(frame.split('data: ', 1)[1]) for frame in frames.text.split('\n\n')
                  if frame.startswith('event: test_results\n')]
        url = events[-1]['csv_download_url']
        assert url == events[0]['csv_download_url']
        assert (output / 'test_results_metrics.csv').is_file()
        replay = alice.post(stream_url)
        assert replay.status_code == 200
        prefixes = ('event: training_metrics\n', 'event: test_results\n', 'event: done\n')
        assert [f for f in replay.text.split('\n\n') if f.startswith(prefixes)] == [
            f for f in frames.text.split('\n\n') if f.startswith(prefixes)
        ]
        assert bob.post(stream_url).status_code == 404
        response = alice.get(url)
        assert response.status_code == 200
        assert 'accuracy,0.750000' in response.text and 'loss,0.500000' in response.text
        assert bob.get(url).status_code == 404
