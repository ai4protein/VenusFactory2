"""Process-level regression checks for detached goals; no model network calls."""

import json
import sys
import time

import pytest

from agent.science_goal import GoalStore
from backend.web_v2.science_goals import ROOT, GoalConflict, ScienceGoalService


class FixtureDriver:
    id = "fixture.v1"
    title = "Fixture"
    cases = ("protein",)
    criteria = ("accepted_model", "fresh_session_reuse")

    def __init__(self, worker):
        self.worker = worker

    def prepare_command(self, service, folder, spec):
        if spec["parameters"] != {"case": "protein"}:
            raise ValueError("Bad fixture case")
        return [
            sys.executable,
            str(self.worker),
            "create",
            str(folder),
            str(service.prepared_profile(folder)),
            json.dumps(spec),
        ]

    def run_command(self, service, folder, action):
        return [sys.executable, str(self.worker), action, str(folder)]


@pytest.fixture
def service(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("""import asyncio,json,sys
from pathlib import Path
from agent.science_goal import GoalStore,GoalSpec,run_goal
from agent.harness import Profile
from agent.harness.contracts import file_hash
folder=Path(sys.argv[2])
if sys.argv[1]=="create":
    spec=json.loads(sys.argv[4])
    GoalStore.create(folder,GoalSpec(spec["objective"],("accepted_model","fresh_session_reuse"),**spec["budget"]),Profile.from_dict(json.loads(Path(sys.argv[3]).read_text())))
else:
    store=GoalStore(folder)
    if sys.argv[1]=="resume": store.resume()
    async def cycle(index,profile,state,context):
        (folder/f"started_{index}").write_text(context)
        while not (folder/f"release_{index}").exists(): await asyncio.sleep(.02)
        count=state["progress"].get("count",0)+1
        (folder/f"evidence_{count}").write_text("verified")
        return {"status":"completed","summary":"episode checkpoint", "usage":{"model_calls":1,"tool_calls":1,"wall_seconds":.1},"progress":{"count":count}}
    def evaluate(state):
        result={}
        for i,name in enumerate(["accepted_model","fresh_session_reuse"],1):
            path=folder/f"evidence_{i}"
            ok=path.exists()
            result[name]={"passed":ok,"evidence":[{"path":path.name,"sha256":file_hash(path)}] if ok else []}
        return {"criteria":result}
    asyncio.run(run_goal(store,cycle,evaluate,pause_poll_seconds=.02))
""")
    svc = ScienceGoalService(
        ROOT, state_root=tmp_path / "goals", python=sys.executable, drivers={"fixture.v1": FixtureDriver(worker)}
    )
    svc.admission = lambda *args: None
    yield svc
    for proc in svc.children.values():
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)


def request():
    return {
        "driver": "fixture.v1",
        "parameters": {"case": "protein"},
        "objective": "Develop and reuse a protein model",
        "budget": {"max_cycles": 6, "model_calls": 96, "tool_calls": 48, "wall_seconds": 3600},
    }


def wait_for(fn):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("Detached fixture did not reach checkpoint")


def test_restart_pause_resume_keeps_goal_and_budget(service):
    goal = service.create("owner-a", request(), "stable-request-01")
    folder, _ = service.paths("owner-a", goal["id"])
    wait_for(lambda: (folder / "started_0").exists())
    assert service.create("owner-a", request(), "stable-request-01")["id"] == goal["id"]
    with pytest.raises(GoalConflict):
        service.create("owner-a", request(), "another-request-01")
    # Lose the supervisor and PID metadata: the inherited worker lease still owns the goal.
    recovered = ScienceGoalService(ROOT, state_root=service.state_root, drivers=service.drivers)
    recovered.admission = lambda *args: None
    (folder / "worker.json").unlink()
    assert recovered.public("owner-a", goal["id"])["worker_alive"]
    recovered.control("owner-a", goal["id"], "pause")
    wait_for(lambda: not recovered.live(folder))
    paused = GoalStore(folder).read()
    assert paused["status"] == "paused" and len(paused["cycles"]) == 1
    assert paused["usage"]["model_calls"] > 0  # Retain unknown interrupted usage.
    recovered.control("owner-a", goal["id"], "resume")
    service.children.update(recovered.children)
    wait_for(lambda: (folder / "started_1").exists())
    (folder / "release_1").touch()
    wait_for(lambda: (folder / "started_2").exists())
    midway = recovered.public("owner-a", goal["id"])
    assert midway["status"] == "active"  # One final report is not whole-goal completion.
    assert midway["criteria"][0]["passed"] and not midway["criteria"][1]["passed"]
    (folder / "release_2").touch()
    wait_for(lambda: GoalStore(folder).read()["status"] == "completed")
    final = recovered.public("owner-a", goal["id"])
    assert all(c["passed"] and c["evidence_count"] for c in final["criteria"])
    assert final["usage"]["model_calls"] >= paused["usage"]["model_calls"] + 2


def test_dead_worker_is_interrupted_not_relaunched(service):
    goal = service.create("a", request(), "request-crash-01")
    folder, _ = service.paths("a", goal["id"])
    wait_for(lambda: (folder / "started_0").exists())
    reserved = GoalStore(folder).read()["usage"].copy()
    proc = service.children[str(folder)]
    proc.kill()
    proc.wait()
    observed = service.public("a", goal["id"])
    assert observed["status"] == "interrupted" and not observed["worker_alive"]
    assert observed["usage"] == reserved and observed["cycles"][0]["status"] == "interrupted"
    assert not (folder / "started_1").exists()


def test_cross_session_ownership_and_request_conflict(service):
    goal = service.create("a", request(), "request-owner-01")
    with pytest.raises(FileNotFoundError):
        service.control("b", goal["id"], "cancel")
    changed = request()
    changed["objective"] = "A different protein task"
    with pytest.raises(GoalConflict):
        service.create("a", changed, "request-owner-01")
    service.control("a", goal["id"], "cancel")
    wait_for(lambda: not service.has_running_goal("a"))
    assert service.list("a")[0]["status"] == "cancelled"


def test_budget_validation_no_child_started(service):
    for key, value in [("model_calls", True), ("wall_seconds", 7201), ("max_cycles", 0)]:
        spec = request()
        spec["budget"][key] = value
        with pytest.raises(ValueError):
            service.create("a", spec, "invalid-budget-01")
    assert not service.children


def test_corrupt_profile_catalog_is_unavailable(service, tmp_path):
    service.profile_path = tmp_path / "bad.json"
    service.profile_path.write_text("not-json")
    assert service.catalog()["available"] is False


@pytest.fixture
def api_client(service, monkeypatch, tmp_path):
    import importlib

    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    monkeypatch.setenv("VENUSFACTORY_SESSION_DB_PATH", str(tmp_path / "sessions.sqlite"))
    api = importlib.import_module('backend.web_v2.chat_api.science_goals')
    shared = importlib.import_module('backend.web_v2.chat_api._shared')
    sessions = importlib.import_module('backend.web_v2.chat_api.sessions')
    messages = importlib.import_module('backend.web_v2.chat_api.messages')
    state = {
        "session_id": "web-owner",
        "owner_key": "owner",
        "session_token_hash": shared._hash_session_token("valid-test-token"),
        "history": [],
    }

    async def get(sid):
        if sid != "web-owner":
            raise HTTPException(404, "Session not found")
        return state

    class Store:
        async def save(self, sid):
            pass

        async def get(self, sid):
            return await get(sid)

    monkeypatch.setattr(api, "service", service)
    monkeypatch.setattr(api, "_runtime_mode", lambda: "local")
    monkeypatch.setattr(shared, "_runtime_mode", lambda: "online")
    monkeypatch.setattr(shared, "_session_owner_key_for_request", lambda req: "owner")
    monkeypatch.setattr(shared, "_build_owner_fingerprint", lambda req: "owner")
    for module in [api, sessions, messages]:
        monkeypatch.setattr(module, "_get_session_or_404", get)
    monkeypatch.setattr(api, "_session_store", Store())
    monkeypatch.setattr(sessions, "_session_store", Store())
    monkeypatch.setattr(sessions, "goal_service", service)
    monkeypatch.setattr(messages, "goal_service", service)
    app = FastAPI()
    for router in [api.router, sessions.router, messages.router]:
        app.include_router(router, prefix="/api/chat")
    with TestClient(app) as client:
        yield client, api, state


def test_api_auth_idempotency_and_active_session_guards(api_client, service):
    client, _, state = api_client
    url = "/api/chat/sessions/web-owner/science-goals"
    payload = {**request(), "request_id": "api-request-01"}
    assert client.post(url, json=payload).status_code == 401
    assert client.post(url, json=payload, headers={"x-session-access-token": "wrong"}).status_code == 403
    assert not service.children
    client.headers["x-session-access-token"] = "valid-test-token"
    started = client.post(url, json=payload)
    assert started.status_code == 200, started.text
    assert client.post(url, json=payload).json()["id"] == started.json()["id"]
    assert len(state["history"]) == 1 and state["chat_mode"] == "science_expert"
    assert client.get(url).headers["cache-control"] == "no-store"
    assert client.get(url).json()["goals"][0]["worker_alive"]
    assert client.delete("/api/chat/sessions/web-owner").status_code == 409
    assert client.post("/api/chat/sessions/web-owner/messages/stream", json={"text": "Hi"}).status_code == 409
    gid = started.json()["id"]
    assert client.post(url + f"/{gid}/pause").json()["status"] == "paused"
    assert client.post(url + f"/{gid}/cancel").json()["status"] == "cancelled"


def test_api_online_restrictions_and_untrusted_fields_are_rejected(api_client, service, monkeypatch):
    client, api, _ = api_client
    client.headers["x-session-access-token"] = "valid-test-token"
    url = "/api/chat/sessions/web-owner/science-goals"
    payload = {**request(), "request_id": "api-request-02", "command": "arbitrary shell"}
    assert client.post(url, json=payload).status_code == 422
    monkeypatch.setattr(api, "_runtime_mode", lambda: "online")
    catalog = client.get("/api/chat/science-goals/catalog").json()
    assert catalog["available"] is False  # Fixture driver is not online-admitted.
    assert catalog["maximum_budget"]["max_cycles"] == 2
    assert catalog["maximum_budget"]["wall_seconds"] == 900
    payload.pop("command")
    assert client.post(url, json=payload).status_code == 400
    assert not service.children


def test_selected_profile_is_frozen_across_catalog_change_and_resume(service, tmp_path, monkeypatch):
    from dataclasses import replace

    from agent.harness import ModelBinding
    from agent.harness.contracts import digest

    base = service.profile()
    alternative = ModelBinding("other", "model-b", "https://other.example/v1", credential_env="TEST_GOAL_B_KEY")
    path = tmp_path / "models.json"
    config = {
        "models": [
            {"id": "other-model", "label": "Other", "binding": replace(base, model=alternative).public()["model"]}
        ]
    }
    path.write_text(json.dumps(config))
    service.catalog_path = path
    monkeypatch.setenv("TEST_GOAL_B_KEY", "selected-private-key")
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-inherit")
    spec = {**request(), "execution": {"harness": "role-graph", "model_id": "other-model"}}
    goal = service.create("frozen-owner", spec, "frozen-execution-01")
    folder, _ = service.paths("frozen-owner", goal["id"])
    wait_for(lambda: (folder / "started_0").exists())
    frozen = service.frozen_profile(folder)
    assert frozen.harness == "role-graph" and frozen.model == alternative
    assert goal["execution"]["profile_sha256"] == digest(frozen.public())
    env = service.environment(frozen)
    assert env["TEST_GOAL_B_KEY"] == "selected-private-key"
    assert "UNRELATED_SECRET" not in env
    assert "selected-private-key" not in json.dumps(goal)
    service.control("frozen-owner", goal["id"], "pause")
    wait_for(lambda: not service.live(folder))
    # Removing the selected model from today's catalog must not rewrite yesterday's goal.
    path.write_text('{"models": []}')
    assert service.create("frozen-owner", spec, "frozen-execution-01")["id"] == goal["id"]
    service.control("frozen-owner", goal["id"], "resume")
    wait_for(lambda: (folder / "started_1").exists())
    assert service.frozen_profile(folder) == frozen
    service.control("frozen-owner", goal["id"], "cancel")
    wait_for(lambda: not service.live(folder))


def test_execution_selection_changes_request_identity(service):
    original = request()
    goal = service.create("identity-owner", original, "same-request-01")
    changed = {**original, "execution": {"harness": "role-graph", "model_id": "dmx-glm51"}}
    with pytest.raises(GoalConflict):
        service.create("identity-owner", changed, "same-request-01")
    service.control("identity-owner", goal["id"], "cancel")


def test_api_rejects_endpoint_and_credential_injection(api_client, service):
    client, _, _ = api_client
    client.headers["x-session-access-token"] = "valid-test-token"
    payload = {
        **request(),
        "request_id": "injected-config-01",
        "execution": {
            "harness": "langgraph",
            "model_id": "dmx-glm51",
            "base_url": "https://elsewhere.example",
            "api_key": "inline-key",
        },
    }
    assert client.post("/api/chat/sessions/web-owner/science-goals", json=payload).status_code == 422
    assert not service.children


def test_unsupported_harness_is_rejected_before_goal_preparation(service, monkeypatch):
    from types import MethodType

    service.admission = MethodType(ScienceGoalService.admission, service)
    service.runtime_paths = lambda: {"python_bin": sys.executable, "kimi_bin": sys.executable}
    monkeypatch.setenv(service.profile().model.credential_env, "fixture-key-not-sent")
    default_model = service.execution_catalog().default_model
    for harness in ["kimi-code", "codex"]:
        spec = {**request(), "execution": {"harness": harness, "model_id": default_model}}
        with pytest.raises(ValueError, match="unsupported|disabled"):
            service.create("rejected-owner", spec, "unsupported-" + harness)
    assert not service.children
    assert service.list("rejected-owner") == []


def test_paper_campaign_api_budget_matches_catalog_without_relaxing_other_drivers(api_client, service, monkeypatch):
    from backend.web_v2.science_goals import PaperCampaignDriver

    client, _, _ = api_client
    client.headers["x-session-access-token"] = "valid-test-token"
    url = "/api/chat/sessions/web-owner/science-goals"
    oversized = {**request(), "request_id": "oversize-fixture-01"}
    oversized["budget"]["max_cycles"] = 16
    assert client.post(url, json=oversized).status_code == 400
    assert not service.children
    driver = PaperCampaignDriver()
    received = []

    def accept(sid, spec, request_id):
        received.append(spec)
        return {"id": "synthetic-campaign", "status": "preparing"}

    monkeypatch.setattr(service, "create", accept)
    for suite, detail in driver.case_details(None).items():
        payload = {
            "request_id": "paper-api-" + suite,
            "driver": driver.id,
            "parameters": {"case": suite},
            "objective": "Reproduce all declared paper scopes",
            "budget": detail["suggested_budget"],
        }
        response = client.post(url, json=payload)
        assert response.status_code == 200, response.text
        assert received[-1]["budget"] == payload["budget"]
    assert len(received) == len(driver.cases)
    payload["budget"]["max_cycles"] = 33
    assert client.post(url, json=payload).status_code == 422
    assert len(received) == len(driver.cases)


def test_project_download_requires_session_access_and_retains_partial_status(api_client, service):
    import io, zipfile
    client, api, state = api_client
    goal = service.create('web-owner', request(), 'project-download-test')
    folder, _ = service.paths('web-owner', goal['id'])
    wait_for(lambda: (folder / 'started_0').exists())
    service.control('web-owner', goal['id'], 'pause')
    wait_for(lambda: not service.live(folder))
    workspace = folder / 'workspace'; workspace.mkdir(exist_ok=True)
    (workspace / 'README.md').write_text('Incomplete project')
    url = '/api/chat/sessions/web-owner/science-goals/' + goal['id'] + '/project'
    assert client.get(url).status_code in {401,403}
    headers = {'x-session-access-token':'valid-test-token'}
    response = client.get(url, headers=headers)
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        assert 'project/README.md' in z.namelist()
        assert 'paused' in z.read('REPORT.md').decode()
    assert client.get(url+'?format=report', headers=headers).status_code == 200
