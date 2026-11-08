import json
from pathlib import Path
import pytest
from agent.harness.model_availability import model_service_blocker


def receipts(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


@pytest.mark.parametrize("status,signals,code", [
    (401, [], "authentication"), (402, [], "quota"), (403, ["quota"], "quota"),
    (403, [], "access"), (429, [], "rate_limit"), (404, [], "configuration"),
])
def test_terminal_service_errors_are_classified_without_raw_payload(tmp_path, status, signals, code):
    path = receipts(tmp_path / "requests.jsonl", [
        {"event": "request", "call": 1, "text": "private prompt"},
        {"event": "upstream_rejection", "call": 1, "status": status,
         "reported_signals": signals, "error_body": "private provider response"},
        {"event": "response", "call": 1, "status": status, "completed": False},
    ])
    result = model_service_blocker(path)
    assert result["code"] == code and result["http_status"] == status
    assert "private" not in json.dumps(result)


def test_successful_sdk_retry_supersedes_the_earlier_rejection(tmp_path):
    path = receipts(tmp_path / "requests.jsonl", [
        {"event": "response", "call": 1, "status": 429, "completed": False},
        {"event": "response", "call": 2, "status": 200, "completed": True},
    ])
    assert model_service_blocker(path) is None


@pytest.mark.parametrize("row", [
    {"event": "tool_result", "status": 403, "completed": False},
    {"event": "text", "text": "HTTP 403 quota exceeded"},
    {"event": "response", "status": {}, "completed": False},
    {"event": "response", "status": 503, "completed": False},
])
def test_tool_prose_and_transient_or_malformed_records_do_not_suspend(tmp_path, row):
    assert model_service_blocker(receipts(tmp_path / "requests.jsonl", [row])) is None


def test_missing_or_partial_audit_is_not_a_provider_rejection(tmp_path):
    path = tmp_path / "requests.jsonl"
    assert model_service_blocker(path) is None
    path.write_text('{"incomplete":\n')
    assert model_service_blocker(path) is None

@pytest.mark.asyncio
async def test_autonomous_provider_quota_stops_after_one_episode(tmp_path, monkeypatch):
    import agent.harness.autonomous_research as module
    from agent.harness import Profile
    class Relay:
        calls = 0
        base_url = "http://unused"
        token = "fixture"
        def __init__(self, **kwargs):
            receipts(Path(kwargs["output"]), [
                {"event": "upstream_rejection", "call": 1, "status": 403,
                 "reported_signals": ["quota"]},
                {"event": "response", "call": 1, "status": 403, "completed": False},
            ])
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    monkeypatch.setattr(module, "EvidenceGateway", Relay)
    monkeypatch.setattr(module, "audit_requests", lambda path: {"model_calls": 1, "passed": False})
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120},
    })
    store = module.create(tmp_path / "goal", profile, "Analyze these supplied measurements",
                          dict(max_cycles=4, model_calls=48, tool_calls=48, wall_seconds=480))
    async def consult(role, context):
        raise RuntimeError("native runtime_error")
    runner = module.AutonomousResearch(store, "fixture", {}, consult=consult, tools=[])
    state = await runner.run()
    assert state["status"] == "interrupted" and len(state["cycles"]) == 1
    assert state["usage"]["model_calls"] == 1 and state["usage"]["tool_calls"] == 0
    assert state["progress"]["execution_blocker"]["code"] == "quota"
    assert (store.root / "episodes/cycle_000/model_service.json").is_file()


@pytest.mark.asyncio
async def test_late_gateway_rejection_overrides_claimed_completion(tmp_path, monkeypatch):
    import agent.harness.autonomous_research as module
    from agent.harness import Profile
    class Relay:
        calls = 0
        base_url = "http://unused"
        token = "fixture"
        def __init__(self, **kwargs):
            receipts(Path(kwargs["output"]), [
                {"event": "response", "call": 1, "status": 403, "completed": False},
            ])
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    monkeypatch.setattr(module, "EvidenceGateway", Relay)
    audits = iter([True, False])
    monkeypatch.setattr(module, "audit_requests",
                        lambda path: {"model_calls": 4, "passed": next(audits)})
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash",
                  "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120},
    })
    store = module.create(tmp_path / "goal", profile, "Analyze supplied peptide lengths",
                          dict(max_cycles=4, model_calls=48, tool_calls=48, wall_seconds=480))
    async def consult(role, context):
        if role == "PI": return "Calculate the supplied lengths."
        if role == "CB":
            return json.dumps({"summary": "Compute lengths", "steps": [
                {"id": "compute", "title": "Compute lengths", "acceptance": "Save measured lengths"}
            ], "needs_input": []})
        return json.dumps({"decision": "ready_for_host_validation",
                           "summary": "The result is present", "issues": []})
    async def executor(request, service, events):
        await service.write_research_file("result.json", '{"lengths":[4,2]}')
        await service.submit_research_checkpoint("Calculated lengths", ["compute"], ["result.json"], [])
        events.extend([{"kind": "tool_call_start"},
                       {"kind": "completed", "text": "Calculated lengths", "event_audit": {"passed": True}}])
    state = await module.AutonomousResearch(
        store, "fixture", {}, consult=consult, executor=executor, tools=[]).run()
    assert state["status"] == "interrupted" and len(state["cycles"]) == 1
    assert state["progress"]["execution_blocker"]["code"] == "access"
    assert "reviewed_episode" not in state["progress"]
