import json
from pathlib import Path

import pytest

from agent.harness.failure_summary import research_failure
from agent.harness.model_availability import model_service_blocker


def empty_response(path):
    path.write_text(json.dumps({
        "event": "response", "status": 200, "completed": True,
        "response_completion": {
            "streaming": True, "done": True, "malformed": False,
            "provider_error": False, "content_present": False,
            "tools_present": False, "finish_reasons": ["stop"],
        },
        "body": "private provider response",
    }) + "\n")
    return path


def test_native_empty_response_is_diagnostic_not_a_service_block(tmp_path):
    path = empty_response(tmp_path / "requests.jsonl")
    result = research_failure(path, RuntimeError("runtime_error"))
    assert result["code"] == "model_response_empty"
    assert "no answer" in result["message"]
    assert "private" not in json.dumps(result)
    assert model_service_blocker(path) is None


@pytest.mark.parametrize("error", [RuntimeError("Cannot load measured model artifact"), ValueError("runtime_error")])
def test_specific_research_errors_are_preserved_despite_earlier_empty_response(tmp_path, error):
    result = research_failure(empty_response(tmp_path / "requests.jsonl"), error)
    assert result == {"error_type": type(error).__name__, "message": str(error)}


def test_missing_audit_keeps_generic_native_error(tmp_path):
    assert research_failure(tmp_path / "missing.jsonl", RuntimeError("runtime_error")) == {
        "error_type": "RuntimeError", "message": "runtime_error",
    }


@pytest.mark.asyncio
async def test_scientist_episode_reports_empty_response_without_interrupting_goal(tmp_path, monkeypatch):
    import agent.harness.autonomous_research as module
    from agent.harness import Profile

    class Relay:
        calls = 0
        base_url = "http://unused"
        token = "fixture"

        def __init__(self, **kwargs):
            empty_response(Path(kwargs["output"]))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(module, "EvidenceGateway", Relay)
    monkeypatch.setattr(module, "audit_requests", lambda path: {"model_calls": 1, "passed": True})
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120},
    })
    store = module.create(tmp_path / "goal", profile, "Analyze supplied peptide lengths",
                          {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 240})

    async def consult(role, context):
        if role == "PI":
            return "Calculate supplied lengths."
        return json.dumps({"summary": "Compute lengths", "steps": [
            {"id": "compute", "title": "Compute lengths", "acceptance": "Save measured lengths"},
        ], "needs_input": []})

    async def executor(request, service, events):
        raise RuntimeError("runtime_error")

    state = await module.AutonomousResearch(
        store, "fixture", {}, consult=consult, executor=executor, tools=[]).run()
    assert len(state["cycles"]) == 2
    assert state["status"] != "interrupted"
    assert not state["progress"].get("execution_blocker")
    for cycle in state["cycles"]:
        assert "model returned no answer" in cycle["summary"]
    failure = json.loads((store.root / "episodes/cycle_000/failure.json").read_text())
    assert failure["code"] == "model_response_empty"
    assert not (store.root / "episodes/cycle_000/model_service.json").exists()
