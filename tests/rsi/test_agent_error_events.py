from agent.kimi_client import _map_event
from backend.web_v2.chat_api._stream_kimi import _apply_event


def test_step_interruption_is_visible_error_not_success():
    ev = _map_event({"type": "turn.step.interrupted", "payload": {"reason": "error", "message": "400 upstream request rejected", "turnId": 1}})
    assert ev.kind == "error" and ev.is_error
    state = {"history": [], "tool_executions": [{"status": "ok"}]}
    frames = _apply_event(state, ev)
    assert state["status"] == "error"
    assert state["history"][-1]["kind"] == "error"
    assert "upstream request rejected" in state["history"][-1]["content"]
    assert any("event: error" in f for f in frames)
    assert state["tool_executions"][0]["status"] == "ok"


def test_user_interruption_is_not_upstream_error():
    ev = _map_event({"type": "turn.step.interrupted", "payload": {"reason": "user", "turnId": 1}})
    assert ev.kind != "error"
