import json
from contextlib import aclosing

import httpx
import pytest

from agent.harness.kimi_web_events import KimiTranscript, TranscriptGap
from agent.harness.kimi_web_client import KimiWebClient


def batch(seq, *ops):
    return {"seq": seq, "ops": list(ops)}


def frame(value):
    return {"op": "frame.upsert", "turnId": "t0", "stepId": "t0.1", "frame": value}


def text_frame(text=""):
    return {"kind": "text", "frameId": "f1", "role": "assistant", "text": text}


def append(text, offset):
    return {"op": "append", "target": {"type": "frame", "turnId": "t0", "stepId": "t0.1", "frameId": "f1"}, "offset": offset, "text": text}


def test_stream_and_final_snapshot_emit_text_once_with_unicode():
    state = KimiTranscript()
    events = state.consume(batch(1, frame(text_frame())))
    events += state.consume(batch(2, append("Protein \U0001f9ec", 0)))
    events += state.consume(batch(3, append(" works.", 10)))
    events += state.consume(batch(4, frame(text_frame("Protein \U0001f9ec works."))))
    assert "".join(e.text for e in events) == "Protein \U0001f9ec works."
    assert state.consume(batch(3, append(" works.", 10))) == []


def test_reconnect_snapshot_adds_only_missing_text_before_completion():
    state = KimiTranscript()
    state.consume(batch(1, frame(text_frame("First "))))
    snapshot = {"items": [{"kind": "turn", "turnId": "t0", "state": "completed", "steps": [
        {"stepId": "t0.1", "frames": [text_frame("First second.")]}]}]}
    events = state.reset(snapshot, 9)
    assert [e.text for e in events if e.kind == "text"] == ["second."]
    assert events[-1].kind == "turn_ended" and not events[-1].is_error
    assert state.reset(snapshot, 9) == []


@pytest.mark.parametrize("value", ["Different", "Firs"])
def test_already_emitted_text_cannot_be_silently_rewritten(value):
    state = KimiTranscript(); state.consume(batch(1, frame(text_frame("First"))))
    with pytest.raises(ValueError, match="rewrote"):
        state.consume(batch(2, frame(text_frame(value))))


def test_missing_frames_require_replay_instead_of_silent_skips():
    state = KimiTranscript(); state.consume(batch(1, frame(text_frame())))
    with pytest.raises(TranscriptGap): state.consume(batch(3, append("missing", 0)))
    with pytest.raises(ValueError, match="offset"): state.consume(batch(2, append("wrong", 9)))


@pytest.mark.parametrize("status", ["done", "error"])
def test_tool_approval_and_final_replay_do_not_duplicate_calls(status):
    state = KimiTranscript()
    tool = {"kind": "tool", "frameId": "tool1", "toolCallId": "c1", "name": "mcp__venusfactory__probe",
            "state": "running", "inputText": '{"sequence":"ACD"}'}
    events = state.consume(batch(1, frame(tool)))
    events += state.consume(batch(2, frame({**tool, "approvalId": "a1"})))
    final = {**tool, "state": status, "input": {"sequence": "ACD"}, "output": '{"length":3}'}
    events += state.consume(batch(3, frame(final)))
    events += state.consume(batch(4, frame(final)))
    assert [e.kind for e in events] == ["tool_call_start", "tool_result"]
    assert events[0].tool_args == {"sequence": "ACD"}
    assert events[1].tool_output == '{"length":3}'
    assert events[1].is_error == (status == "error")


def test_internal_markers_and_user_text_are_not_assistant_output():
    state = KimiTranscript()
    assert state.consume(batch(1, {"op": "marker.upsert", "marker": {"text": "private context"}},
        frame({**text_frame("user input"), "role": "user"}))) == []


def test_resolved_interactions_cannot_be_reactivated_by_late_projection():
    state = KimiTranscript()
    item = {"interaction_id": "a1", "kind": "approval", "status": "pending", "request": {"tool_name": "probe"}}
    state.interaction(item)
    assert state.interactions["a1"]["tool_name"] == "probe"
    state.interaction({**item, "status": "ended"})
    state.interaction({"interactionId": "a1", "interactionKind": "approval", "state": "pending", "request": {"toolName": "probe"}})
    assert not state.interactions


@pytest.mark.asyncio
async def test_auth_and_interaction_responses_use_unified_endpoint():
    requests = []
    def endpoint(request):
        assert request.headers["authorization"] == "Bearer fixture"
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"code": 0, "data": {"outcome": "accepted"}})
    client = KimiWebClient("http://127.0.0.1:12345", token="fixture", transport=httpx.MockTransport(endpoint))
    try:
        client.transcript("s1").interaction({"interaction_id": "a1", "kind": "approval", "status": "pending", "request": {"tool_name": "probe"}})
        assert len(await client.list_pending_approvals("s1")) == 1
        await client.decide_approval("s1", "a1", decision="approved")
        await client.answer_question("s1", "q1", answers={"field": {"kind": "other", "text": "human answer"}})
        await client.dismiss_question("s1", "q2")
        assert not await client.list_pending_approvals("s1")
        assert [p for p, _ in requests] == ["/api/v1/sessions/s1/interactions/" + n for n in ["a1", "q1", "q2"]]
        assert [body["kind"] for _, body in requests] == ["approval", "question", "question"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_sequence_gap_replays_missing_batches_without_duplicate_text():
    def endpoint(request):
        assert request.url.params["since_seq"] == "1"
        return httpx.Response(200, json={"code": 0, "data": {"complete": True, "latest_seq": 3,
            "batches": [batch(2, append("B", 1)), batch(3, append("C", 2))]}})
    client = KimiWebClient("http://127.0.0.1:12345", token="fixture", transport=httpx.MockTransport(endpoint))
    try:
        state = client.transcript("s1");state.consume(batch(1, frame(text_frame("A"))))
        assert "".join(e.text for e in await client.catch_up("s1")) == "BC"
        assert state.consume(batch(3, append("C", 2))) == []
    finally: await client.aclose()


@pytest.mark.asyncio
async def test_lost_journal_recovers_from_current_snapshot():
    def endpoint(request):
        if request.url.path.endswith("/ops"):
            data = {"complete": False, "latest_seq": 8, "batches": []}
        else:
            data = {"seq": 8, "has_more": False, "items": [{"kind": "turn", "turnId": "t0", "state": "completed",
                "steps": [{"stepId": "t0.1", "frames": [text_frame("ABC")]}]}]}
        return httpx.Response(200, json={"code": 0, "data": data})
    client = KimiWebClient("http://127.0.0.1:12345", token="fixture", transport=httpx.MockTransport(endpoint))
    try:
        client.transcript("s1").consume(batch(1, frame(text_frame("A"))))
        events = await client.catch_up("s1")
        assert [e.text for e in events if e.kind == "text"] == ["BC"]
        assert events[-1].kind == "turn_ended"
    finally: await client.aclose()


@pytest.mark.asyncio
async def test_status_verifies_live_agent_instead_of_saved_profile(monkeypatch):
    client = KimiWebClient("http://127.0.0.1:12345", token="fixture")
    async def frames(sid, since=0):
        yield {"type": "agent", "session_id": "other", "agent_id": "main", "model": "wrong"}
        yield {"type": "agent", "session_id": sid, "agent_id": "main", "status": "idle", "model": "research",
               "thinking": "medium", "permission_mode": "manual", "plan_mode": False, "swarm_mode": False}
    monkeypatch.setattr(client, "_wire_frames", frames)
    try:
        status = await client.get_status("s1")
        assert status["model"] == "research" and status["permission"] == "manual"
        assert status["plan_mode"] is False and status["swarm_mode"] is False
    finally: await client.aclose()


def test_tool_error_without_output_preserves_error_detail():
    state = KimiTranscript()
    events = state.consume(batch(1, frame({"kind": "tool", "frameId": "f1", "toolCallId": "c1",
        "name": "probe", "state": "error", "input": {}, "output": None, "error": "invalid sequence"})))
    assert events[-1].is_error and events[-1].tool_output == "invalid sequence"


@pytest.mark.parametrize("url", ["https://127.0.0.1:1", "http://localhost:1", "http://example.org:1",
    "http://user@127.0.0.1:1", "http://127.0.0.1:1/path", "http://127.0.0.1:1/#fragment"])
def test_auth_token_is_never_sent_to_nonisolated_endpoint(url):
    with pytest.raises(ValueError, match="loopback"):
        KimiWebClient(url, token="fixture")


@pytest.mark.asyncio
async def test_usage_is_a_cumulative_snapshot_not_sum_of_stream_events():
    def endpoint(request):
        assert request.url.path == "/api/v1/sessions/s1"
        return httpx.Response(200, json={"code": 0, "data": {"usage": {
            "input_tokens": 100, "output_tokens": 40, "context_tokens": 90, "turn_count": 1}}})
    async with aclosing(KimiWebClient("http://127.0.0.1:12345", token="fixture", transport=httpx.MockTransport(endpoint))) as client:
        assert await client.usage_rollup("s1") == await client.usage_rollup("s1") == {
            "input_tokens": 100, "output_tokens": 40, "context_tokens": 90, "turn_count": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [{"input_tokens": True}, {"output_tokens": -1},
    {"input_tokens": 0.5}, {"total_cost_usd": float("inf")}])
async def test_invalid_usage_is_not_silently_reported(usage, monkeypatch):
    from agent.kimi_client import KimiAPIError
    async with aclosing(KimiWebClient("http://127.0.0.1:12345", token="fixture")) as client:
        async def session(_): return {"usage": usage}
        monkeypatch.setattr(client, "get_session", session)
        with pytest.raises(KimiAPIError, match="usage"):
            await client.usage_rollup("s1")
