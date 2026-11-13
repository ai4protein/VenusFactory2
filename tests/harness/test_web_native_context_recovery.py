"""Complete goal routing through the real Web worker with local native fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import autonomous_research as ar, web_native_worker as worker
from agent.harness.adapters import _dsh_observation
from agent.harness.contracts import Profile


@pytest.mark.asyncio
@pytest.mark.parametrize("role_failure,followup", [(None, False), ("PI", False), ("SC", False), (None, True)])
async def test_goal_binds_native_overflow_without_forgiving_role_failure(tmp_path, monkeypatch, role_failure, followup):
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {})
    relays = []
    turns = []
    call_budget = 16 if followup else 12

    class Relay:
        base_url = "http://unused"
        token = "fixture"
        def __init__(self, **kwargs):
            self.policy = kwargs["policy"]
            self.path = Path(kwargs["output"])
            self.path.touch()
            self.calls = 0
            relays.append(self)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def record(self, row):
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        def model_call(self, status=200):
            self.calls += 1
            self.record({"event": "request", "call": self.calls})
            if status != 200:
                self.record({"event": "upstream_rejection", "call": self.calls, "status": status,
                             "reported_signals": ["context_limit"]})
            self.record({"event": "response", "call": self.calls, "status": status,
                         "completed": status == 200, "error_type": None})

    monkeypatch.setattr(ar, "EvidenceGateway", Relay)
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "fixture", "model": "fixture", "base_url": "http://127.0.0.1:1"},
        "budget": {"model_calls": call_budget, "tool_calls": 4, "wall_seconds": 60,
                   "output_tokens": 512, "thinking": "disabled"},
    })
    store = ar.create(tmp_path / "goal", profile, "Inspect the local project files and retain evidence",
                      {"max_cycles": 1, "model_calls": call_budget, "tool_calls": 4, "wall_seconds": 60})

    async def consult(role, context):
        relays[-1].model_call(400 if role == role_failure else 200)
        if role == "PI":
            return "Inspect the supplied files."
        if role == "CB":
            return json.dumps({"summary": "Inspect local files", "deliverable": "analysis", "needs_input": [],
                "steps": [{"id": "inspect", "title": "Inspect files", "acceptance": "Retain checked observations",
                           "verification": "execution"}]})
        return json.dumps({"decision": "needs_revision", "summary": "No final report exists yet",
                           "issues": ["A checked report is still required"], "evidence_ids": [],
                           "next_actions": ["Produce a report from the actual observations."]})

    class Adapter:
        async def run(self, ctx):
            relay = relays[-1]
            turn = len(turns)
            turns.append(ctx)
            assert ctx.relay.calls == (2 if turn == 0 else 6)
            sid = "native-goal-session-" + str(turn)
            trace = ctx.task.output / "events.jsonl"
            trace.touch()
            def record(event):
                notification = {"sessionId": sid, "event": event}
                for name, value in [("runtime_events.jsonl", event),
                                    ("runtime_notifications.jsonl", notification)]:
                    with (ctx.task.output / name).open("a") as stream:
                        stream.write(json.dumps(value) + "\n")
                with trace.open("a") as stream:
                    for row in _dsh_observation(event, sid, ctx.relay.calls, notification):
                        stream.write(json.dumps(row) + "\n")
            relay.model_call(400)
            record({"type": "compaction/start", "data": {"compactionId": "c"}})
            relay.model_call()
            record({"type": "compaction/end", "data": {"compactionId": "c"}})
            relay.model_call()
            connection = json.loads((ctx.task.capability / "connection.json").read_text())
            record({"type": "tool/call", "data": {"callId": "t", "name": "mcp__venusfactory__list_session_files",
                                                 "arguments": {}}})
            async with httpx.AsyncClient(trust_env=False) as client:
                reply = await client.post(connection["url"],
                    headers={"Authorization": "Bearer " + connection["token"]},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "list_session_files", "arguments": {}}})
            output = reply.json()["result"]
            assert not output["isError"]
            record({"type": "tool/result", "data": {"message": {"content": [{
                "type": "tool-result", "toolCallId": "t", "isError": False,
                "content": output["content"]}]}}})
            # Spend the admitted native phase, preserving the coordinator reserve.
            # No checkpoint is invented; the scientific result remains unfinished.
            ceiling = 6 if followup and turn == 0 else call_budget - 3
            while relay.calls < ceiling:
                relay.model_call()
            record({"type": "turn/end", "data": {"reason": {"kind": "completed"}}})
            return {"status": "completed", "session_id": sid, "tool_calls": 1, "text": "Inspected local files",
                    "sandbox_stopped": True}

    monkeypatch.setattr(worker, "builtin_registry",
                        lambda: SimpleNamespace(resolve=lambda *args: (Adapter(), {})))
    runner = ar.AutonomousResearch(store, "fixture", {}, consult=consult, tools=[])
    state = await runner.run()
    episodes = list((store.root / "episodes").glob("*/request_audit.json"))
    assert len(episodes) == 1, state
    audit = json.loads(episodes[0].read_text())
    assert audit["passed"] is (role_failure is None), audit
    assert audit["model_calls"] == call_budget - 2
    assert audit["native_context_recovery"]["valid"] is (role_failure is None)
    recovered = audit["native_context_recovery"]["recovered_calls"]
    assert [row["failed_call"] for row in recovered] == (
        ([3, 7] if followup else [3]) if role_failure is None else [])
    binding = json.loads((episodes[0].parent / "native/run/native_binding.json").read_text())
    assert (binding["calls_before"], binding["calls_after"]) == (2, 6 if followup else 9)
    assert binding["session_id"] == "native-goal-session-0"
    if followup:
        second = json.loads((episodes[0].parent / "execution_retry/native/run/native_binding.json").read_text())
        assert (second["calls_before"], second["calls_after"]) == (6, 13)
        assert second["session_id"] == "native-goal-session-1"
    assert state["progress"]["research_completed"] is False


def test_missing_host_counter_is_unknown_for_ordinary_agent():
    relay = worker._ObservedRelay("http://local", "fixture", None)
    assert relay.calls is None
