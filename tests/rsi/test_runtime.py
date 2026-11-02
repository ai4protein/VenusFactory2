import asyncio
import json

import pytest

from agent.kimi_client import KimiEvent
from agent.rsi.engine.runtime import run_task, safe_approval


class FakeClient:
    def __init__(self, events, stall=False):
        self.events, self.stall, self.submitted, self.aborted = events, stall, False, False

    async def create_session(self, **kw):
        return "sid"

    async def get_status(self, sid):
        return {"status": "idle", "model": "m", "permission": "manual", "plan_mode": False, "swarm_mode": False}

    async def update_profile(self, sid, **kwargs):
        self.profile = kwargs

    async def list_mcp_servers(self):
        return [{"id": "venusfactory", "status": "connected"}]

    @property
    def _http(self):
        return self

    async def get(self, *args, **kwargs):
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": {"tools": [{"name": "mcp__venusfactory__sequence_properties"}]}}

    async def submit_prompt(self, sid, text):
        self.submitted_text = text
        self.submitted = True
        return {"id": "prompt"}

    async def list_pending_approvals(self, sid):
        return []

    async def list_pending_questions(self, sid):
        return []

    async def abort_prompt(self, sid, pid):
        self.aborted = True

    async def stream_session_events(self, sid):
        yield KimiEvent(kind="subscribed")
        assert self.submitted
        if self.stall:
            await asyncio.sleep(10)
        for event in self.events:
            yield event


@pytest.mark.asyncio
async def test_subscribe_before_submission_and_unknown_usage(tmp_path):
    client = FakeClient([KimiEvent(kind="text", text="done"), KimiEvent(kind="turn_ended", text="completed")])
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path)
    assert result["status"] == "completed" and result["usage"] is None
    assert json.loads((tmp_path / "run.json").read_text())["text"] == "done"


@pytest.mark.asyncio
async def test_timeout_aborts_and_records_failure(tmp_path):
    client = FakeClient([], stall=True)
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path, timeout_s=0.02)
    assert result["status"] == "timeout" and client.aborted


@pytest.mark.asyncio
async def test_tool_budget_aborts(tmp_path):
    client = FakeClient([KimiEvent(kind="tool_call_start", tool_name="tool")])
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path, max_tool_calls=0)
    assert result["status"] == "budget_exceeded" and client.aborted


def test_pilot_does_not_approve_shell_or_other_mcp():
    assert not safe_approval({"toolName": "Bash"})
    assert not safe_approval({"toolName": "mcp__production__train"})
    assert safe_approval({"toolName": "mcp__venusfactory__sequence_properties"})


@pytest.mark.asyncio
async def test_auth_failure_is_structured_and_cannot_report_success(tmp_path):
    client = FakeClient(
        [
            KimiEvent(
                kind="unknown",
                raw_type="turn.step.interrupted",
                raw={"reason": "error", "message": "401 Authentication failed"},
            ),
            KimiEvent(kind="turn_ended", text="completed"),
        ]
    )
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path)
    assert result["status"] == "runtime_error"
    assert result["error_code"] == "provider_authentication_failed"
    assert client.aborted


@pytest.mark.asyncio
async def test_approval_failure_aborts_without_waiting_for_next_event(tmp_path):
    class BrokenApprovalClient(FakeClient):
        async def list_pending_approvals(self, sid):
            await asyncio.sleep(0.01)
            raise RuntimeError("approval endpoint unavailable")

    client = BrokenApprovalClient([], stall=True)
    result = await asyncio.wait_for(run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path), 1)
    assert result["status"] == "runtime_error"
    assert "approval endpoint unavailable" in result["error"]
    assert client.aborted


@pytest.mark.asyncio
async def test_single_approval_omits_unsupported_scope(tmp_path):
    class ApprovalClient(FakeClient):
        decision = None

        async def list_pending_approvals(self, sid):
            if self.submitted and not self.decision:
                return [{"approval_id": "a", "tool_name": "mcp__venusfactory__sequence_properties"}]
            return []

        async def decide_approval(self, sid, aid, **kwargs):
            self.decision = kwargs

        async def stream_session_events(self, sid):
            yield KimiEvent(kind="subscribed")
            while self.decision is None:
                await asyncio.sleep(0.01)
            yield KimiEvent(kind="turn_ended", text="completed")

    client = ApprovalClient([])
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path, timeout_s=2)
    assert result["status"] == "completed"
    assert client.decision == {"decision": "approved"}


@pytest.mark.asyncio
async def test_provider_empty_response_is_not_task_success(tmp_path):
    client = FakeClient(
        [
            KimiEvent(
                kind="other",
                raw_type="turn.step.interrupted",
                raw={"reason": "error", "message": "The API returned an empty response (no content, no tool calls)."},
            ),
            KimiEvent(kind="turn_ended", text="failed", is_error=True),
        ]
    )
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path)
    assert result["status"] == "runtime_error"
    assert result["error_code"] == "provider_empty_response"


@pytest.mark.asyncio
async def test_model_binding_mismatch_prevents_prompt(tmp_path):
    class DriftClient(FakeClient):
        async def get_status(self, sid):
            return {"model": "wrong", "permission": "manual"}

    client = DriftClient([])
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path)
    assert result["status"] == "runtime_error" and not client.submitted


@pytest.mark.asyncio
async def test_actual_instruction_and_undeclared_tool_audit(tmp_path):
    client = FakeClient([KimiEvent(kind="tool_call_start", tool_name="EnterPlanMode", tool_call_id="x")])
    result = await run_task(
        client,
        prompt="task sentinel",
        system_prompt="policy sentinel",
        model="m",
        output_dir=tmp_path,
        allowed_mcp_tools=frozenset({"sequence_properties"}),
    )
    assert result["status"] == "runtime_error" and client.aborted
    assert "policy sentinel" in client.submitted_text and "task sentinel" in client.submitted_text
    assert not result["tool_visibility_enforced"]
    assert "EnterPlanMode" in (tmp_path / "events.jsonl").read_text()


@pytest.mark.asyncio
async def test_usage_is_read_from_snapshot_not_zero_session_projection(tmp_path):
    class SnapshotClient(FakeClient):
        def json(self):
            return {
                "data": {"session": {"usage": {"input_tokens": 12, "output_tokens": 4, "turn_count": 1}}, "tools": []}
            }

    client = SnapshotClient([KimiEvent(kind="turn_ended", text="completed")])
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path)
    assert result["usage"]["input_tokens"] == 12 and result["usage"]["output_tokens"] == 4
    assert "snapshot_rollup" in result["usage_status"]


@pytest.mark.asyncio
async def test_chat_can_answer_without_tools_or_artifacts(tmp_path):
    client = FakeClient([KimiEvent(kind="text", text="Which protein?"), KimiEvent(kind="turn_ended")])
    result = await run_task(
        client,
        prompt="Predict its properties",
        system_prompt="Assist research",
        model="m",
        output_dir=tmp_path,
        allowed_mcp_tools=frozenset(),
        interaction="chat",
    )
    assert result["status"] == "completed" and result["tool_calls"] == 0
    assert "clarification questions are valid" in client.submitted_text
    assert "Do not enter plan mode" not in client.submitted_text
    assert "Predict its properties" in client.submitted_text


@pytest.mark.asyncio
async def test_empty_chat_completion_is_incomplete(tmp_path):
    client = FakeClient([KimiEvent(kind="turn_ended")])
    result = await run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path, interaction="chat")
    assert result["status"] == "incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize("interaction,expected", [("chat", "needs_input"), ("bounded", "runtime_error")])
async def test_pending_question_keeps_conversation_distinct_from_autonomous_episode(tmp_path, interaction, expected):
    class QuestionClient(FakeClient):
        async def list_pending_questions(self, sid):
            await asyncio.sleep(0.01)
            return [{"question_id": "q1", "questions": [{"question": "Which protein?"}]}]

    client = QuestionClient([], stall=True)
    result = await run_task(
        client,
        prompt="q",
        system_prompt="p",
        model="m",
        output_dir=tmp_path,
        interaction=interaction,
    )
    assert result["status"] == expected and client.aborted
    if interaction == "chat":
        assert result["questions"][0]["question_id"] == "q1"


def test_shared_prompt_retains_original_bounded_policy():
    from agent.rsi.engine.runtime import build_task_message

    assert build_task_message("p", "q") == (
        "Research task instructions (execute in this turn):\np"
        "\nUse only the declared venusfactory MCP tools. Do not enter plan mode, "
        "create a goal, ask questions, or invoke native tools. Complete the task now.\n\nTask:\nq"
    )
    with pytest.raises(ValueError):
        build_task_message("p", "q", interaction="unknown")


@pytest.mark.asyncio
async def test_mcp_readiness_has_a_separate_bounded_deadline(tmp_path):
    class SlowMcp(FakeClient):
        async def list_mcp_servers(self):
            await asyncio.sleep(0.04)
            return [{"id": "venusfactory", "status": "connected"}]

    fast = SlowMcp([KimiEvent(kind="turn_ended")])
    failed = await run_task(
        fast,
        prompt="q",
        system_prompt="p",
        model="m",
        output_dir=tmp_path / "short",
        timeout_s=1,
        mcp_startup_timeout_s=0.01,
    )
    assert failed["status"] == "timeout" and failed["timeout_stage"] == "mcp_connection"
    assert not fast.submitted and not fast.aborted and failed["tool_calls"] == 0
    slow = SlowMcp([KimiEvent(kind="turn_ended")])
    completed = await run_task(
        slow,
        prompt="q",
        system_prompt="p",
        model="m",
        output_dir=tmp_path / "long",
        timeout_s=1,
        mcp_startup_timeout_s=0.2,
    )
    assert completed["status"] == "completed" and slow.submitted


@pytest.mark.asyncio
async def test_larger_startup_allowance_never_extends_total_deadline(tmp_path):
    class StalledMcp(FakeClient):
        async def list_mcp_servers(self):
            await asyncio.sleep(10)

    client = StalledMcp([])
    result = await asyncio.wait_for(
        run_task(
            client,
            prompt="q",
            system_prompt="p",
            model="m",
            output_dir=tmp_path,
            timeout_s=0.02,
            mcp_startup_timeout_s=180,
        ),
        1,
    )
    assert result["status"] == "timeout" and result["timeout_stage"] == "mcp_connection" and not client.submitted
    assert result["mcp_startup_timeout_s"] == 0.02


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, True, float("nan"), float("inf")])
async def test_invalid_startup_limit_rejected_before_session(tmp_path, limit):
    client = FakeClient([])
    with pytest.raises(ValueError):
        await run_task(
            client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path, mcp_startup_timeout_s=limit
        )
    assert not client.submitted


@pytest.mark.asyncio
async def test_native_mcp_failure_stops_before_model_submission(tmp_path):
    class FailedMcp(FakeClient):
        async def list_mcp_servers(self):
            return [{"id": "venusfactory", "status": "failed"}]

    client = FailedMcp([])
    result = await asyncio.wait_for(
        run_task(client, prompt="q", system_prompt="p", model="m", output_dir=tmp_path, mcp_startup_timeout_s=180), 1
    )
    assert result["status"] == "runtime_error" and result["error_code"] == "mcp_startup_failed"
    assert not client.submitted and result["tool_calls"] == 0
