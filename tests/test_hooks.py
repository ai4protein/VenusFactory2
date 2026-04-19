"""Tests for lifecycle hooks system."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import pytest
from agent.hooks import (
    RunHooks,
    AgentHooks,
    CompositeRunHooks,
    LoggingRunHooks,
    ToolCallInfo,
    ToolResultInfo,
    _resolve,
)


class TestToolInfoDataclasses:
    def test_tool_call_info(self):
        info = ToolCallInfo(tool_name="blast", tool_input={"seq": "ACGT"}, step_index=1)
        assert info.tool_name == "blast"
        assert info.step_index == 1

    def test_tool_result_info(self):
        info = ToolResultInfo(
            tool_name="blast",
            success=False,
            error_message="timeout",
            duration_seconds=5.5,
        )
        assert not info.success
        assert info.duration_seconds == 5.5


class TestRunHooksNoop:
    @pytest.mark.asyncio
    async def test_all_noop(self):
        hooks = RunHooks()
        await _resolve(hooks.on_run_start(session_id="s1", user_message="hi"))
        await _resolve(hooks.on_run_end(session_id="s1", success=True))
        await _resolve(hooks.on_run_error(session_id="s1", error=RuntimeError("x")))
        await _resolve(hooks.on_tool_start(info=ToolCallInfo()))
        await _resolve(hooks.on_tool_end(info=ToolResultInfo()))
        await _resolve(hooks.on_step_start(step_index=0))
        await _resolve(hooks.on_step_end(step_index=0, success=True))
        await _resolve(hooks.on_llm_call(model_name="gpt-4"))


class TestAgentHooksNoop:
    @pytest.mark.asyncio
    async def test_all_noop(self):
        hooks = AgentHooks()
        await _resolve(hooks.on_agent_start(agent_name="PI"))
        await _resolve(hooks.on_agent_end(agent_name="PI", success=True))


class _TrackingHooks(RunHooks):
    def __init__(self):
        self.events = []

    def on_run_start(self, **kwargs):
        self.events.append(("start", kwargs.get("session_id")))

    def on_run_end(self, **kwargs):
        self.events.append(("end", kwargs.get("session_id")))

    def on_tool_start(self, **kwargs):
        info = kwargs.get("info")
        self.events.append(("tool_start", info.tool_name if info else None))


class TestCompositeRunHooks:
    @pytest.mark.asyncio
    async def test_fires_all(self):
        h1 = _TrackingHooks()
        h2 = _TrackingHooks()
        composite = CompositeRunHooks([h1, h2])

        await composite.on_run_start(session_id="s1", user_message="hi")
        assert len(h1.events) == 1
        assert len(h2.events) == 1
        assert h1.events[0] == ("start", "s1")

    @pytest.mark.asyncio
    async def test_add_hook(self):
        composite = CompositeRunHooks()
        h = _TrackingHooks()
        composite.add(h)
        await composite.on_run_start(session_id="s2", user_message="test")
        assert len(h.events) == 1

    @pytest.mark.asyncio
    async def test_error_in_one_does_not_block_others(self):
        class _Failing(RunHooks):
            def on_run_start(self, **kwargs):
                raise RuntimeError("boom")

        h = _TrackingHooks()
        composite = CompositeRunHooks([_Failing(), h])
        await composite.on_run_start(session_id="s3", user_message="x")
        assert len(h.events) == 1


class TestMaybeAwaitable:
    @pytest.mark.asyncio
    async def test_sync_value(self):
        result = await _resolve(42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_value(self):
        async def get():
            return 99

        result = await _resolve(get())
        assert result == 99


class TestLoggingRunHooks:
    @pytest.mark.asyncio
    async def test_instantiation(self):
        hooks = LoggingRunHooks()
        await _resolve(hooks.on_run_start(session_id="s1", user_message="hello"))
        await _resolve(hooks.on_run_end(session_id="s1", success=True))
