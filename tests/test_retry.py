"""Tests for retry policies and execution."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import pytest
from agent.retry import (
    RetryPolicy,
    RetryResult,
    retry_async,
    retry_sync,
    DEFAULT_RETRY,
    NO_RETRY,
    TOOL_RETRY,
    LLM_RETRY,
)
from exceptions import ToolExecutionError, ToolTimeoutError


class TestRetryPolicy:
    def test_defaults(self):
        p = RetryPolicy()
        assert p.max_retries == 2
        assert p.initial_delay == 1.0
        assert p.backoff_multiplier == 2.0

    def test_invalid_max_retries(self):
        with pytest.raises(ValueError, match="max_retries"):
            RetryPolicy(max_retries=-1)

    def test_invalid_initial_delay(self):
        with pytest.raises(ValueError, match="initial_delay"):
            RetryPolicy(initial_delay=-0.5)

    def test_invalid_backoff(self):
        with pytest.raises(ValueError, match="backoff_multiplier"):
            RetryPolicy(backoff_multiplier=0.5)

    def test_delay_for_attempt(self):
        p = RetryPolicy(initial_delay=1.0, backoff_multiplier=2.0, max_delay=10.0, jitter=False)
        assert p.delay_for_attempt(0) == 1.0
        assert p.delay_for_attempt(1) == 2.0
        assert p.delay_for_attempt(2) == 4.0
        assert p.delay_for_attempt(10) == 10.0

    def test_delay_with_jitter(self):
        p = RetryPolicy(initial_delay=1.0, jitter=True)
        delays = [p.delay_for_attempt(0) for _ in range(100)]
        assert all(1.0 <= d <= 1.25 for d in delays)

    def test_should_retry_matching(self):
        p = RetryPolicy(retry_on=(ValueError, TypeError))
        assert p.should_retry(ValueError("x"))
        assert p.should_retry(TypeError("x"))
        assert not p.should_retry(KeyError("x"))

    def test_should_retry_any(self):
        p = RetryPolicy(retry_on=(Exception,))
        assert p.should_retry(RuntimeError("x"))


class TestPredefinedPolicies:
    def test_no_retry(self):
        assert NO_RETRY.max_retries == 0

    def test_tool_retry_types(self):
        assert ToolExecutionError in TOOL_RETRY.retry_on
        assert ToolTimeoutError in TOOL_RETRY.retry_on


class TestRetrySync:
    def test_success_no_retry(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return 42

        result = retry_sync(fn, policy=DEFAULT_RETRY)
        assert result.success
        assert result.value == 42
        assert result.attempts == 1
        assert call_count == 1

    def test_eventual_success(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        policy = RetryPolicy(max_retries=3, initial_delay=0.01, jitter=False)
        result = retry_sync(fn, policy=policy)
        assert result.success
        assert result.value == "ok"
        assert result.attempts == 3

    def test_all_attempts_fail(self):
        def fn():
            raise ValueError("always fails")

        policy = RetryPolicy(max_retries=2, initial_delay=0.01, jitter=False)
        result = retry_sync(fn, policy=policy)
        assert not result.success
        assert isinstance(result.last_error, ValueError)
        assert result.attempts == 3

    def test_non_retryable_exception(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise KeyError("wrong type")

        policy = RetryPolicy(max_retries=3, initial_delay=0.01, retry_on=(ValueError,))
        result = retry_sync(fn, policy=policy)
        assert not result.success
        assert call_count == 1


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_success(self):
        async def fn():
            return 99

        result = await retry_async(fn, policy=DEFAULT_RETRY)
        assert result.success
        assert result.value == 99

    @pytest.mark.asyncio
    async def test_eventual_success(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry")
            return "done"

        policy = RetryPolicy(max_retries=3, initial_delay=0.01, jitter=False)
        result = await retry_async(fn, policy=policy)
        assert result.success
        assert result.value == "done"

    @pytest.mark.asyncio
    async def test_sync_fn_in_async_retry(self):
        def fn():
            return "sync_result"

        result = await retry_async(fn, policy=DEFAULT_RETRY)
        assert result.success
        assert result.value == "sync_result"
