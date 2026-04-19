"""
Retry policies with configurable backoff for VenusFactory.

Provides composable retry logic that replaces hardcoded MAX_STEP_RETRIES
constants with policy objects that can be passed through the system.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from exceptions import ToolExecutionError, ToolTimeoutError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Declarative retry configuration.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        initial_delay: Seconds to wait before the first retry.
        backoff_multiplier: Factor to multiply the delay after each attempt.
        max_delay: Upper bound on delay between retries.
        jitter: If True, add random jitter (0–25 % of delay) to prevent thundering herd.
        retry_on: Tuple of exception types that should trigger a retry.
            If empty, retries on any Exception.
    """

    max_retries: int = 2
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {self.initial_delay}")
        if self.backoff_multiplier < 1:
            raise ValueError(f"backoff_multiplier must be >= 1, got {self.backoff_multiplier}")

    def delay_for_attempt(self, attempt: int) -> float:
        """Compute the delay (in seconds) before retry *attempt* (0-indexed)."""
        delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay += random.uniform(0, delay * 0.25)
        return delay

    def should_retry(self, error: Exception) -> bool:
        if not self.retry_on:
            return True
        return isinstance(error, self.retry_on)


# Predefined policies
DEFAULT_RETRY = RetryPolicy(max_retries=2, initial_delay=1.0)
NO_RETRY = RetryPolicy(max_retries=0)
TOOL_RETRY = RetryPolicy(
    max_retries=2,
    initial_delay=2.0,
    backoff_multiplier=2.0,
    retry_on=(ToolExecutionError, ToolTimeoutError, TimeoutError, ConnectionError),
)
LLM_RETRY = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    backoff_multiplier=2.0,
    max_delay=15.0,
    retry_on=(ConnectionError, TimeoutError, OSError),
)


@dataclass
class RetryResult:
    """Outcome of a retried operation."""

    success: bool
    value: Any = None
    last_error: Exception | None = None
    attempts: int = 0
    total_delay: float = 0.0


async def retry_async(
    fn: Callable[..., Any],
    *args: Any,
    policy: RetryPolicy = DEFAULT_RETRY,
    **kwargs: Any,
) -> RetryResult:
    """Execute *fn* with retries according to *policy*.

    Works with both sync and async callables.
    """
    last_error: Exception | None = None
    total_delay = 0.0

    for attempt in range(policy.max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return RetryResult(success=True, value=result, attempts=attempt + 1, total_delay=total_delay)
        except Exception as exc:
            last_error = exc
            if attempt >= policy.max_retries or not policy.should_retry(exc):
                break
            delay = policy.delay_for_attempt(attempt)
            total_delay += delay
            await asyncio.sleep(delay)

    return RetryResult(
        success=False,
        last_error=last_error,
        attempts=policy.max_retries + 1,
        total_delay=total_delay,
    )


def retry_sync(
    fn: Callable[..., T],
    *args: Any,
    policy: RetryPolicy = DEFAULT_RETRY,
    **kwargs: Any,
) -> RetryResult:
    """Synchronous version of retry."""
    last_error: Exception | None = None
    total_delay = 0.0

    for attempt in range(policy.max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            return RetryResult(success=True, value=result, attempts=attempt + 1, total_delay=total_delay)
        except Exception as exc:
            last_error = exc
            if attempt >= policy.max_retries or not policy.should_retry(exc):
                break
            delay = policy.delay_for_attempt(attempt)
            total_delay += delay
            time.sleep(delay)

    return RetryResult(
        success=False,
        last_error=last_error,
        attempts=policy.max_retries + 1,
        total_delay=total_delay,
    )
