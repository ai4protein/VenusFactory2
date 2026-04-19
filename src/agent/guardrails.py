"""
Guardrails framework for VenusFactory tool execution.

Provides input/output validation with:
  - Decorator-based guardrail registration
  - Tripwire pattern (halt on violation)
  - Built-in guardrails for path traversal, code safety, input length
  - Parallel/sequential execution modes
"""
from __future__ import annotations

import inspect
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, Union, overload

from exceptions import InputGuardrailTripped, OutputGuardrailTripped

T = TypeVar("T")
MaybeAwaitable = Union[Awaitable[T], T]

GuardrailFn = Callable[..., MaybeAwaitable["GuardrailResult"]]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Output of a guardrail check."""

    tripwire_triggered: bool = False
    output_info: Any = None
    message: str = ""

    @staticmethod
    def passed(message: str = "") -> GuardrailResult:
        return GuardrailResult(tripwire_triggered=False, message=message)

    @staticmethod
    def tripped(message: str, output_info: Any = None) -> GuardrailResult:
        return GuardrailResult(tripwire_triggered=True, message=message, output_info=output_info)


# ---------------------------------------------------------------------------
# Guardrail classes
# ---------------------------------------------------------------------------

@dataclass
class InputGuardrail:
    """Validates input before tool execution.

    Attributes:
        name: Human-readable guardrail name.
        fn: Callable that takes (tool_name, tool_input, **kwargs) -> GuardrailResult.
        run_in_parallel: If True, runs concurrently with other guardrails.
    """

    name: str = ""
    fn: GuardrailFn | None = None
    run_in_parallel: bool = True

    async def run(self, tool_name: str, tool_input: dict[str, Any], **kwargs: Any) -> GuardrailResult:
        if self.fn is None:
            return GuardrailResult.passed()
        result = self.fn(tool_name=tool_name, tool_input=tool_input, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result  # type: ignore[return-value]


@dataclass
class OutputGuardrail:
    """Validates output after tool execution.

    Attributes:
        name: Human-readable guardrail name.
        fn: Callable that takes (tool_name, tool_output, **kwargs) -> GuardrailResult.
        run_in_parallel: If True, runs concurrently with other guardrails.
    """

    name: str = ""
    fn: GuardrailFn | None = None
    run_in_parallel: bool = True

    async def run(self, tool_name: str, tool_output: Any, **kwargs: Any) -> GuardrailResult:
        if self.fn is None:
            return GuardrailResult.passed()
        result = self.fn(tool_name=tool_name, tool_output=tool_output, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

@overload
def input_guardrail(fn: GuardrailFn) -> InputGuardrail: ...

@overload
def input_guardrail(
    *, name: str = "", run_in_parallel: bool = True
) -> Callable[[GuardrailFn], InputGuardrail]: ...

def input_guardrail(
    fn: GuardrailFn | None = None,
    *,
    name: str = "",
    run_in_parallel: bool = True,
) -> InputGuardrail | Callable[[GuardrailFn], InputGuardrail]:
    """Decorator to create an InputGuardrail.

    Usage::

        @input_guardrail
        def my_guard(tool_name, tool_input, **kwargs):
            if len(str(tool_input)) > 10000:
                return GuardrailResult.tripped("Input too large")
            return GuardrailResult.passed()

        @input_guardrail(name="path_check", run_in_parallel=False)
        def path_guard(tool_name, tool_input, **kwargs):
            ...
    """
    def decorator(f: GuardrailFn) -> InputGuardrail:
        return InputGuardrail(
            name=name or f.__name__,
            fn=f,
            run_in_parallel=run_in_parallel,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


@overload
def output_guardrail(fn: GuardrailFn) -> OutputGuardrail: ...

@overload
def output_guardrail(
    *, name: str = "", run_in_parallel: bool = True
) -> Callable[[GuardrailFn], OutputGuardrail]: ...

def output_guardrail(
    fn: GuardrailFn | None = None,
    *,
    name: str = "",
    run_in_parallel: bool = True,
) -> OutputGuardrail | Callable[[GuardrailFn], OutputGuardrail]:
    """Decorator to create an OutputGuardrail."""
    def decorator(f: GuardrailFn) -> OutputGuardrail:
        return OutputGuardrail(
            name=name or f.__name__,
            fn=f,
            run_in_parallel=run_in_parallel,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# Guardrail runner
# ---------------------------------------------------------------------------

async def run_input_guardrails(
    guardrails: list[InputGuardrail],
    tool_name: str,
    tool_input: dict[str, Any],
    **kwargs: Any,
) -> list[GuardrailResult]:
    """Execute input guardrails. Raises InputGuardrailTripped on first violation."""
    import asyncio

    results: list[GuardrailResult] = []
    parallel = [g for g in guardrails if g.run_in_parallel]
    sequential = [g for g in guardrails if not g.run_in_parallel]

    for g in sequential:
        result = await g.run(tool_name, tool_input, **kwargs)
        results.append(result)
        if result.tripwire_triggered:
            raise InputGuardrailTripped(
                result.message or f"Input guardrail '{g.name}' tripped",
                guardrail_name=g.name,
            )

    if parallel:
        coros = [g.run(tool_name, tool_input, **kwargs) for g in parallel]
        parallel_results = await asyncio.gather(*coros, return_exceptions=True)
        for g, r in zip(parallel, parallel_results):
            if isinstance(r, Exception):
                results.append(GuardrailResult.tripped(f"Guardrail error: {r}"))
                continue
            results.append(r)
            if r.tripwire_triggered:
                raise InputGuardrailTripped(
                    r.message or f"Input guardrail '{g.name}' tripped",
                    guardrail_name=g.name,
                )

    return results


async def run_output_guardrails(
    guardrails: list[OutputGuardrail],
    tool_name: str,
    tool_output: Any,
    **kwargs: Any,
) -> list[GuardrailResult]:
    """Execute output guardrails. Raises OutputGuardrailTripped on first violation."""
    import asyncio

    results: list[GuardrailResult] = []
    parallel = [g for g in guardrails if g.run_in_parallel]
    sequential = [g for g in guardrails if not g.run_in_parallel]

    for g in sequential:
        result = await g.run(tool_name, tool_output, **kwargs)
        results.append(result)
        if result.tripwire_triggered:
            raise OutputGuardrailTripped(
                result.message or f"Output guardrail '{g.name}' tripped",
                guardrail_name=g.name,
            )

    if parallel:
        coros = [g.run(tool_name, tool_output, **kwargs) for g in parallel]
        parallel_results = await asyncio.gather(*coros, return_exceptions=True)
        for g, r in zip(parallel, parallel_results):
            if isinstance(r, Exception):
                results.append(GuardrailResult.tripped(f"Guardrail error: {r}"))
                continue
            results.append(r)
            if r.tripwire_triggered:
                raise OutputGuardrailTripped(
                    r.message or f"Output guardrail '{g.name}' tripped",
                    guardrail_name=g.name,
                )

    return results


# ---------------------------------------------------------------------------
# Built-in guardrails
# ---------------------------------------------------------------------------

_FORBIDDEN_CODE_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\.\w+",
    r"\b__import__\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bshutil\.rmtree\b",
    r"\bos\.remove\b",
    r"\bos\.unlink\b",
    r"\bos\.rmdir\b",
    r"\bsocket\.\w+",
    r"\bos\.environ\[",
    r"\bos\.environ\.update\b",
]

_FORBIDDEN_CODE_RE = re.compile("|".join(_FORBIDDEN_CODE_PATTERNS), re.IGNORECASE)


@input_guardrail(name="code_safety", run_in_parallel=False)
def code_safety_guardrail(tool_name: str, tool_input: dict[str, Any], **kwargs: Any) -> GuardrailResult:
    """Block dangerous code patterns in python_repl and agent_generated_code."""
    if tool_name not in ("python_repl", "agent_generated_code"):
        return GuardrailResult.passed()

    code = ""
    for key in ("query", "code", "script", "python", "source"):
        val = tool_input.get(key)
        if isinstance(val, str) and val.strip():
            code = val
            break

    if not code:
        return GuardrailResult.passed()

    match = _FORBIDDEN_CODE_RE.search(code)
    if match:
        return GuardrailResult.tripped(
            f"Forbidden code pattern detected: '{match.group()}'",
            output_info={"pattern": match.group(), "position": match.start()},
        )
    return GuardrailResult.passed()


@input_guardrail(name="path_traversal", run_in_parallel=True)
def path_traversal_guardrail(tool_name: str, tool_input: dict[str, Any], **kwargs: Any) -> GuardrailResult:
    """Prevent directory traversal attacks via file path parameters."""
    sensitive_dirs = {"/etc", "/root", "/proc", "/sys", "/dev"}

    for key, value in tool_input.items():
        if not isinstance(value, str):
            continue
        key_lower = key.lower()
        if not any(tok in key_lower for tok in ("path", "file", "dir", "fasta")):
            continue

        normalized = os.path.normpath(value)
        if ".." in normalized.split(os.sep):
            return GuardrailResult.tripped(
                f"Path traversal detected in parameter '{key}': {value}",
                output_info={"parameter": key, "value": value},
            )

        for d in sensitive_dirs:
            if normalized.startswith(d):
                return GuardrailResult.tripped(
                    f"Access to sensitive directory '{d}' blocked in parameter '{key}'",
                    output_info={"parameter": key, "directory": d},
                )

    return GuardrailResult.passed()


@input_guardrail(name="input_size_limit", run_in_parallel=True)
def input_size_guardrail(tool_name: str, tool_input: dict[str, Any], **kwargs: Any) -> GuardrailResult:
    """Reject tool inputs that exceed size limits."""
    max_total_bytes = kwargs.get("max_input_bytes", 10 * 1024 * 1024)  # 10 MB default

    import json
    try:
        total = len(json.dumps(tool_input, default=str).encode())
    except Exception:
        total = 0

    if total > max_total_bytes:
        return GuardrailResult.tripped(
            f"Tool input size ({total:,} bytes) exceeds limit ({max_total_bytes:,} bytes)",
            output_info={"size_bytes": total, "limit_bytes": max_total_bytes},
        )
    return GuardrailResult.passed()


DEFAULT_INPUT_GUARDRAILS: list[InputGuardrail] = [
    code_safety_guardrail,
    path_traversal_guardrail,
    input_size_guardrail,
]
