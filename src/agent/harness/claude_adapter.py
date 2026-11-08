"""Experimental native Claude execution through the admitted free-model bridge.

The upstream is Chat Completions; only the local native client uses Messages.
Capability loading, native events and shutdown remain explicit host operations.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path

from .adapters import task_message
from .claude_events import ClaudeEvents
from .claude_policy import validate_profile as validate_profile
from .codex_adapter import validate_tool_result
from .host import write_json
from .messages_bridge import PREFIX, TRANSPORT
from .prepared_mcp import PreparedMcp


def sandbox_command(ctx, home, prepared):
    p, task = ctx.profile, ctx.task
    validate_profile(p)
    command = [
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    from agent.harness.system_mounts import system_mounts
    command += system_mounts()
    from agent.harness.python_runtime_mounts import python_runtime_mounts
    command += python_runtime_mounts(ctx.runtimes["python_root"])
    command += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--ro-bind",
        str(Path(ctx.runtimes["claude_bin"]).resolve()),
        "/opt/claude",
        "--bind",
        str(task.workspace),
        "/workspace",
        "--ro-bind",
        str(task.capability),
        "/capabilities",
        "--bind",
        str(home),
        "/runtime_home",
        "--ro-bind",
        str(home / "passwd"),
        "/etc/passwd",
        "--ro-bind",
        str(home / "group"),
        "/etc/group",
    ]
    command += prepared.mounts()
    command += ["--chdir", "/workspace", "--clearenv"]
    env = {
        "HOME": "/runtime_home",
        "PATH": "/opt/venus-runtime/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "VENUS_RSI_WORKSPACE": "/workspace",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "ANTHROPIC_BASE_URL": ctx.relay.base_url.removesuffix("/v1"),
        "ANTHROPIC_API_KEY": ctx.relay.token,
        "ANTHROPIC_MODEL": p.model.model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": p.model.model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": p.model.model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": p.model.model,
        "CLAUDE_CODE_SUBAGENT_MODEL": p.model.model,
        "CLAUDE_CONFIG_DIR": "/runtime_home/config",
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "MAX_THINKING_TOKENS": "0",
        "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
        "CLAUDE_CODE_EFFORT_LEVEL": "unset",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(p.budget.output_tokens),
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(p.model.context_tokens),
        "CLAUDE_CODE_MAX_RETRIES": "0",
        "MCP_DISCOVERY_CACHE": "0",
        "MCP_CONNECTION_NONBLOCKING": "0",
        "MCP_CONNECT_TIMEOUT_MS": "120000",
        "MCP_TIMEOUT": str(int(min(120, p.budget.wall_seconds) * 1000)),
        "MCP_TOOL_TIMEOUT": str(int(p.budget.wall_seconds * 1000)),
    }
    for name, value in env.items():
        command += ["--setenv", name, value]
    return command + [
        "--",
        "/opt/claude",
        "--bare",
        "-p",
        "--model",
        p.model.model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--mcp-config",
        "/runtime_home/mcp.json",
        "--strict-mcp-config",
        "--tools",
        "",
        "--allowedTools",
        ",".join(PREFIX + name for name in sorted(task.tools)),
        "--max-turns",
        str(p.budget.model_calls),
        "--no-session-persistence",
    ]


async def stop_process(process):
    if process is None or process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 3)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


class ClaudeAdapter:
    def __init__(self, runtimes):
        self.runtimes = runtimes

    async def run(self, ctx):
        p, task = ctx.profile, ctx.task
        validate_profile(p)
        home = task.output / "claude_home"
        home.mkdir(mode=0o700)
        (home / "passwd").write_text(f"venus:x:{os.getuid()}:{os.getgid()}:Research runtime:/runtime_home:/bin/false\n")
        (home / "group").write_text(f"venus:x:{os.getgid()}:\n")
        prepared = PreparedMcp(
            workspace=task.workspace,
            capability=task.capability,
            python_root=Path(self.runtimes["python_root"]),
            control_dir=task.output / "capability_worker",
            entry=task.mcp_entry,
            timeout_s=min(180, p.budget.wall_seconds),
        )
        web_tools = task.phase == 'chat' and task.mcp_entry == 'web_tools.py'
        validator = None if web_tools else lambda name, data: validate_tool_result(task, name, data)
        state = ClaudeEvents(task.tools, p.budget.tool_calls, validator)
        result = {
            "status": "running",
            "engine": "claude-code",
            "phase": task.phase,
            "mode": p.mode,
            "human_interventions": 0,
            "model": p.model.model,
            "transport": TRANSPORT,
            "experimental_non_claude_bridge": True,
        }
        started = time.monotonic()
        process = None
        stage = "capability_loading"
        cancelled = False
        try:
            with (
                (task.output / "stderr.log").open("xb") as stderr,
                (task.output / "native_events.jsonl").open("xb") as raw,
                (task.output / "events.jsonl").open("x") as trace,
            ):
                async with asyncio.timeout(p.budget.wall_seconds):
                    result["capability_preparation"] = await prepared.start()
                    write_json(
                        home / "mcp.json", {"mcpServers": {"venusfactory": {"type": "stdio", **prepared.connection()}}}
                    )
                    stage = "native_execution"
                    process = await asyncio.create_subprocess_exec(
                        *sandbox_command(ctx, home, prepared),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=stderr,
                        limit=8_000_000,
                        start_new_session=True,
                    )
                    try:
                        process.stdin.write(task_message(task).encode())
                        await process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        # An early CLI exit can close stdin before drain. Still
                        # consume stdout and record its actual terminal status.
                        pass
                    finally:
                        process.stdin.close()
                    while line := await process.stdout.readline():
                        raw.write(line)
                        raw.flush()
                        for event in state.consume(json.loads(line)):
                            trace.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                            trace.flush()
                    await process.wait()
                    result.update(
                        returncode=process.returncode,
                        status="completed"
                        if process.returncode == 0 and state.completed and not state.failed
                        else "runtime_error",
                    )
        except asyncio.CancelledError:
            cancelled = True
            result.update(status="interrupted", error_type="CancelledError")
        except TimeoutError:
            result.update(status="timeout", timeout_stage=stage, error_type="TimeoutError")
        except Exception as exc:
            result.update(status="protocol_failed", error_type=type(exc).__name__, failure_stage=stage)
        finally:

            async def cleanup():
                try:
                    await stop_process(process)
                finally:
                    await prepared.stop()

            cleanup_task = asyncio.create_task(cleanup())
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cancelled = True
                result.update(status="interrupted", error_type="CancelledError")
                await cleanup_task
            result.update(
                session_id=state.session_id,
                tool_calls=len(state.started),
                tool_errors=state.tool_errors,
                text=state.final_text,
                final_text=state.final_text,
                final_message=state.final_text,
                sandbox_stopped=process is None or process.returncode is not None,
                capability_worker_stopped=prepared.stopped,
                wall_seconds=round(time.monotonic() - started, 3),
            )
            write_json(task.output / "native_checkpoint.json", result)
        if cancelled:
            raise asyncio.CancelledError
        return result
