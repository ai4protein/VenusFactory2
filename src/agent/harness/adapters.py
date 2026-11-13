"""Runtime-specific code is confined to these replaceable adapters."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path

from .host import write_json
from .kimi_adapter import KimiAdapter  # Compatibility for existing expert and CLI callers.


def task_message(task):
    from agent.rsi.engine.runtime import build_task_message

    return build_task_message(task.instructions, task.prompt, interaction=task.interaction_mode)


class ScientistAdapter:
    def __init__(self, runtimes):
        self.runtimes = runtimes

    async def run(self, ctx):
        from agent.rsi.engine.scientist_runtime import run_scientist

        p, t = ctx.profile, ctx.task
        if t.mcp_entry != "generated_tools.py" or t.phase not in {"develop", "reuse", "chat"}:
            raise ValueError("Role graph requires a supported scientific episode and MCP entry")
        return await run_scientist(
            workspace=t.workspace,
            capability=t.capability,
            python_root=Path(self.runtimes["python_root"]),
            output=t.output,
            prompt=task_message(t),
            model=p.model.model,
            base_url=ctx.relay.base_url,
            api_key=ctx.relay.token,
            allowed_tools=t.tools,
            timeout_s=p.budget.wall_seconds,
            max_tool_calls=p.budget.tool_calls,
            research_policy="task_context_only",
            phase=t.phase,
            request_timeout_s=p.budget.request_seconds + 30,
            max_output_tokens=p.budget.output_tokens,
            max_model_calls=p.budget.model_calls,
        )


class LangGraphAdapter:
    """One execution-library family with two explicit scientific workflows."""

    def __init__(self, runtimes):
        self.runtimes = runtimes

    async def run(self, ctx):
        if ctx.profile.mode == "science_expert":
            return await ScientistAdapter(self.runtimes).run(ctx)
        from agent.rsi.engine.adaptive_runtime import run_adaptive

        p, t = ctx.profile, ctx.task
        if t.mcp_entry != "generated_tools.py":
            raise ValueError("LangGraph adapter currently requires the generated-model MCP entry")
        return await run_adaptive(
            workspace=t.workspace,
            capability=t.capability,
            python_root=Path(self.runtimes["python_root"]),
            output=t.output,
            prompt=task_message(t),
            model=p.model.model,
            base_url=ctx.relay.base_url,
            api_key=ctx.relay.token,
            allowed_tools=t.tools,
            timeout_s=p.budget.wall_seconds,
            max_tool_calls=p.budget.tool_calls,
            max_model_calls=p.budget.model_calls,
            phase=t.phase,
            request_timeout_s=p.budget.request_seconds + 30,
            max_output_tokens=p.budget.output_tokens,
        )


def dsh_command(ctx, controls):
    t, runtimes = ctx.task, ctx.runtimes
    cmd = [
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
    cmd += system_mounts()
    from agent.harness.python_runtime_mounts import python_runtime_mounts
    cmd += python_runtime_mounts(runtimes["python_root"])
    cmd += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--dir",
        "/opt",
        "--ro-bind",
        runtimes["dsh_site"],
        "/opt/dsh-site",
        "--ro-bind",
        str(Path(__file__).parent),
        "/runner",
        "--bind",
        str(t.workspace),
        "/workspace",
        "--ro-bind",
        str(t.capability),
        "/capabilities",
        "--bind",
        str(controls),
        "/evidence",
        # SDK boot generates hundreds of package shims. Keep these disposable
        # files off shared storage; only events and diagnostics are persisted.
        "--tmpfs",
        "/runtime_home",
        "--chdir",
        "/workspace",
        "--clearenv",
        "--setenv",
        "HOME",
        "/runtime_home",
        "--setenv",
        "DSH_HOME",
        "/runtime_home/dsh",
        "--setenv",
        "PATH",
        "/opt/venus-runtime/bin:/usr/bin:/bin",
        "--setenv",
        "PYTHONPATH",
        "/opt/dsh-site",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--setenv",
        "VENUS_RSI_WORKSPACE",
        "/workspace",
        "--",
        "/opt/venus-runtime/bin/python",
        "-u",
        "/runner/dsh_worker.py",
    ]
    return cmd


async def _worker_lines(stream):
    """Read complete worker records without truncating native event payloads."""
    pending = bytearray()
    while chunk := await stream.read(65536):
        pending.extend(chunk)
        while (end := pending.find(b"\n")) >= 0:
            yield bytes(pending[:end])
            del pending[:end + 1]
    if pending:
        yield bytes(pending)


def _dsh_observation(event, session_id, gateway_calls, notification=None):
    from .events import native_compaction, normalize_dsh

    reference = native_compaction(event, phase="completed", native_type=event.get("type"),
                                  session_id=session_id, path="runtime_events.jsonl")
    rows = normalize_dsh(event, session_id=session_id)
    if (event.get("type") == "turn/end"
            and event.get("data", {}).get("reason", {}).get("kind") == "completed"):
        rows.append({**reference, "kind": "native_terminal"})
    for row in rows:
        row.update(gateway_calls=gateway_calls, native_event_path=reference["native_event_path"],
                   native_event_sha256=reference["native_event_sha256"])
        if notification is not None:
            if notification.get("sessionId") != session_id or notification.get("event") != event:
                raise ValueError("Native notification session or event mismatch")
            envelope = native_compaction(notification, phase="completed", native_type=event.get("type"),
                session_id=session_id, path="runtime_notifications.jsonl")
            row.update(native_notification_path=envelope["native_event_path"],
                       native_notification_sha256=envelope["native_event_sha256"])
    return rows


class DeepSeekAdapter:
    def __init__(self, runtimes):
        self.runtimes = runtimes

    async def run(self, ctx):
        t, p = ctx.task, ctx.profile
        controls = t.output / "control"
        controls.mkdir()
        controls.chmod(0o700)
        request = {
            "model": p.model.model,
            "phase": t.phase,
            "base_url": ctx.relay.base_url,
            "token": ctx.relay.token,
            "prompt": task_message(t),
            "tools": sorted(t.tools),
            "entry": t.mcp_entry,
            "max_tokens": p.budget.output_tokens,
            "timeout": p.budget.wall_seconds,
            "tool_calls": p.budget.tool_calls,
            "context_tokens": p.model.context_tokens,
        }
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *dsh_command(ctx, controls),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        result = {"status": "runtime_error", "tool_calls": 0}
        stderr_reader = None
        try:
            stderr_reader = asyncio.create_task(proc.stderr.read())
            async with asyncio.timeout(p.budget.wall_seconds + 35):
                proc.stdin.write(json.dumps(request).encode())
                await proc.stdin.drain()
                proc.stdin.close()
                with (t.output / "events.jsonl").open("x") as trace:
                    async for line in _worker_lines(proc.stdout):
                        record = json.loads(line)
                        if "native_event" in record:
                            for row in _dsh_observation(record["native_event"], record["session_id"],
                                                       ctx.relay.calls, record.get("native_notification")):
                                trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                trace.flush()
                        else:
                            result = record
                await proc.wait()
            stderr = await stderr_reader
            # Redact the ephemeral token even in retained boot diagnostics.
            (t.output / "runtime.log").write_text(
                stderr.decode(errors="replace").replace(ctx.relay.token, "[redacted]")
            )
            if proc.returncode != 0:
                result.update(status="runtime_error", exit_code=proc.returncode)
        except TimeoutError:
            result["status"] = "timeout"
        finally:
            if proc.returncode is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), 5)
                except TimeoutError:
                    os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()
            if stderr_reader is not None:
                stderr = await stderr_reader
                (t.output / "runtime.log").write_text(
                    stderr.decode(errors="replace").replace(ctx.relay.token, "[redacted]"))
            for name in ["runtime_events.jsonl", "runtime_notifications.jsonl", "composition.patch.yml"]:
                src = controls / name
                if src.exists():
                    (t.output / name).write_bytes(src.read_bytes())
            result["sandbox_stopped"] = proc.returncode is not None
            result["wall_seconds"] = round(time.monotonic() - started, 3)
        write_json(t.output / "run.json", result)
        return result
