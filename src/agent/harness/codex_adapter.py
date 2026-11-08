"""Pinned native Codex CLI behind the host's Responses evidence gateway."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path

from .adapters import task_message
from .mcp_config import codex_mcp_configuration, scientific_server
from .responses_gateway import ResponsesPolicy


class CodexEvents:
    def __init__(self, allowed_tools, limit, validator=None):
        self.allowed = set(allowed_tools)
        self.limit, self.validator = limit, validator
        self.started, self.finished = {}, set()
        self.thread_id = None
        self.turn_started = False
        self.turn_completed = False
        self.failed = False
        self.final_text = ""
        self.metadata_fallback = False
        self.tool_errors = 0

    def consume(self, event):
        if not isinstance(event, dict):
            raise ValueError("Malformed native Codex event")
        kind = event.get("type")
        if kind == "thread.started":
            if self.thread_id is not None or not isinstance(event.get("thread_id"), str) or not event["thread_id"]:
                raise ValueError("Invalid native thread identity")
            self.thread_id = event["thread_id"]
        elif kind == "turn.started":
            if self.thread_id is None or self.turn_started:
                raise ValueError("Unexpected native turn")
            self.turn_started = True
        elif kind == "turn.completed":
            if not self.turn_started or self.turn_completed or self.failed or self.started.keys() != self.finished:
                raise ValueError("Unfinished native turn evidence")
            self.turn_completed = True
        elif kind in {"error", "turn.failed"}:
            self.failed = True
        elif kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                raise ValueError("Malformed native item")
            item_type = item.get("type")
            if item_type == "mcp_tool_call":
                if not self.turn_started or self.turn_completed or self.failed:
                    raise ValueError("Tool outside active native turn")
                identity, name = item.get("id"), item.get("tool")
                if (
                    item.get("server") != "venusfactory"
                    or name not in self.allowed
                    or not isinstance(identity, str)
                    or not identity
                ):
                    raise ValueError("Unadmitted MCP tool identity")
                if not isinstance(item.get("arguments"), dict):
                    raise ValueError("Malformed MCP arguments")
                binding = {"tool": name, "arguments": item["arguments"]}
                if kind == "item.started":
                    if identity in self.started or item.get("status") != "in_progress":
                        raise ValueError("Repeated or malformed MCP start")
                    if len(self.started) >= self.limit:
                        raise ValueError("Native tool budget exceeded")
                    self.started[identity] = binding
                    return [
                        {
                            "kind": "tool_call_start",
                            "tool_call_id": identity,
                            "tool_name": "mcp__venusfactory__" + name,
                            "args": item["arguments"],
                        }
                    ]
                if self.started.get(identity) != binding or identity in self.finished:
                    raise ValueError("Unpaired or changed MCP completion")
                if kind == "item.updated":
                    return []
                if item.get("status") not in {"completed", "failed"}:
                    raise ValueError("Unknown MCP completion status")
                failed = item["status"] == "failed" or item.get("error") is not None
                result = item.get("result")
                if not failed and not isinstance(result, dict):
                    raise ValueError("Missing MCP result")
                if isinstance(result, dict):
                    failed |= bool(result.get("isError", result.get("is_error", False)))
                if failed:
                    text = json.dumps({"native_tool_error": True}, sort_keys=True)
                else:
                    data = result.get("structured_content")
                    if data is None:
                        parts = result.get("content", [])
                        text = "\n".join(part.get("text", "") for part in parts if part.get("type") == "text")
                        if self.validator:
                            data = json.loads(text)
                    else:
                        text = json.dumps(data, ensure_ascii=False, allow_nan=False)
                    if self.validator:
                        if not isinstance(data, dict):
                            raise ValueError("MCP result must be a structured mapping")
                        self.validator(name, data)
                        text = json.dumps(data, ensure_ascii=False, allow_nan=False)
                self.finished.add(identity)
                self.tool_errors += int(failed)
                return [{"kind": "tool_result", "tool_call_id": identity, "is_error": failed, "output": text}]
            if item_type == "agent_message" and kind == "item.completed":
                if not self.turn_started or self.turn_completed or not isinstance(item.get("text"), str):
                    raise ValueError("Unexpected native assistant message")
                self.final_text = item["text"]
            elif item_type == "error":
                message = item.get("message", "")
                if (
                    isinstance(message, str)
                    and message.startswith("Model metadata for `")
                    and "fallback metadata" in message
                ):
                    self.metadata_fallback = True
                else:
                    self.failed = True
            elif item_type not in {"reasoning", "agent_message"}:
                raise ValueError("Unadmitted native action")
        else:
            raise ValueError("Unknown native event")
        return []


def configuration(ctx):
    p, task = ctx.profile, ctx.task
    if p.model.protocol == "responses":
        ResponsesPolicy.from_profile(p, task.tools)
    else:
        from .responses_bridge import validate_profile
        validate_profile(p)
    q = json.dumps
    lines = [
        f"model = {q(p.model.model)}",
        'model_provider = "venus_research"',
        f"model_context_window = {p.model.context_tokens}",
        'approval_policy = "never"',
        'sandbox_mode = "read-only"',
        'web_search = "disabled"',
    ]
    if p.budget.reasoning_effort is not None:
        lines.append(f"model_reasoning_effort = {q(p.budget.reasoning_effort)}")
    lines += [
        "[features]",
        "shell_tool = false",
        "shell_snapshot = false",
        "remote_plugin = false",
        "multi_agent = false",
        "goals = false",
        "hooks = false",
        "[analytics]",
        "enabled = false",
        "[feedback]",
        "enabled = false",
        "[model_providers.venus_research]",
        'name = "Venus research evidence gateway"',
        f"base_url = {q(ctx.relay.base_url)}",
        'wire_api = "responses"',
        'env_key = "VENUS_RELAY_TOKEN"',
        "requires_openai_auth = false",
        "supports_websockets = false",
        "request_max_retries = 0",
        "stream_max_retries = 0",
        f"stream_idle_timeout_ms = {int((p.budget.request_seconds + 5) * 1000)}",
    ]
    server = scientific_server(entry=task.mcp_entry, tools=task.tools)
    return "\n".join(lines) + "\n" + codex_mcp_configuration(server, p.budget.wall_seconds)


def sandbox_command(ctx, home):
    task, paths = ctx.task, ctx.runtimes
    binary = Path(paths["codex_bin"]).resolve()
    if binary.name != "codex" or binary.parent.name != "bin":
        raise ValueError("Codex runtime must retain the admitted package bin layout")
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
    from .system_mounts import system_mounts
    command += system_mounts()
    from agent.harness.python_runtime_mounts import python_runtime_mounts
    command += python_runtime_mounts(paths["python_root"])
    command += [
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
        str(binary.parent.parent),
        "/opt/codex",
        "--bind",
        str(task.workspace),
        "/workspace",
        "--ro-bind",
        str(task.capability),
        "/capabilities",
        "--bind",
        str(home),
        "/runtime_home",
        "--chdir",
        "/workspace",
        "--clearenv",
    ]
    env = {
        "HOME": "/runtime_home",
        "CODEX_HOME": "/runtime_home/codex",
        "PATH": "/opt/venus-runtime/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "VENUS_RELAY_TOKEN": ctx.relay.token,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    for key, value in env.items():
        command += ["--setenv", key, value]
    return command + [
        "--",
        "/opt/codex/bin/codex",
        "exec",
        "--json",
        "--ephemeral",
        "--strict-config",
        "--skip-git-repo-check",
        "--color",
        "never",
        "-",
    ]


def validate_tool_result(task, name, data):
    if task.phase == "chat" and task.mcp_entry == "web_tools.py":
        return

    from agent.rsi.engine.reproduction_tool_contracts import SUPPORTED, validate_reproduction_result
    from agent.rsi.engine.tool_contracts import validate_generated_result

    if name in SUPPORTED:
        validate_reproduction_result(name, data, workspace=task.workspace, capability=task.capability)
    elif task.mcp_entry == "generated_tools.py" and task.phase in {"develop", "reuse"}:
        validate_generated_result(name, data, workspace=task.workspace, capability=task.capability)


class CodexAdapter:
    def __init__(self, runtimes):
        self.runtimes = runtimes

    async def run(self, ctx):
        task = ctx.task
        home = task.output / "codex_home"
        (home / "codex").mkdir(parents=True)
        (home / "codex/config.toml").write_text(configuration(ctx))

        def validator(name, data):
            validate_tool_result(task, name, data)

        web_tools = task.phase == 'chat' and task.mcp_entry == 'web_tools.py'
        state = CodexEvents(task.tools, ctx.profile.budget.tool_calls, None if web_tools else validator)
        result = {
            "status": "running",
            "engine": "codex",
            "phase": task.phase,
            "mode": ctx.profile.mode,
            "human_interventions": 0,
            "model": ctx.profile.model.model,
        }
        started = time.monotonic()
        process = None
        try:
            with (
                (task.output / "stderr.log").open("xb") as stderr,
                (task.output / "native_events.jsonl").open("xb") as raw,
                (task.output / "events.jsonl").open("x") as trace,
            ):
                process = await asyncio.create_subprocess_exec(
                    *sandbox_command(ctx, home),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=stderr,
                    limit=8_000_000,
                    start_new_session=True,
                )
                process.stdin.write(task_message(task).encode())
                await process.stdin.drain()
                process.stdin.close()
                async with asyncio.timeout(ctx.profile.budget.wall_seconds):
                    while line := await process.stdout.readline():
                        raw.write(line)
                        raw.flush()
                        for event in state.consume(json.loads(line)):
                            trace.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                            trace.flush()
                    await process.wait()
            result["returncode"] = process.returncode
            result["status"] = (
                "completed"
                if process.returncode == 0 and state.turn_completed and not state.failed
                else "runtime_error"
            )
        except TimeoutError:
            result.update(status="timeout", error_type="TimeoutError")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result.update(status="protocol_failed", error_type=type(error).__name__)
        finally:
            if process is not None and process.returncode is None:
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
        result.update(
            session_id=state.thread_id,
            tool_calls=len(state.started),
            tool_errors=state.tool_errors,
            text=state.final_text,
            final_text=state.final_text,
            final_message=state.final_text,
            model_metadata_fallback=state.metadata_fallback,
            sandbox_stopped=process is None or process.returncode is not None,
            wall_seconds=round(time.monotonic() - started, 3),
        )
        return result
