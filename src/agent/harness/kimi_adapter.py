"""Versioned Kimi adapter; legacy research runtime remains source-stable."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import tempfile
import time
from pathlib import Path


async def native_version(binary, output):
    """Probe only the configured executable, with an empty private home."""
    with tempfile.TemporaryDirectory(prefix="kimi-version-", dir=output) as folder:
        process = await asyncio.create_subprocess_exec(str(binary), "--version", cwd=folder,
            env={"HOME": folder, "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), 10)
            match = re.fullmatch(r"(?:kimi(?:-code)?\s+)?(\d+\.\d+\.\d+)\s*", stdout.decode().strip())
            if process.returncode or not match or match[1] not in {"0.19.2", "2.5.2"}:
                raise ValueError("Unsupported Kimi runtime version")
            return match[1]
        finally:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()


class KimiAdapter:
    def __init__(self, runtimes):
        self.runtimes = runtimes

    async def run(self, ctx):
        from agent.rsi.engine.model_policy import read_model_configuration
        from agent.rsi.engine.runtime import run_task
        from agent.rsi.engine.sandbox import SandboxKimi

        p, t = ctx.profile, ctx.task
        version = await native_version(self.runtimes["kimi_bin"], t.output)
        web_protocol = version == "2.5.2"
        if web_protocol:
            from .kimi_web_sandbox import KimiWebSandbox
            sandbox_class = KimiWebSandbox
        else:
            sandbox_class = SandboxKimi
        if t.mcp_entry not in {"pilot_tools.py", "protein_tools.py", "generated_tools.py", "web_tools.py"}:
            raise ValueError("Unsupported MCP entry for this pinned Kimi sandbox adapter")
        cfg = t.output / "runtime-private.toml"
        # The child receives only a short-lived local gateway token, never the upstream key.
        q = json.dumps
        loop_control = ""
        if t.phase == "chat" and t.mcp_entry == "web_tools.py":
            loop_control = (
                "[loop_control]\nmax_retries_per_step = 1\n"
                + "max_steps_per_turn = " + str(p.budget.model_calls) + "\n"
            )
        cfg.touch(mode=0o600, exist_ok=False)
        try:
            cfg.write_text(
                'default_model = "research"\n'
                + loop_control
                + '[models.research]\nprovider = "research"\n'
                + "model = "
                + q(p.model.model)
                + "\nmax_context_size = "
                + str(p.model.context_tokens)
                + "\nmax_output_size = "
                + str(p.budget.output_tokens)
                + '\ncapabilities = ["tool_use"]\n[providers.research]\ntype = "openai"\n'
                + "base_url = "
                + q(ctx.relay.base_url)
                + "\napi_key = "
                + q(ctx.relay.token)
                + "\n"
            )
            _, identity = read_model_configuration(cfg)
            started = time.monotonic()
            startup_limit = min(180.0, p.budget.wall_seconds)
            sandbox = sandbox_class(
                workspace=t.workspace,
                control_dir=t.output / "control",
                capability_dir=t.capability,
                kimi_bin=Path(self.runtimes["kimi_bin"]),
                config_path=cfg,
                python_root=Path(self.runtimes["python_root"]),
                expected_model=identity,
                mcp_entry=t.mcp_entry,
                relay=ctx.relay,
                startup_timeout_s=startup_limit,
                mcp_startup_timeout_s=startup_limit,
                mcp_tool_timeout_s=p.budget.wall_seconds,
                prepare_mcp=True,
                **({"system_prompt": t.instructions} if web_protocol else {}),
            )
            stage = "server_startup"
            async with sandbox:
                stage = "phase_deadline"
                remaining = p.budget.wall_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Native phase deadline exhausted during startup")
                result = await run_task(
                    sandbox.client,
                    prompt=t.prompt,
                    system_prompt="" if web_protocol else t.instructions,
                    model="research",
                    output_dir=t.output,
                    timeout_s=remaining,
                    mcp_startup_timeout_s=min(startup_limit, remaining),
                    max_tool_calls=p.budget.tool_calls,
                    allowed_mcp_tools=t.tools,
                    request_gateway=True,
                    interaction="chat" if t.phase == "chat" else "bounded",
                )
                if web_protocol and result.get("session_id"):
                    try:
                        async with asyncio.timeout(3):
                            usage = await sandbox.client.usage_rollup(result["session_id"])
                        if usage is not None:
                            result["runtime_session_usage"] = usage
                            # The SDK can persist its rollup after turn completion.
                            # A still-empty snapshot must not erase streamed usage.
                            if usage.get("input_tokens", 0) or usage.get("output_tokens", 0):
                                result["usage"] = usage
                                result["usage_status"] = "runtime_session_rollup; provider billing not independently verified"
                    except Exception as exc:
                        result["usage_rollup_error_type"] = type(exc).__name__
            result["native_version"] = version
            if web_protocol:
                result["native_prompt_retained"] = False
                result["instruction_binding"] = "private SYSTEM.md"
                result["system_prompt_sha256"] = hashlib.sha256(t.instructions.encode()).hexdigest()
            result["sandbox_stopped"] = sandbox.stopped
            result["server_startup_timeout_s"] = startup_limit
            result["capability_preparation"] = sandbox.capability_preparation
            return result
        except TimeoutError:
            return {
                "status": "timeout",
                "timeout_stage": sandbox.startup_stage if stage == "server_startup" else stage,
                "tool_calls": 0,
                "sandbox_stopped": sandbox.stopped,
                "server_startup_timeout_s": startup_limit,
                "wall_seconds": round(time.monotonic() - started, 3),
            }
        finally:
            cfg.unlink(missing_ok=True)
