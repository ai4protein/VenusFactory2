"""Ephemeral authenticated Kimi Web server in the existing filesystem boundary."""
from __future__ import annotations

import asyncio
import json
import math
import os
import signal
from pathlib import Path

from agent.rsi.engine.model_policy import read_model_configuration
from agent.rsi.engine.sandbox import bwrap_command, free_port
from .kimi_web_client import KimiWebClient
from .prepared_mcp import PreparedMcp


class KimiWebSandbox:
    def __init__(self, *, workspace, control_dir, capability_dir, kimi_bin, config_path,
                 python_root, expected_model, mcp_entry, relay, startup_timeout_s,
                 mcp_startup_timeout_s, mcp_tool_timeout_s, prepare_mcp, system_prompt):
        if prepare_mcp is not True or not system_prompt.strip():
            raise ValueError("Kimi Web requires prepared MCP and explicit system instructions")
        if mcp_entry not in {"pilot_tools.py", "protein_tools.py", "generated_tools.py", "web_tools.py"}:
            raise ValueError("Unsupported MCP entry")
        for value in (startup_timeout_s, mcp_startup_timeout_s, mcp_tool_timeout_s):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid Kimi Web timeout")
        self.workspace, self.control_dir = Path(workspace), Path(control_dir)
        self.capability, self.python_root, self.binary = Path(capability_dir), Path(python_root), Path(kimi_bin)
        self.config_path, self.expected_model, self.relay = config_path, expected_model, relay
        self.entry, self.system_prompt = mcp_entry, system_prompt
        self.startup_timeout_s = startup_timeout_s
        self.mcp_startup_timeout_s, self.mcp_tool_timeout_s = mcp_startup_timeout_s, mcp_tool_timeout_s
        self.port = free_port()
        self.private_home = self.control_dir / "private_home"
        self.client = self.proc = self.log = self.prepared_mcp = None
        self.capability_preparation = None
        self.startup_stage = "not_started"
        self.stopped = False
        self._token = ""

    async def start(self):
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            config_root = self.private_home / ".kimi-code"
            config_root.mkdir(parents=True, mode=0o700)
            self.private_home.chmod(0o700)
            config, _ = read_model_configuration(self.config_path, expected=self.expected_model)
            (config_root / "config.toml").write_text(config)
            (config_root / "config.toml").chmod(0o600)
            (config_root / "SYSTEM.md").write_text(self.system_prompt)
            self.startup_stage = "capability_loading"
            self.prepared_mcp = PreparedMcp(workspace=self.workspace, capability=self.capability,
                python_root=self.python_root, control_dir=self.control_dir / "capability_worker",
                entry=self.entry, timeout_s=self.mcp_startup_timeout_s)
            self.capability_preparation = await self.prepared_mcp.start()
            connection = self.prepared_mcp.connection()
            connection.update(startupTimeoutMs=max(1, int(self.mcp_startup_timeout_s * 1000)),
                              toolTimeoutMs=max(1, int(self.mcp_tool_timeout_s * 1000)))
            (config_root / "mcp.json").write_text(json.dumps({"mcpServers": {"venusfactory": connection}}))
            command = bwrap_command(self.workspace, self.private_home, self.capability,
                self.binary, self.python_root, self.port, prepared_mcp=self.prepared_mcp)
            # Share the audited mounts and environment, replacing only the native CLI.
            command = command[:command.index("--") + 1] + ["/opt/kimi/kimi", "web", "--no-open",
                "--port", str(self.port), "--log-level", "warn"]
            self.log = (self.control_dir / "server.log").open("wb")
            self.startup_stage = "server_startup"
            self.proc = await asyncio.create_subprocess_exec(*command, stdout=self.log,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            async with asyncio.timeout(self.startup_timeout_s):
                while True:
                    if self.proc.returncode is not None:
                        raise RuntimeError("Isolated Kimi Web server exited during startup")
                    token_file = config_root / "server.token"
                    if self.client is None and token_file.is_file():
                        self._token = token_file.read_text().strip()
                        if self._token:
                            self.client = KimiWebClient(f"http://127.0.0.1:{self.port}", token=self._token)
                    if self.client is not None and await self.client.healthz():
                        response = await self.client._http.get("/api/v1/meta")
                        response.raise_for_status()
                        if response.json().get("data", {}).get("server_version") != "2.5.2":
                            raise RuntimeError("Kimi Web runtime version differs from the verified protocol")
                        if response.json()["data"].get("dangerous_bypass_auth") is not False:
                            raise RuntimeError("Kimi Web authentication must remain enabled")
                        self.startup_stage = "server_ready"
                        return self.client
                    await asyncio.sleep(0.2)
        except BaseException:
            await self.stop()
            raise

    async def stop(self):
        cleanup = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _stop(self):
        try:
            if self.proc is not None:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.proc.wait(), 4)
                except TimeoutError:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await self.proc.wait()
        finally:
            try:
                if self.prepared_mcp is not None:
                    await self.prepared_mcp.stop()
            finally:
                try:
                    if self.client is not None:
                        await self.client.aclose()
                finally:
                    self._close_private_state()
                    self.stopped = (self.proc is None or self.proc.returncode is not None) and (
                        self.prepared_mcp is None or self.prepared_mcp.stopped)

    def _close_private_state(self):
        if self.log is not None:
            self.log.close()
        config_root = self.private_home / ".kimi-code"
        token_file = config_root / "server.token"
        tokens = {self._token, self.relay.token}
        try:
            # Startup may be cancelled before the token was read into memory.
            if token_file.is_file():
                tokens.add(token_file.read_text().strip())
            log = self.control_dir / "server.log"
            if log.is_file():
                text = log.read_text(errors="replace")
                for token in tokens - {""}:
                    text = text.replace(token, "[redacted]")
                log.write_text(text)
        finally:
            (config_root / "config.toml").unlink(missing_ok=True)
            token_file.unlink(missing_ok=True)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()
