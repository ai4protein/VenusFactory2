"""Dedicated Kimi process with a private filesystem and PID namespace.

Network is shared to reach the configured provider. This is not a network sandbox.
Only the Kimi binary, Python environment, capability package and task workspace
are mounted; the host repository, experiment results and other sessions are not.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import signal
import socket
from pathlib import Path

from agent.kimi_client import KimiClient

from .model_policy import read_model_configuration


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def bwrap_command(
    workspace: Path,
    private_home: Path,
    capability_dir: Path,
    kimi_bin: Path,
    python_root: Path,
    port: int,
    *,
    prepared_mcp=None,
) -> list[str]:
    cmd = [
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
    ]
    from agent.harness.system_mounts import system_mounts
    from agent.harness.python_runtime_mounts import python_runtime_mounts
    cmd += system_mounts()
    cmd += python_runtime_mounts(python_root)
    cmd += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/cache",
        "--tmpfs",
        "/home",
        "--dir",
        "/opt",
        "--dir",
        "/opt/kimi",
        "--ro-bind",
        str(kimi_bin.resolve()),
        "/opt/kimi/kimi",
        "--bind",
        str(workspace.resolve()),
        "/workspace",
        "--bind",
        str(private_home.resolve()),
        "/runtime_home",
        "--ro-bind",
        str(capability_dir.resolve()),
        "/capabilities",
    ]
    if prepared_mcp is not None:
        cmd += prepared_mcp.mounts()
    dns = Path("/run/systemd/resolve")
    if dns.exists():
        cmd += ["--ro-bind", str(dns), str(dns)]
    cmd += [
        "--chdir",
        "/workspace",
        "--clearenv",
        "--setenv",
        "HOME",
        "/runtime_home",
        # Kimi extracts its bundled web assets into the XDG cache even in server
        # mode. Keep these reproducible assets out of persistent research evidence;
        # /tmp already belongs to this namespace and disappears with its processes.
        "--setenv",
        "XDG_CACHE_HOME",
        "/tmp/cache",
        "--setenv",
        "PATH",
        "/opt/venus-runtime/bin:/usr/bin:/bin",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "VENUS_RSI_WORKSPACE",
        "/workspace",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--",
        "/opt/kimi/kimi",
        "server",
        "run",
        "--port",
        str(port),
        "--foreground",
        "--log-level",
        "warn",
    ]
    return cmd


class SandboxKimi:
    def __init__(
        self,
        *,
        workspace: Path,
        control_dir: Path,
        capability_dir: Path,
        kimi_bin: Path,
        config_path: Path,
        python_root: Path,
        expected_model: dict | None = None,
        mcp_entry: str = "pilot_tools.py",
        relay=None,
        startup_timeout_s: float = 35,
        mcp_startup_timeout_s: float | None = None,
        mcp_tool_timeout_s: float | None = None,
        prepare_mcp: bool = False,
    ):
        if type(prepare_mcp) is not bool:
            raise ValueError("Invalid capability preparation option")
        self.prepare_mcp = prepare_mcp
        self.prepared_mcp = None
        self.capability_preparation = None
        self.startup_stage = "not_started"
        self.workspace, self.control_dir, self.capability_dir = workspace, control_dir, capability_dir
        self.kimi_bin, self.config_path, self.python_root = kimi_bin, config_path, python_root
        self.expected_model = expected_model
        if mcp_entry not in {"pilot_tools.py", "protein_tools.py", "generated_tools.py", "web_tools.py"}:
            raise ValueError("unsupported research MCP entry point")
        self.mcp_entry = mcp_entry
        if (
            type(startup_timeout_s) not in (int, float)
            or not math.isfinite(startup_timeout_s)
            or startup_timeout_s <= 0
        ):
            raise ValueError("Invalid server startup timeout")
        if mcp_startup_timeout_s is not None and (
            type(mcp_startup_timeout_s) not in (int, float)
            or not math.isfinite(mcp_startup_timeout_s)
            or not 0.001 <= mcp_startup_timeout_s <= 2147483.647
        ):
            raise ValueError("Invalid native MCP startup timeout")
        if mcp_tool_timeout_s is not None and (
            type(mcp_tool_timeout_s) not in (int, float)
            or not math.isfinite(mcp_tool_timeout_s)
            or not 0.001 <= mcp_tool_timeout_s <= 2147483.647
        ):
            raise ValueError("Invalid native MCP tool timeout")
        self.mcp_tool_timeout_s = mcp_tool_timeout_s
        self.mcp_startup_timeout_s = mcp_startup_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.relay = relay
        self.port = free_port()
        self.client = KimiClient(f"http://127.0.0.1:{self.port}")
        self.proc = None
        self.log = None
        self.stopped = False

    async def start(self) -> KimiClient:
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            self.control_dir.mkdir(parents=True, exist_ok=True)
            private_home = self.control_dir / "private_home"
            config_root = private_home / ".kimi-code"
            config_root.mkdir(parents=True, mode=0o700)
            private_home.chmod(0o700)
            config = config_root / "config.toml"
            if self.expected_model is None:
                config_text = self.config_path.read_text()
            else:
                config_text, _ = read_model_configuration(self.config_path, expected=self.expected_model)
            if self.relay is not None:
                import tomllib

                import toml

                data = tomllib.loads(config_text)
                selected = data["models"][data["default_model"]]
                provider = data["providers"][selected["provider"]]
                provider["base_url"] = self.relay.base_url
                provider["api_key"] = self.relay.token
                config_text = toml.dumps(data)
            config.write_text(config_text)
            config.chmod(0o600)
            if self.prepare_mcp:
                from agent.harness.prepared_mcp import PreparedMcp

                self.startup_stage = "capability_loading"
                self.prepared_mcp = PreparedMcp(
                    workspace=self.workspace,
                    capability=self.capability_dir,
                    python_root=self.python_root,
                    control_dir=self.control_dir / "capability_worker",
                    entry=self.mcp_entry,
                    timeout_s=self.mcp_startup_timeout_s or self.startup_timeout_s,
                )
                self.capability_preparation = await self.prepared_mcp.start()
            mcp = {
                "mcpServers": {
                    "venusfactory": {
                        "command": "/opt/venus-runtime/bin/python",
                        "args": ["-u", f"/capabilities/{self.mcp_entry}"],
                    }
                }
            }
            if self.prepared_mcp is not None:
                mcp["mcpServers"]["venusfactory"] = self.prepared_mcp.connection()
            if self.mcp_startup_timeout_s is not None:
                mcp["mcpServers"]["venusfactory"]["startupTimeoutMs"] = max(1, int(self.mcp_startup_timeout_s * 1000))
            if self.mcp_tool_timeout_s is not None:
                mcp["mcpServers"]["venusfactory"]["toolTimeoutMs"] = max(1, int(self.mcp_tool_timeout_s * 1000))
            (config_root / "mcp.json").write_text(json.dumps(mcp))
            self.log = (self.control_dir / "server.log").open("wb")
            self.startup_stage = "server_startup"
            self.proc = await asyncio.create_subprocess_exec(
                *bwrap_command(
                    self.workspace,
                    private_home,
                    self.capability_dir,
                    self.kimi_bin,
                    self.python_root,
                    self.port,
                    prepared_mcp=self.prepared_mcp,
                ),
                stdout=self.log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            async with asyncio.timeout(self.startup_timeout_s):
                while True:
                    if self.proc.returncode is not None:
                        raise RuntimeError(f"isolated Kimi exited with code {self.proc.returncode}")
                    if await self.client.healthz():
                        self.startup_stage = "server_ready"
                        return self.client
                    await asyncio.sleep(0.3)
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        try:
            if self.proc is not None:
                # Kill the dedicated bubblewrap process group even if its leader
                # has already exited; the PID namespace also reaps descendants.
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=4)
                except TimeoutError:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await self.proc.wait()
                self.stopped = True
        finally:
            try:
                if self.prepared_mcp is not None:
                    await self.prepared_mcp.stop()
            finally:
                await self._close_private_state()

    async def _close_private_state(self):
        try:
            await self.client.aclose()
        finally:
            if self.log is not None:
                self.log.close()
            # Remove only the credential copy created for this runtime.
            config = self.control_dir / "private_home/.kimi-code/config.toml"
            if config.exists():
                config.unlink()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()
