import asyncio
import os
import sys
from pathlib import Path

import pytest

from agent.rsi.engine.sandbox import SandboxKimi, bwrap_command


@pytest.mark.asyncio
async def test_pid_namespace_stops_session_detached_child(tmp_path):
    import shutil

    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap unavailable")
    kimi = Path(os.environ.get("KIMI_CLI_PATH") or shutil.which("kimi") or Path.home() / ".kimi-code/bin/kimi")
    if not kimi.exists():
        pytest.skip("host integration fixture requires Kimi binary")
    work = tmp_path / "workspace"
    work.mkdir()
    control = tmp_path / "control"
    control.mkdir()
    private = control / "private_home"
    private.mkdir()
    capabilities = tmp_path / "capabilities"
    capabilities.mkdir()
    runtime = Path(sys.prefix)
    child = "import time; from pathlib import Path; p=Path('/workspace/heartbeat'); exec('while True:\\n p.write_text(str(time.time()))\\n time.sleep(0.03)')"
    parent = f"import subprocess,time; subprocess.Popen(['/opt/venus-runtime/bin/python','-c',{child!r}],start_new_session=True); time.sleep(100)"
    cmd = bwrap_command(work, private, capabilities, kimi, runtime, 0)
    cmd = cmd[: cmd.index("--") + 1] + ["/opt/venus-runtime/bin/python", "-c", parent]
    sandbox = SandboxKimi(
        workspace=work,
        control_dir=control,
        capability_dir=capabilities,
        kimi_bin=kimi,
        config_path=tmp_path / "unused",
        python_root=runtime,
    )
    try:
        sandbox.proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, start_new_session=True
        )
        async with asyncio.timeout(5):
            while not (work / "heartbeat").exists():
                if sandbox.proc.returncode is not None:
                    pytest.fail((await sandbox.proc.stderr.read()).decode())
                await asyncio.sleep(0.03)
    finally:
        await sandbox.stop()
    heartbeat = work / "heartbeat"
    before = heartbeat.read_text()
    await asyncio.sleep(0.15)
    assert heartbeat.read_text() == before
    assert sandbox.stopped


@pytest.mark.asyncio
async def test_spawn_failure_removes_private_credential_copy(tmp_path, monkeypatch):
    config = tmp_path / "source_config.toml"
    config.write_text('default_model = "fixture"\n')
    sandbox = SandboxKimi(
        workspace=tmp_path / "workspace",
        control_dir=tmp_path / "control",
        capability_dir=tmp_path / "capabilities",
        kimi_bin=tmp_path / "kimi",
        config_path=config,
        python_root=Path(sys.prefix),
    )

    async def failed_spawn(*args, **kwargs):
        raise OSError("simulated process creation failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", failed_spawn)
    with pytest.raises(OSError, match="simulated"):
        await sandbox.start()
    assert not (sandbox.control_dir / "private_home/.kimi-code/config.toml").exists()
    assert config.exists()
    assert sandbox.log.closed


@pytest.mark.asyncio
async def test_native_kimi_keeps_admitted_output_and_mcp_timeouts(tmp_path, monkeypatch):
    import json
    import tomllib

    config = tmp_path / "source.toml"
    config.write_text('default_model = "research"\n[models.research]\nmax_output_size = 12288\n')
    sandbox = SandboxKimi(
        workspace=tmp_path / "work", control_dir=tmp_path / "control",
        capability_dir=tmp_path / "capabilities", kimi_bin=tmp_path / "kimi",
        config_path=config, python_root=Path(sys.prefix),
        mcp_startup_timeout_s=30, mcp_tool_timeout_s=210,
    )

    async def inspect_spawn(*args, **kwargs):
        private = sandbox.control_dir / "private_home/.kimi-code"
        assert tomllib.loads((private / "config.toml").read_text())["models"]["research"]["max_output_size"] == 12288
        server = json.loads((private / "mcp.json").read_text())["mcpServers"]["venusfactory"]
        assert server["startupTimeoutMs"] == 30000
        assert server["toolTimeoutMs"] == 210000
        raise OSError("inspected without launching")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", inspect_spawn)
    with pytest.raises(OSError, match="inspected"):
        await sandbox.start()
    assert not (sandbox.control_dir / "private_home/.kimi-code/config.toml").exists()


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan"), "120"])
def test_kimi_rejects_invalid_mcp_timeouts(tmp_path, timeout):
    with pytest.raises(ValueError, match="MCP tool timeout"):
        SandboxKimi(
            workspace=tmp_path, control_dir=tmp_path / "control",
            capability_dir=tmp_path, kimi_bin=tmp_path / "kimi",
            config_path=tmp_path / "config", python_root=Path(sys.prefix),
            mcp_tool_timeout_s=timeout,
        )


@pytest.mark.asyncio
async def test_native_cache_is_disposable_and_workspace_and_session_state_persist(tmp_path):
    import json
    import shutil

    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap unavailable")
    work, private, capabilities = [tmp_path / name for name in ("workspace", "home", "capabilities")]
    for path in (work, private, capabilities):
        path.mkdir()
    command = bwrap_command(work, private, capabilities, Path(sys.executable), Path(sys.prefix), 0)
    prefix = command[: command.index("--") + 1]
    script = (
        "import json,os; from pathlib import Path; "
        "cache=Path(os.environ['XDG_CACHE_HOME']); "
        "assert cache == Path('/tmp/cache') and cache.is_dir(); "
        "assert not (cache/'web-asset.js').exists(); "
        "(cache/'web-asset.js').write_text('disposable'); "
        "Path('/workspace/model.py').write_text('model artifact'); "
        "state=Path.home()/'.kimi-code'; state.mkdir(exist_ok=True); "
        "(state/'session.json').write_text(json.dumps({'turns': 1})); "
        "print('isolated cache')"
    )
    # Separate namespaces must never reuse cache content, even with the same HOME.
    for _ in range(2):
        process = await asyncio.create_subprocess_exec(
            *prefix, "/opt/venus-runtime/bin/python", "-c", script,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert process.returncode == 0, stderr.decode()
        assert stdout.decode().strip() == "isolated cache"
        assert not (private / ".cache").exists()
        assert (work / "model.py").read_text() == "model artifact"
        assert json.loads((private / ".kimi-code/session.json").read_text()) == {"turns": 1}
