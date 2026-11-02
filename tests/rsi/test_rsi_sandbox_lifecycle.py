import asyncio
import sys
from pathlib import Path

import pytest

from agent.rsi.sandbox import SandboxKimi, bwrap_command


@pytest.mark.asyncio
async def test_pid_namespace_stops_session_detached_child(tmp_path):
    import shutil

    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap unavailable")
    kimi = Path("/home/tanyang/.kimi-code/bin/kimi")
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
