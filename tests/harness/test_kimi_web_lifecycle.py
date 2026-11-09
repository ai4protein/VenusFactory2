import asyncio
import io
from types import SimpleNamespace

import pytest

from agent.harness.kimi_adapter import native_version
from agent.harness.kimi_web_sandbox import KimiWebSandbox


def sandbox(tmp_path, **overrides):
    args = dict(workspace=tmp_path / "work", control_dir=tmp_path / "control",
        capability_dir=tmp_path / "cap", kimi_bin=tmp_path / "kimi", config_path=tmp_path / "config",
        python_root=tmp_path / "python", expected_model={}, mcp_entry="generated_tools.py",
        relay=SimpleNamespace(token="fixture-relay"), startup_timeout_s=10,
        mcp_startup_timeout_s=10, mcp_tool_timeout_s=10, prepare_mcp=True, system_prompt="task instructions")
    return KimiWebSandbox(**{**args, **overrides})


@pytest.mark.parametrize("field", ["startup_timeout_s", "mcp_startup_timeout_s", "mcp_tool_timeout_s"])
@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_deadlines_cannot_disable_resource_limits(tmp_path, field, value):
    with pytest.raises(ValueError, match="timeout"):
        sandbox(tmp_path, **{field: value})


@pytest.mark.asyncio
async def test_credentials_removed_even_when_client_close_fails_before_token_was_read(tmp_path):
    instance = sandbox(tmp_path)
    root = instance.private_home / ".kimi-code"
    root.mkdir(parents=True)
    (root / "config.toml").write_text("fixture-relay")
    (root / "server.token").write_text("fixture-server")
    (instance.control_dir / "server.log").write_text("fixture-server fixture-relay")
    async def close(): raise RuntimeError("close failed")
    instance.client = SimpleNamespace(aclose=close)
    instance.log = io.BytesIO()
    with pytest.raises(RuntimeError, match="close failed"):
        await instance.stop()
    assert instance.log.closed
    assert not (root / "config.toml").exists() and not (root / "server.token").exists()
    assert (instance.control_dir / "server.log").read_text() == "[redacted] [redacted]"
    assert instance.stopped


@pytest.mark.asyncio
async def test_cancellation_waits_for_cleanup(tmp_path):
    instance = sandbox(tmp_path)
    reached, release = asyncio.Event(), asyncio.Event()
    worker = SimpleNamespace(stopped=False)
    async def stop():
        reached.set(); await release.wait(); worker.stopped = True
    worker.stop = stop
    instance.prepared_mcp = worker
    cleanup = asyncio.create_task(instance.stop())
    await reached.wait()
    cleanup.cancel()
    await asyncio.sleep(0)
    assert not cleanup.done()
    release.set()
    with pytest.raises(asyncio.CancelledError): await cleanup
    assert worker.stopped and instance.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", ["kimi 9.9.9", "unrelated 2.5.2"])
async def test_unknown_native_versions_are_rejected(tmp_path, reported):
    binary = tmp_path / "kimi"
    binary.write_text("#!/bin/sh\nprintf '%s\\n' '" + reported + "'\n")
    binary.chmod(0o700)
    with pytest.raises(ValueError, match="Unsupported"):
        await native_version(binary, tmp_path)
    assert not list(tmp_path.glob("kimi-version-*"))


@pytest.mark.asyncio
async def test_adapter_cleans_configuration_if_sandbox_construction_fails(tmp_path, monkeypatch):
    from agent.harness import Budget, ModelBinding, Profile, Task
    from agent.harness.host import RuntimeContext
    from agent.harness import kimi_adapter
    async def version(*_): return "2.5.2"
    monkeypatch.setattr(kimi_adapter, "native_version", version)
    task = Task(tmp_path / "work", tmp_path / "cap", tmp_path, "question", "",
        frozenset({"probe"}), mcp_entry="generated_tools.py", phase="chat")
    profile = Profile("science_agent", "kimi-code", ModelBinding("fixture", "fixture", "https://example.org/v1"), Budget())
    relay = SimpleNamespace(base_url="http://127.0.0.1:12345/v1", token="fixture-relay")
    runtimes = {"kimi_bin": str(tmp_path / "kimi"), "python_root": str(tmp_path / "python")}
    with pytest.raises(ValueError, match="explicit system instructions"):
        await kimi_adapter.KimiAdapter(runtimes).run(RuntimeContext(profile, task, relay, runtimes))
    assert not (tmp_path / "runtime-private.toml").exists()


@pytest.mark.asyncio
async def test_adapter_does_not_overwrite_existing_private_configuration(tmp_path, monkeypatch):
    from agent.harness import Budget, ModelBinding, Profile, Task
    from agent.harness.host import RuntimeContext
    from agent.harness import kimi_adapter
    async def version(*_): return "2.5.2"
    monkeypatch.setattr(kimi_adapter, "native_version", version)
    config = tmp_path / "runtime-private.toml"
    config.write_text("existing configuration")
    task = Task(tmp_path / "work", tmp_path / "cap", tmp_path, "question", "instructions",
        frozenset({"probe"}), mcp_entry="generated_tools.py", phase="chat")
    profile = Profile("science_agent", "kimi-code", ModelBinding("fixture", "fixture", "https://example.org/v1"), Budget())
    relay = SimpleNamespace(base_url="http://127.0.0.1:12345/v1", token="fixture-relay")
    runtimes = {"kimi_bin": str(tmp_path / "kimi"), "python_root": str(tmp_path / "python")}
    with pytest.raises(FileExistsError):
        await kimi_adapter.KimiAdapter(runtimes).run(RuntimeContext(profile, task, relay, runtimes))
    assert config.read_text() == "existing configuration"
