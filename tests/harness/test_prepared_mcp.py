import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from agent.harness import prepared_mcp
from agent.harness.prepared_mcp import PreparedMcp, check_entry

VALID = """from fastmcp import FastMCP
mcp = FastMCP("fixture")
@mcp.tool()
def protein_probe(sequence: str) -> dict:
    from pathlib import Path
    import json
    import sys
    assert sys.modules[__name__].protein_probe is protein_probe
    result = {"sequence":sequence,"length":len(sequence),"cysteines":sequence.count("C")}
    Path("/workspace/observed.json").write_text(json.dumps(result))
    return result
if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
"""


def fixture(tmp_path, source=VALID, timeout=180):
    if not shutil.which("bwrap"):
        pytest.skip("Linux bubblewrap integration required")
    work = tmp_path / "workspace"
    work.mkdir()
    cap = tmp_path / "capability"
    cap.mkdir()
    (cap / "generated_tools.py").write_text(source)
    return PreparedMcp(
        workspace=work,
        capability=cap,
        python_root=Path(sys.prefix),
        control_dir=tmp_path / "control",
        entry="generated_tools.py",
        timeout_s=timeout,
    )


def test_declared_entry_hash_and_main_behavior(tmp_path):
    path = tmp_path / "generated_tools.py"
    path.write_text(VALID)
    assert check_entry(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(VALID.replace('mcp.run(transport="stdio", show_banner=False)', "mcp.run()"))
    assert check_entry(path)


@pytest.mark.parametrize(
    "body",
    [
        "main()",
        'mcp.run(transport="http")',
        'mcp.run("stdio")',
        "mcp.run(show_banner=True)",
        'mcp.run(); print("extra behavior")',
        "mcp.run(**config)",
        "mcp.run(custom_behavior=True)",
    ],
)
def test_unmatched_main_contract_is_not_silently_rewritten(tmp_path, body):
    path = tmp_path / "generated_tools.py"
    path.write_text('if __name__ == "__main__":\n    ' + body + "\n")
    with pytest.raises(ValueError):
        check_entry(path)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_load_budget_rejected(tmp_path, timeout):
    with pytest.raises(ValueError):
        PreparedMcp(
            workspace=tmp_path,
            capability=tmp_path,
            python_root=Path(sys.prefix),
            control_dir=tmp_path / "control",
            entry="generated_tools.py",
            timeout_s=timeout,
        )


@pytest.mark.asyncio
async def test_prepared_worker_retains_real_mcp_handshake_and_tool_result(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    worker = fixture(tmp_path)
    try:
        ready = await worker.start()
        assert ready["stage"] == "capability_loaded" and ready["native_handshake_completed"] is False
        assert not (worker.workspace / "observed.json").exists()
        connection = worker.connection()
        assert connection["args"][-2] == "relay"
        socket_path = worker.ipc_root / "worker.sock"
        server = StdioServerParameters(
            command=sys.executable, args=["-u", str(Path(prepared_mcp.__file__)), "relay", str(socket_path)]
        )
        async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            assert [tool.name for tool in (await session.list_tools()).tools] == ["protein_probe"]
            assert not socket_path.exists(), "Only one native connection may claim a prepared worker"
            result = await session.call_tool("protein_probe", {"sequence": "ACDEFGHIKC"})
            assert not result.isError
            assert json.loads(result.content[0].text) == {"sequence": "ACDEFGHIKC", "length": 10, "cysteines": 2}
        assert json.loads((worker.workspace / "observed.json").read_text())["cysteines"] == 2
        assert json.loads((worker.control_dir / "connection.json").read_text())["connections"] == 1
    finally:
        await worker.stop()
    assert worker.stopped and worker.process.returncode is not None and not worker.ipc_root.exists()


@pytest.mark.asyncio
async def test_worker_import_failure_cleans_private_socket_and_process(tmp_path):
    worker = fixture(tmp_path, 'raise RuntimeError("fixture import failure")\n' + VALID)
    with pytest.raises(RuntimeError):
        await worker.start()
    assert worker.stopped and worker.process.returncode is not None and not worker.ipc_root.exists()
    assert json.loads((worker.control_dir / "host_failure.json").read_text())["stage"] == "capability_loading"
    assert not (worker.control_dir / "ready.json").exists()


@pytest.mark.asyncio
async def test_cancelled_loading_stops_session_detached_descendant(tmp_path):
    child = "from pathlib import Path; import time; p=Path('/workspace/heartbeat'); exec('while True:\\n p.write_text(str(time.time()))\\n time.sleep(.02)')"
    source = (
        'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",'
        + repr(child)
        + "],start_new_session=True)\ntime.sleep(60)\n"
        + VALID
    )
    worker = fixture(tmp_path, source)
    task = asyncio.create_task(worker.start())
    try:
        async with asyncio.timeout(15):
            while not (worker.workspace / "heartbeat").exists():
                if task.done():
                    task.result()
                await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await worker.stop()
    heartbeat = (worker.workspace / "heartbeat").read_text()
    await asyncio.sleep(0.15)
    assert (worker.workspace / "heartbeat").read_text() == heartbeat
    assert worker.stopped and not worker.ipc_root.exists()


@pytest.mark.asyncio
async def test_load_deadline_stops_before_mcp_can_be_exposed(tmp_path):
    worker = fixture(tmp_path, "import time\ntime.sleep(60)\n" + VALID, timeout=0.1)
    with pytest.raises(TimeoutError):
        await worker.start()
    assert worker.stopped and worker.process.returncode is not None and not worker.ipc_root.exists()
    assert not (worker.control_dir / "ready.json").exists()


@pytest.mark.asyncio
async def test_changed_entry_never_crosses_host_readiness_gate(tmp_path):
    worker = fixture(tmp_path)
    (worker.capability / "generated_tools.py").write_text(VALID + "\n# altered after binding\n")
    with pytest.raises(ValueError, match="binding"):
        await worker.start()
    assert worker.stopped and not worker.ipc_root.exists()
    assert not (worker.control_dir / "connection.json").exists()


@pytest.mark.asyncio
async def test_stdio_relay_preserves_binary_chunks_and_eof(tmp_path):
    path = tmp_path / "echo.sock"

    async def echo(reader, writer):
        while data := await reader.read(7919):
            writer.write(data)
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(echo, path=str(path))
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(prepared_mcp.__file__)),
            "relay",
            str(path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        payload = bytes(range(256)) * 1025 + b'\n{"id":7,"result":{"count":2}}\n'
        stdout, stderr = await asyncio.wait_for(process.communicate(payload), 5)
        assert process.returncode == 0 and stdout == payload and not stderr
    finally:
        server.close()
        await server.wait_closed()
