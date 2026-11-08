"""Load a scientific FastMCP capability before the native initialization deadline.

The native client still initializes, discovers and calls the actual MCP server.
The relay forwards bytes; it never invents protocol responses or tool receipts.
This module is also executed as a stdlib-only sandbox supervisor and relay.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

SCHEMA = "prepared_scientific_mcp.v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def check_entry(path):
    """Only the repository's global FastMCP entry contract can be preloaded."""
    path = Path(path)
    if (
        path.name not in {"generated_tools.py", "protein_tools.py", "pilot_tools.py", "web_tools.py"}
        or path.is_symlink()
    ):
        raise ValueError("Unsupported prepared capability entry")
    tree = ast.parse(path.read_text())
    guards = []
    for node in tree.body:
        if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(
            ast.parse('__name__ == "__main__"', mode="eval").body
        ):
            guards.append(node)
    if len(guards) != 1 or guards[0].orelse or len(guards[0].body) != 1:
        raise ValueError("Capability must have one unmodified global FastMCP entry")
    expr = guards[0].body[0]
    call = expr.value if isinstance(expr, ast.Expr) else None
    if path.name == "web_tools.py":
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "main"
            or call.args
            or call.keywords
        ):
            raise ValueError("Invalid Web MCP entry")
        return sha256(path)
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "mcp"
        or call.func.attr != "run"
        or call.args
    ):
        raise ValueError("Capability entry has additional executable main behavior")
    allowed = {"transport": "stdio", "show_banner": False}
    seen = set()
    for keyword in call.keywords:
        if (
            keyword.arg not in allowed
            or keyword.arg in seen
            or not isinstance(keyword.value, ast.Constant)
            or keyword.value.value != allowed[keyword.arg]
        ):
            raise ValueError("Unsupported capability startup arguments")
        seen.add(keyword.arg)
    return sha256(path)


def worker(entry, ready_fd):
    expected = check_entry(entry)
    sys.path.insert(0, str(Path(entry).resolve().parent))
    name = "venus_prepared_capability"
    spec = importlib.util.spec_from_file_location(name, entry)
    if spec is None or spec.loader is None:
        raise ValueError("Capability module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(entry).name == "web_tools.py":
        serve = module.main
    else:
        mcp = getattr(module, "mcp", None)
        from fastmcp import FastMCP

        if not isinstance(mcp, FastMCP):
            raise ValueError("Capability did not export a FastMCP server")

        def serve():
            mcp.run(transport="stdio", show_banner=False)

    if sha256(entry) != expected:
        raise ValueError("Capability changed during loading")
    # Loading is not claimed to be a completed MCP handshake. The native client
    # performs that handshake with this same process after receiving its socket.
    os.write(
        ready_fd,
        (json.dumps({"schema": SCHEMA, "stage": "capability_loaded", "entry_sha256": expected}) + "\n").encode(),
    )
    os.close(ready_fd)
    serve()


async def supervisor(entry, socket_path, record_dir):
    root = Path(record_dir)
    started = time.monotonic()
    read_fd, ready_fd = os.pipe()
    process, listener, writer = None, None, None
    pumps = []
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            __file__,
            "worker",
            entry,
            str(ready_fd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            pass_fds=(ready_fd,),
        )
        os.close(ready_fd)
        ready_fd = None
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=8192)
        transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(read_fd, "rb", buffering=0)
        )
        read_fd = None
        try:
            raw = await reader.readline()
        finally:
            transport.close()
        ready = json.loads(raw)
        if (
            ready != {"schema": SCHEMA, "stage": "capability_loaded", "entry_sha256": check_entry(entry)}
            or process.returncode is not None
        ):
            raise ValueError("Invalid capability readiness evidence")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(socket_path)
        os.chmod(socket_path, 0o600)
        listener.listen(1)
        listener.setblocking(False)
        loaded_at = time.monotonic()
        write_json(
            root / "ready.json",
            {
                **ready,
                "worker_pid_in_namespace": process.pid,
                "load_seconds": loaded_at - started,
                "socket_bound": True,
                "native_handshake_completed": False,
            },
        )
        connection, _ = await loop.sock_accept(listener)
        # There is precisely one scientific session per worker. Do not silently
        # reconnect a new native session to an already-used worker's state.
        listener.close()
        listener = None
        Path(socket_path).unlink()
        reader, writer = await asyncio.open_unix_connection(sock=connection)
        write_json(
            root / "connection.json",
            {"schema": SCHEMA, "connections": 1, "seconds_after_load": time.monotonic() - loaded_at},
        )

        async def forward(source, destination):
            while chunk := await source.read(65536):
                destination.write(chunk)
                await destination.drain()

        pumps = [
            asyncio.create_task(forward(reader, process.stdin)),
            asyncio.create_task(forward(process.stdout, writer)),
        ]
        done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        write_json(
            root / "transport_end.json",
            {"schema": SCHEMA, "stage": "transport_closed", "wall_seconds": time.monotonic() - started},
        )
    except BaseException as exc:
        write_json(
            root / "failure.json",
            {
                "schema": SCHEMA,
                "stage": "supervisor_failed",
                "error_type": type(exc).__name__,
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        for task in pumps:
            task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        if writer is not None:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
        if listener is not None:
            listener.close()
        for fd in (read_fd, ready_fd):
            if fd is not None:
                os.close(fd)
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                process.kill()
                await process.wait()


def relay(socket_path):
    """One stdio connection, byte-for-byte, with no protocol emulation."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path)

        def send():
            try:
                while data := os.read(sys.stdin.fileno(), 65536):
                    sock.sendall(data)
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        thread = threading.Thread(target=send, daemon=True)
        thread.start()
        while data := sock.recv(65536):
            view = memoryview(data)
            while view:
                written = os.write(sys.stdout.fileno(), view)
                if written <= 0:
                    raise OSError("Native stdio output closed")
                view = view[written:]


def sandbox_command(*, workspace, capability, python_root, ipc_root, control_dir, entry):
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
    from .python_runtime_mounts import python_runtime_mounts
    command += system_mounts()
    command += python_runtime_mounts(python_root)
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
        str(Path(capability).resolve()),
        "/capabilities",
        "--ro-bind",
        str(Path(__file__).resolve()),
        "/prepared_runner.py",
        "--bind",
        str(Path(workspace).resolve()),
        "/workspace",
        "--bind",
        str(Path(ipc_root).resolve()),
        "/prepared_mcp",
        "--bind",
        str(Path(control_dir).resolve()),
        "/control",
        "--chdir",
        "/workspace",
        "--clearenv",
    ]
    for name, value in {
        "HOME": "/tmp",
        "PATH": "/opt/venus-runtime/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "VENUS_RSI_WORKSPACE": "/workspace",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        command += ["--setenv", name, value]
    return command + [
        "--",
        "/opt/venus-runtime/bin/python",
        "-u",
        "/prepared_runner.py",
        "supervise",
        "/capabilities/" + entry,
        "/prepared_mcp/worker.sock",
        "/control",
    ]


class PreparedMcp:
    def __init__(self, *, workspace, capability, python_root, control_dir, entry, timeout_s):
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid capability loading budget")
        self.workspace, self.capability = Path(workspace), Path(capability)
        self.python_root, self.control_dir = Path(python_root), Path(control_dir)
        self.entry, self.timeout_s = entry, timeout_s
        self.expected = check_entry(self.capability / entry)
        self.process = self.log = self.ipc = None
        self.stopped = False

    async def start(self):
        self.control_dir.mkdir(parents=True, exist_ok=False)
        self.ipc = tempfile.TemporaryDirectory(prefix="vf-mcp-", dir="/tmp")
        self.ipc_root = Path(self.ipc.name)
        self.log = (self.control_dir / "worker.log").open("xb")
        started = time.monotonic()
        try:
            self.process = await asyncio.create_subprocess_exec(
                *sandbox_command(
                    workspace=self.workspace,
                    capability=self.capability,
                    python_root=self.python_root,
                    ipc_root=self.ipc_root,
                    control_dir=self.control_dir,
                    entry=self.entry,
                ),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=self.log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
            write_json(
                self.control_dir / "launch.json",
                {
                    "schema": SCHEMA,
                    "entry_sha256": self.expected,
                    "supervisor_source_sha256": sha256(__file__),
                    "pid": self.process.pid,
                    "timeout_seconds": self.timeout_s,
                    "network_policy": "Same shared network as the existing MCP host; numerical workers keep their separate isolation",
                    "credential_files_mounted": False,
                },
            )
            deadline = started + self.timeout_s
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Capability loading deadline exhausted")
                    if self.process.returncode is not None:
                        raise RuntimeError("Capability worker exited before readiness")
                    ready_path = self.control_dir / "ready.json"
                    if ready_path.exists():
                        ready = json.loads(ready_path.read_text())
                        if (
                            ready.get("schema") != SCHEMA
                            or ready.get("entry_sha256") != self.expected
                            or ready.get("stage") != "capability_loaded"
                            or ready.get("native_handshake_completed") is not False
                        ):
                            raise ValueError("Capability readiness binding mismatch")
                        if stat.S_ISSOCK((self.ipc_root / "worker.sock").stat().st_mode):
                            return ready
                    await asyncio.sleep(0.1)
        except BaseException as exc:
            write_json(
                self.control_dir / "host_failure.json",
                {
                    "schema": SCHEMA,
                    "stage": "capability_loading",
                    "error_type": type(exc).__name__,
                    "wall_seconds": time.monotonic() - started,
                },
            )
            await self.stop()
            raise

    def connection(self):
        if self.process is None or self.process.returncode is not None or not (self.ipc_root / "worker.sock").exists():
            raise ValueError("Prepared capability is unavailable")
        return {
            "command": "/opt/venus-runtime/bin/python",
            "args": ["-u", "/prepared_runner.py", "relay", "/prepared_mcp/worker.sock"],
        }

    def mounts(self):
        self.connection()
        return [
            "--ro-bind",
            str(Path(__file__).resolve()),
            "/prepared_runner.py",
            "--bind",
            str(self.ipc_root),
            "/prepared_mcp",
        ]

    async def stop(self):
        cleanup = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _stop(self):
        try:
            if self.process is not None and self.process.returncode is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.process.wait(), 4)
                except TimeoutError:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await self.process.wait()
            self.stopped = True
        finally:
            if self.log is not None:
                self.log.close()
            if self.ipc is not None:
                if Path(self.ipc.name).resolve().parent != Path("/tmp").resolve():
                    raise ValueError("Unexpected IPC cleanup directory")
                self.ipc.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "relay", "supervise"])
    parser.add_argument("values", nargs="+")
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.values[0], int(args.values[1]))
    elif args.mode == "relay":
        relay(args.values[0])
    else:
        asyncio.run(supervisor(*args.values))


if __name__ == "__main__":
    main()
