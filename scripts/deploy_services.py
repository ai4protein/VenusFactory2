#!/usr/bin/env python3
"""Supervise local/online API and MCP services inside one PBS allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local", "online", "both"], default="both")
    parser.add_argument("--online-port", type=int, default=7863)
    parser.add_argument("--local-port", type=int, default=7862)
    parser.add_argument("--online-mcp-port", type=int, default=8085)
    parser.add_argument("--local-mcp-port", type=int, default=8084)
    parser.add_argument("--local-kimi-port", type=int, default=58628)
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "runtime")
    parser.add_argument("--frontend-dist", type=Path, default=ROOT / "frontend/dist")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--frp-executable", type=Path)
    parser.add_argument("--frp-config", type=Path)
    parser.add_argument("--startup-timeout", type=int, default=120)
    args = parser.parse_args(argv)
    if bool(args.frp_executable) != bool(args.frp_config):
        parser.error("Specify both FRP executable and private configuration, or neither")
    return args


def service_plan(args, inherited=None):
    from backend.web_v2.runtime_paths import AREAS
    base = dict(os.environ if inherited is None else inherited)
    modes = ["online", "local"] if args.mode == "both" else [args.mode]
    ports = [getattr(args, mode + suffix) for mode in modes for suffix in ("_port", "_mcp_port")]
    if "local" in modes:
        ports.append(args.local_kimi_port)
    if len(set(ports)) != len(ports) or any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("API and MCP services require distinct valid ports")
    plans = []
    for mode in modes:
        runtime = args.runtime_root.resolve() / mode
        env = dict(base, WEBUI_V2_MODE=mode, VENUS_RUNTIME_ROOT=str(args.runtime_root.resolve()),
                   PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        env.update({name: str(runtime / area) for name, area in AREAS.items()})
        env["KIMI_CODE_HOME"] = str(runtime / "control/kimi")
        if mode == "local":
            env.update(KIMI_PORT=str(args.local_kimi_port), KIMI_EXTERNAL="0",
                       KIMI_BASE_URL=f"http://127.0.0.1:{args.local_kimi_port}")
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("OPENBLAS_NUM_THREADS", "2")
        command = [sys.executable, "-u", str(ROOT / "scripts/serve.py"), "--mode", mode,
                   "--port", str(getattr(args, mode + "_port")),
                   "--mcp-port", str(getattr(args, mode + "_mcp_port")),
                   "--runtime-root", str(args.runtime_root.resolve()),
                   "--frontend-dist", str(args.frontend_dist.resolve()), "--env-file", str(args.env_file.resolve())]
        plans.append({"mode": mode, "command": command, "env": env, "port": getattr(args, mode + "_port"),
                      "mcp_port": getattr(args, mode + "_mcp_port"), "runtime": runtime})
    return plans


def prepare_harness_home(plan, source=None):
    """Seed settings once; each mode owns its MCP configuration and daemon state."""
    source = Path(source or os.environ.get("KIMI_CONFIG_DIR") or Path.home() / ".kimi-code")
    destination = Path(plan["env"]["KIMI_CODE_HOME"])
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("config.toml", "providers", "credentials.json", "auth.json"):
        src, dst = source / name, destination / name
        if not src.exists() or dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
            for item in dst.rglob("*"):
                item.chmod(0o700 if item.is_dir() else 0o600)
            dst.chmod(0o700)
        else:
            shutil.copy2(src, dst)
            dst.chmod(0o600)


def service_ready(port, kind):
    """Require working chat catalogs before advertising an API as ready."""
    def read(path):
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
            return json.load(response)

    try:
        health = read("/health")
        if not isinstance(health, dict):
            return False
        if kind == "mcp":
            return health.get("status") == "ready"
        if health.get("success") is not True:
            return False
        agent = read("/api/chat/agent-execution-options")
        if not isinstance(agent, dict) or not all(
            isinstance(agent.get(key), list) for key in ("harnesses", "models", "combinations")
        ):
            return False
        scientist = read("/api/chat/science-goals/catalog?view=composer")
        return (isinstance(scientist, dict)
                and isinstance(scientist.get("drivers"), list)
                and isinstance(scientist.get("execution"), dict))
    except (OSError, ValueError):
        return False


def remember_children(processes, owned):
    import psutil
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            parent = psutil.Process(process.pid)
            for child in [parent, *parent.children(recursive=True)]:
                owned[child.pid] = child
        except psutil.NoSuchProcess:
            pass


def stop_children(processes, owned=None):
    """Stop recorded descendants too if their parent API has already exited."""
    import psutil
    owned = {} if owned is None else owned
    remember_children(processes, owned)
    for child in owned.values():
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(owned.values()), timeout=20)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def main(argv=None):
    import fcntl
    args = arguments(argv)
    os.umask(0o077)
    plans = service_plan(args)
    if not (args.frontend_dist / "index.html").is_file():
        raise SystemExit("Build frontend/ before submitting this job")
    control = args.runtime_root.resolve() / "services"
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "deployment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for plan in plans:
        ports = [plan["port"], plan["mcp_port"]]
        if plan["mode"] == "local":
            ports.append(args.local_kimi_port)
        for port in ports:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
    children, streams, owned = [], [], {}
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, request_stop)
    state = {"job_id": os.getenv("PBS_JOBID"), "host": socket.gethostname(), "status": "starting", "services": []}

    def save():
        pending = control / "deployment.json.tmp"
        pending.write_text(json.dumps(state, indent=2) + "\n")
        pending.replace(control / "deployment.json")

    try:
        for plan in plans:
            prepare_harness_home(plan)
            logs = plan["runtime"] / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            log = (logs / "web.log").open("ab"); streams.append(log)
            child = subprocess.Popen(plan["command"], cwd=ROOT, env=plan["env"], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            state["services"].append({"mode": plan["mode"], "pid": child.pid, "port": plan["port"],
                                      "mcp_port": plan["mcp_port"], "log": str(logs / "web.log")})
        save()
        deadline = time.monotonic() + args.startup_timeout
        pending = {(plan["port"], "api") for plan in plans} | {(plan["mcp_port"], "mcp") for plan in plans}
        while pending and not stopping:
            remember_children(children, owned)
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A web process exited during startup; inspect its log")
            for port, kind in list(pending):
                if service_ready(port, kind):
                    pending.remove((port, kind))
            if pending and time.monotonic() >= deadline:
                raise RuntimeError("Services or chat catalogs did not become ready before the startup deadline")
            time.sleep(.5)
        if not stopping and args.frp_executable:
            log = (control / "frp.log").open("ab"); streams.append(log)
            children.append(subprocess.Popen([str(args.frp_executable.resolve()), "-c", str(args.frp_config.resolve())],
                                             cwd=args.frp_executable.resolve().parent, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        if not stopping:
            state["status"] = "ready"; save()
            print(json.dumps(state), flush=True)
        while not stopping:
            remember_children(children, owned)
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A managed service exited; stopping this allocation cleanly")
            time.sleep(1)
    except BaseException:
        state["status"] = "failed"
        raise
    finally:
        stop_children(children, owned)
        if state["status"] != "failed":
            state["status"] = "stopped"
        save()
        for stream in streams:
            stream.close()
        lock.close()


if __name__ == "__main__":
    main()
