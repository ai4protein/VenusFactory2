"""Fixed whole-goal worker inside a verified, operator-owned PBS allocation."""
from __future__ import annotations

import asyncio
import os
import stat
import sys
import tomllib
from pathlib import Path

from backend.compute.access import require_local_execution


def _protected_text(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("Credential reference must be an absolute regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid not in {0, os.geteuid()} or info.st_mode & 0o077:
            raise ValueError("Credential source must be private to the service owner")
        if info.st_size > 1024 * 1024:
            raise ValueError("Credential source is too large")
        value = stream.read(1024 * 1024 + 1)
        if len(value.encode("utf-8")) > 1024 * 1024:
            raise ValueError("Credential source is too large")
        return value


def _selected_credential(model, execution):
    """Resolve only this model binding; never export provider keys to children."""
    key = None
    if execution.get("env_file"):
        from io import StringIO

        from dotenv import dotenv_values
        values = dotenv_values(stream=StringIO(_protected_text(execution["env_file"])), interpolate=False)
        key = values.get(model.credential_env)
    if not key and execution.get("credential_toml"):
        config = tomllib.loads(_protected_text(execution["credential_toml"]))
        provider = config.get("providers", {}).get(model.provider, {})
        if str(provider.get("base_url", "")).rstrip("/") != model.base_url.rstrip("/"):
            raise ValueError("Credential endpoint differs from the selected model")
        key = provider.get("api_key")
    if not isinstance(key, str) or not key.strip() or "${" in key or "\n" in key or "\r" in key:
        raise ValueError("Selected model credential is unavailable")
    return key


def _environment(record_path, record):
    """Replace the submission host environment with fixed job-private settings."""
    execution = record["execution"]
    configured = execution.get("environment", {})
    if not isinstance(configured, dict):
        raise ValueError("Invalid worker environment")
    from .pbs_goal_config import ENVIRONMENT
    allowed = ENVIRONMENT
    if set(configured) - allowed or any(not isinstance(v, str) or "\x00" in v for v in configured.values()):
        raise ValueError("Unsupported worker environment setting")
    if configured.get("WEBUI_V2_MODE", "local") != "local" or configured.get("VENUS_SCIENTIST_ONLINE") == "1":
        raise ValueError("Authored PBS GPU research is local-only")
    from .cluster_allocation import backend
    scheduler = backend(record)
    allocated_names = (("PBS_JOBID", "PBS_NODEFILE", "PBS_GPUFILE", "CUDA_VISIBLE_DEVICES", "VENUS_PBS_GOAL_ATTEMPT")
                       if scheduler == "pbs" else ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_STEP_ID"))
    allocated = {name: os.environ[name] for name in allocated_names if name in os.environ}
    private = Path(record_path).parent / "runtime"
    home = private / "home"
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = {"PATH": str(Path(execution["python"]).parent) + ":/usr/bin:/bin", "LANG": "C.UTF-8",
           "HOME": str(home), "PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **configured, **allocated,
           "WEBUI_V2_MODE": "local", "VENUS_SCIENTIST_ONLINE": "0", "VENUS_RESEARCH_GPU_BACKEND": "pbs" if scheduler == "pbs" else "disabled",
           "VENUS_SANDBOX_BACKEND": "bwrap", "VENUS_SESSION_SANDBOX": "bwrap",
           "VENUS_SCIENTIST_CLUSTER_RECORD": str(Path(record_path).resolve()),
           "VENUS_RUNTIME_ROOT": str(private), "VENUS_JOB_QUEUE_ROOT": str(private / "queue"),
           "VENUS_RESEARCH_CAPABILITY_ROOT": str(private / "research_capabilities"),
           "VENUS_SANDBOX_CONTROL_ROOT": str(private / "sandbox"), "TEMP_OUTPUTS_DIR": str(private / "outputs"),
           "VENUS_USAGE_DB": str(private / "usage.sqlite"), "VENUS_ACCESS_DB": str(private / "access.sqlite"),
           "VENUS_QUEUE_CPU_SLOTS": str(record["resources"]["ncpus"]), "VENUS_QUEUE_GPU_SLOTS": str(record["resources"]["ngpus"]),
           "VENUS_SANDBOX_TOTAL_MEMORY_MB": str(record["resources"]["memory_gb"] * 1024),
           "VENUS_SANDBOX_LOCAL_MEMORY_MB": "8192", "VENUS_SANDBOX_MEMORY_RESERVE_MB": "4096",
           "VENUS_SANDBOX_MAX_PARALLEL": "1"}
    if scheduler == "pbs":
        env["VENUS_SCIENTIST_PBS_RECORD"] = str(Path(record_path).resolve())
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    source = Path(record["source"]["root"])
    extras = execution.get("extra_python_paths", [])
    if not isinstance(extras, list) or any(not isinstance(p, str) or not Path(p).is_absolute() or not Path(p).is_dir() for p in extras):
        raise ValueError("Worker dependency paths must be explicit absolute directories")
    env["PYTHONPATH"] = os.pathsep.join([str(source / "src"), str(source), *extras])
    from .pbs_goal_policy import verify_settings
    env.update(verify_settings(record))
    env.update({"VENUS_WORKERS_RESEARCH": "1", "VENUS_WORKERS_CPU": "1",
                "VENUS_WORKERS_GPU": "1", "VENUS_WORKERS_API": "1",
                "VENUS_QUEUE_API_SLOTS": "1", "VENUS_QUEUE_RESEARCH_SLOTS": "1"})
    return env


def _allocation(record):
    from .cluster_allocation import capture_allocation
    allocation = capture_allocation(record)
    if allocation.job_id != record["job_id"]:
        raise ValueError("Allocation differs from the claimed cluster job")
    return allocation


def _source_bindings(source):
    source = Path(source).resolve()
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in {"agent", "backend", "mcp_tools"}:
            continue
        path = getattr(module, "__file__", None)
        if path and not Path(path).resolve().is_relative_to(source):
            raise ValueError("A platform module was loaded outside the frozen source")


def run_record(record_path):
    """Claim once, validate allocation, then run the original active GoalStore."""
    require_local_execution()
    from . import pbs_goal
    record_path = Path(record_path).resolve()
    record = pbs_goal.worker_claim(record_path)
    goal_status, error_type = "interrupted", None
    try:
        from .pbs_goal_snapshot import verify_source
        verify_source(record["source"])
        if os.path.abspath(sys.executable) != os.path.abspath(record["execution"]["python"]):
            raise ValueError("Worker Python differs from the admitted interpreter")
        env = _environment(record_path, record)
        os.environ.clear()
        os.environ.update(env)
        sys.dont_write_bytecode = True
        allocation = _allocation(record)  # Must precede credentials and any provider request.
        pbs_goal.worker_ready(record_path, allocation.receipt())
        from agent.science_goal import GoalStore

        from .catalog import free_model
        from .contracts import Profile
        from .installation import default_runtimes
        store = GoalStore(record["goal"])
        state = store.read()
        if state["status"] != "active" or state["inflight"] is not None:
            raise ValueError("Only active checkpointed goals can enter a PBS worker")
        profile = Profile.from_dict(state["definition"]["profile"])
        if not free_model(profile.model):
            raise ValueError("PBS research requires an enabled free model route")
        runtimes = default_runtimes(Path(record["execution"]["runtime_root"]))
        from . import builtin_registry
        if not builtin_registry().preflight(profile, runtimes)["ready"]:
            raise ValueError("The selected research harness is unavailable")
        from .autonomous_research import AutonomousResearch
        _source_bindings(record["source"]["root"])
        credential = _selected_credential(profile.model, record["execution"])
        workspace = store.root / "workspace"
        if not workspace.is_dir() or workspace.is_symlink():
            raise ValueError("Research workspace is unavailable")
        os.chdir(workspace)
        async def execute():
            from .pbs_goal_queue import queue_run
            from .pbs_goal_tools import tool_service
            async with queue_run(record), tool_service(record):
                return await AutonomousResearch(store, credential, runtimes).run()
        result = asyncio.run(execute())
        verify_source(record["source"])  # Children must not mutate the admitted source.
        goal_status = result["status"]
        return result
    except BaseException as exc:
        error_type = type(exc).__name__
        # User control may have changed the goal while the worker was starting
        # or stopping. Preserve that observed state without rewriting the goal.
        try:
            from agent.science_goal import GoalStore
            observed = GoalStore(record["goal"]).read()["status"]
            if observed != "active":
                goal_status = observed
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise
    finally:
        pbs_goal.worker_finish(record_path, goal_status, error_type=error_type)
