"""Operator configuration for standalone PBS Scientist execution."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

SCHEMA = "venusfactory.pbs-config.v1"
ENVIRONMENT = frozenset({
    "VENUS_KIMI_BIN", "VENUS_CODEX_BIN", "VENUS_CLAUDE_BIN", "VENUS_DSH_BIN", "VENUS_DSH_SITE",
    "VENUS_MODEL_RUNTIMES_CONFIG", "VENUS_SESSION_SANDBOX", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
})


def configuration(value, base=None):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Invalid PBS configuration schema")
    allowed = {"schema", "repository", "control_root", "goal_roots", "python", "runtime_root", "pbs_bin",
               "profiles", "env_file", "credential_toml", "environment", "extra_python_paths",
               "capability_db", "model_runtimes", "execution_catalog", "tool_credential_refs",
               "checkpoint_root", "huggingface_cache"}
    if set(value) - allowed:
        raise ValueError("Unknown PBS configuration fields")
    result = copy.deepcopy(value)
    base = Path(base or Path.cwd())

    def path(raw):
        if not isinstance(raw, str) or not raw or any(c in raw for c in "\r\n\x00"):
            raise ValueError("PBS paths must be explicit local paths")
        # Do not resolve venv Python symlinks to their base interpreter.
        return os.path.abspath(base / Path(raw).expanduser())

    for field in ("repository", "control_root", "python", "runtime_root", "pbs_bin"):
        result[field] = path(result[field])
    for field in ("env_file", "credential_toml", "capability_db", "model_runtimes", "execution_catalog",
                  "checkpoint_root", "huggingface_cache"):
        result[field] = path(result[field]) if result.get(field) else None
    for field in ("goal_roots", "extra_python_paths"):
        values = result.get(field, [])
        if not isinstance(values, list) or (field == "goal_roots" and not values):
            raise ValueError("PBS goal roots must be a nonempty list")
        result[field] = [path(item) for item in values]
    if not Path(result["python"]).is_file():
        raise ValueError("Configured PBS Python does not exist")
    env = result.setdefault("environment", {})
    if not isinstance(env, dict) or set(env) - ENVIRONMENT:
        raise ValueError("PBS environment supports only named runtime settings, never credentials")
    if any(not isinstance(v, str) or len(v) > 4096 or "\x00" in v for v in env.values()):
        raise ValueError("Invalid PBS runtime environment")
    from .pbs_goal_policy import TOOL_CREDENTIALS
    references = result.setdefault("tool_credential_refs", [])
    if (not isinstance(references, list) or any(not isinstance(name, str) for name in references)
            or len(set(references)) != len(references) or set(references) - TOOL_CREDENTIALS):
        raise ValueError("Use supported tool credential names, never values")
    profiles = result.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("At least one bounded PBS resource profile is required")
    for name, resource in profiles.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
            raise ValueError("Invalid PBS resource profile name")
        if not isinstance(resource, dict) or set(resource) != {"queue", "ncpus", "ngpus", "memory_gb", "wall_seconds"}:
            raise ValueError("Invalid PBS resource fields")
        if not isinstance(resource["queue"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", resource["queue"]):
            raise ValueError("Use a queue name without PBS options")
        bounds = {"ncpus": (1, 8), "ngpus": (1, 1), "memory_gb": (16, 64), "wall_seconds": (30, 86400)}
        for field, (low, high) in bounds.items():
            if type(resource[field]) is not int or not low <= resource[field] <= high:
                raise ValueError("PBS resource request exceeds its supported bounds")
    return result


def validate_observed_resources(row, expected):
    """A held job may be altered by a hook/operator; verify before release/use."""
    resources = row.get("Resource_List")
    if not isinstance(resources, dict) or row.get("queue") != expected["queue"]:
        raise ValueError("PBS resource queue differs from the admitted profile")
    if str(resources.get("nodect")) != "1":
        raise ValueError("PBS research requires one compute node")
    selection = str(resources.get("select", ""))
    if not selection.startswith("1:") or "+" in selection:
        raise ValueError("PBS research requires one resource chunk")
    if row.get("Rerunable") not in (False, "False", "false"):
        raise ValueError("PBS research jobs must not be automatically rerun")
    for key in ("ncpus", "ngpus"):
        value = resources.get(key)
        if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)) or int(value) != expected[key]:
            raise ValueError("PBS CPU/GPU request differs from the admitted profile")
    memory = re.fullmatch(r"([0-9]+)(b|kb|mb|gb|tb)", str(resources.get("mem", "")).lower())
    if not memory or int(memory[1]) * 1024 ** ("b", "kb", "mb", "gb", "tb").index(memory[2]) != expected["memory_gb"] * 1024**3:
        raise ValueError("PBS memory request differs from the admitted profile")
    wall = re.fullmatch(r"([0-9]+):([0-5][0-9]):([0-5][0-9])", str(resources.get("walltime", "")))
    if not wall or int(wall[1]) * 3600 + int(wall[2]) * 60 + int(wall[3]) != expected["wall_seconds"]:
        raise ValueError("PBS wall-time request differs from the admitted profile")
