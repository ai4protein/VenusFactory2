"""Freeze administrator tool configuration without copying credentials or models."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path

from agent.rsi.engine.storage import atomic_json

TOOL_CREDENTIALS = frozenset({"TAVILY_API_KEY", "SERPAPI_API_KEY", "NCBI_API_KEY"})


def _read(path, limit=16 * 1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022:
            raise ValueError("Administrator configuration must be a non-writable regular file")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Administrator configuration is too large")
    return raw


def _reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(_read(path)).hexdigest()}


def _asset_directory(value, *, required):
    if (not isinstance(value, str) or not value or any(c in value for c in "\r\n\x00")
            or not Path(value).is_absolute()):
        raise ValueError("PBS asset directories must be explicit absolute paths")
    path = Path(value).resolve()
    if (required and not path.is_dir()) or (path.exists() and not path.is_dir()):
        raise ValueError("Configured PBS asset directory is unavailable")
    return str(path)


def _freeze_assets(settings):
    # This binds asset locations, not weight bytes. Installed weights are not
    # copied into a source snapshot and must remain operator-managed.
    raw = str(settings.get("checkpoint_root") or Path(settings["runtime_root"]) / "ckpt")
    required = bool(settings.get("checkpoint_root")) or Path(raw).exists()
    cache = settings.get("huggingface_cache")
    return {"schema": "venusfactory.pbs-assets.v1",
            "checkpoint_root": _asset_directory(raw, required=required),
            "checkpoint_root_required": required,
            "huggingface_cache": _asset_directory(cache, required=True) if cache else None}


def _asset_environment(value):
    fields = {"schema", "checkpoint_root", "checkpoint_root_required", "huggingface_cache"}
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema"] != "venusfactory.pbs-assets.v1"
            or type(value["checkpoint_root_required"]) is not bool):
        raise ValueError("Invalid PBS asset configuration")
    result = {}
    for field, env_name, required in (
        ("checkpoint_root", "VENUS_CKPT_DIR", value["checkpoint_root_required"]),
        ("huggingface_cache", "HF_HUB_CACHE", True),
    ):
        raw = value[field]
        if field == "huggingface_cache" and raw is None:
            continue
        path = _asset_directory(raw, required=required)
        if path != raw:
            raise ValueError("PBS asset directory was redirected")
        result[env_name] = path
    return result


def freeze_settings(settings, destination):
    from mcp_tools.models.profiles import normalize_runtime_entry
    from mcp_tools.tools.capability_policy import workspace_switch_rows
    from mcp_tools.tools.capability_snapshot import document
    installation = Path(settings["runtime_root"])
    destination = Path(destination)
    atomic_json(destination / "assets.json", _freeze_assets(settings))
    database = Path(settings.get("capability_db") or installation / "runtime/local/state/capabilities.sqlite3")
    rows = {}
    if database.exists():
        if database.is_symlink() or not database.is_file():
            raise ValueError("Invalid administrator capability database")
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5) as connection:
            connection.row_factory = sqlite3.Row
            rows = workspace_switch_rows(connection.execute("SELECT key,enabled,revision,selector FROM switches"))
    elif settings.get("capability_db"):
        raise ValueError("Configured capability database is unavailable")
    atomic_json(destination / "capability-policy.json", document(rows))
    path = Path(settings.get("model_runtimes") or settings["environment"].get("VENUS_MODEL_RUNTIMES_CONFIG") or installation / "configs/model_runtimes.json")
    if (settings.get("model_runtimes") or settings["environment"].get("VENUS_MODEL_RUNTIMES_CONFIG")) and not path.exists():
        raise ValueError("Configured model runtimes are unavailable")
    runtimes = json.loads(_read(path)) if path.exists() else {}
    if not isinstance(runtimes, dict):
        raise ValueError("Invalid administrator model runtimes")
    def check_credentials(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if any(secret in key.lower() for secret in ("api_key", "password", "secret", "access_token")):
                    raise ValueError("Model runtime configuration must not contain credentials")
                check_credentials(item)
        elif isinstance(value, list):
            for item in value:
                check_credentials(item)

    check_credentials(runtimes)
    runtimes = {name: normalize_runtime_entry(entry, name=name, root=installation)
                for name, entry in runtimes.items()}
    atomic_json(destination / "model-runtimes.json", runtimes)
    catalog_path = Path(settings.get("execution_catalog") or Path(settings["repository"]) / "configs/harness/free_execution_catalog.json")
    catalog = json.loads(_read(catalog_path))
    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
        raise ValueError("Invalid execution catalog")
    from .contracts import ModelBinding
    for row in catalog["models"]:
        ModelBinding(**row["binding"])
        if set(row) - {"id", "label", "binding", "credential_toml", "inference_controls", "enabled", "pricing", "max_output_tokens", "protocols"}:
            raise ValueError("Unsupported execution catalog fields")
        if any(key in row for key in ("api_key", "token", "password")) or any(key in row.get("binding", {}) for key in ("api_key", "token", "password")):
            raise ValueError("Execution catalogs must use credential references")
    atomic_json(destination / "execution-catalog.json", catalog)
    return {name: _reference(destination / filename) for name, filename in {
        "capabilities": "capability-policy.json", "runtimes": "model-runtimes.json", "catalog": "execution-catalog.json",
        "assets": "assets.json"}.items()}


def verify_settings(record):
    paths = {}
    root = Path(record["record_path"]).parent
    bindings = record["tool_settings"]
    if set(bindings) != {"capabilities", "runtimes", "catalog", "assets"}:
        raise ValueError("Incomplete PBS tool configuration")
    for name, reference in bindings.items():
        path = Path(reference["path"])
        if path.parent != root or path.is_symlink() or hashlib.sha256(_read(path)).hexdigest() != reference["sha256"]:
            raise ValueError("PBS tool configuration changed")
        paths[name] = str(path)
    return {"VENUS_CAPABILITY_POLICY_SNAPSHOT": paths["capabilities"],
            "VENUS_MODEL_RUNTIMES_CONFIG": paths["runtimes"], "VENUS_EXECUTION_CATALOG": paths["catalog"],
            **_asset_environment(json.loads(_read(paths["assets"])))}


def service_environment(record):
    """Only named search-service credentials enter trusted MCP children."""
    names = record["settings"].get("tool_credential_refs", [])
    if not isinstance(names, list) or set(names) - TOOL_CREDENTIALS:
        raise ValueError("Unsupported MCP credential references")
    if not names:
        return {}
    from io import StringIO

    from dotenv import dotenv_values

    from .pbs_goal_worker import _protected_text
    source = record["execution"].get("env_file")
    if not source:
        raise ValueError("Tool credentials need a protected environment file")
    values = dotenv_values(stream=StringIO(_protected_text(source)), interpolate=False)
    result = {}
    for name in names:
        value = values.get(name)
        if not isinstance(value, str) or not value.strip() or any(c in value for c in ("\n", "\r", "${")):
            raise ValueError("Configured tool credential is unavailable")
        result[name] = value
    return result
