"""Admission and frozen inference identity for native Web Agent adapters."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from agent.harness import Profile
from agent.harness.catalog import ExecutionCatalog, free_model, resolve_credential
from agent.harness.contracts import digest
from agent.harness.provider_config import online

ROOT = Path(__file__).resolve().parents[2]
WEB_HARNESSES = {"kimi-code", "langgraph", "deepseek-harness", "codex", "claude-code"}


def catalog():
    profile = Profile.from_dict(json.loads((ROOT / "configs/harness/free_science_goal.json").read_text()))
    profile = replace(
        profile,
        mode="science_agent",
        required_controls=("request_gateway", "events"),
        budget=replace(profile.budget, thinking="disabled", output_tokens=4096),
    )
    path = Path(os.environ.get("VENUS_EXECUTION_CATALOG") or ROOT / "configs/harness/free_execution_catalog.json")
    return ExecutionCatalog(ROOT, profile, path=path, free_only=online(), include_configured=True)


def runtime_paths():
    from backend.web_v2.science_goals import ScienceGoalService

    return ScienceGoalService(root=ROOT).runtime_paths()


def options():
    cat = catalog()
    public = cat.public(runtime_paths())
    names = {
        "kimi-code": "Kimi Code",
        "deepseek-harness": "DeepSeek Harness",
        "codex": "Codex",
        "claude-code": "Claude Code",
    }
    combinations = [c for c in public["combinations"] if c["harness"] in names]
    harnesses = []
    for name, label in names.items():
        if cat.enabled_harnesses is not None and name not in cat.enabled_harnesses:
            continue
        pairs = [c for c in combinations if c["harness"] == name]
        installed = any("runtime_missing" not in c["reasons"] for c in pairs)
        harnesses.append(
            {
                "id": name,
                "label": label,
                "ready": installed,
                "model_catalog": "shared",
                "reason": None if installed else "runtime_missing",
            }
        )
    models = [
        {**m, "ready": any(c["ready"] for c in combinations if c["model_id"] == m["id"])} for m in public["models"]
    ]
    return {
        "default": {"harness": cat.default_profile.harness, "model_id": cat.default_model},
        "models": models,
        "harnesses": harnesses,
        "combinations": combinations,
        "shared_default_model": cat.default_model,
        "readiness": public["readiness"],
    }


def admit(state, selection=None):
    """Return a frozen binding; never migrate a conversation or read client paths."""
    saved = state.get("agent_execution")
    if saved:
        payload = {k: v for k, v in saved.items() if k != "sha256"}
        if saved.get("sha256") != digest(payload):
            raise ValueError("Stored Agent execution binding changed")
        if selection is not None and selection != saved["selection"]:
            raise ValueError("Start a new conversation to change its execution profile")
        profile = Profile.from_dict(saved["profile"])
        if profile.harness not in WEB_HARNESSES or profile.mode != "science_agent":
            raise ValueError("Unsupported stored Agent profile")
        if online() and not free_model(profile.model):
            raise ValueError("Select a free model in a new conversation")
        return saved, resolve_credential(profile.model, saved.get("credential_toml"))
    migrating = selection is None
    if migrating:
        defaults = options()
        chosen = defaults["default"]
        free_ids = {m["id"] for m in defaults["models"] if m.get("free")}
        ready = [p for p in defaults["combinations"] if p["ready"] and p["harness"] != "langgraph" and p["model_id"] in free_ids]
        if not ready: raise ValueError("Choose a configured model and harness for a new conversation")
        selection = chosen if any(p["harness"] == chosen["harness"] and p["model_id"] == chosen["model_id"] for p in ready) else {k:ready[0][k] for k in ("harness","model_id")}
    if not selection or set(selection) != {"harness", "model_id"} or selection["harness"] not in WEB_HARNESSES:
        raise ValueError("Selected Web Agent adapter is unavailable")
    if not migrating and any(row.get("role") == "user" for row in state.get("history", [])):
        raise ValueError("Start a new conversation to choose an Agent harness")
    profile, model = catalog().select(selection)
    if online() and not free_model(profile.model):
        raise ValueError("Select a free model")
    if not catalog().registry.preflight(profile, runtime_paths(), include_runtime_digest=False)["ready"]:
        raise ValueError("Selected harness and model are unavailable")
    key = resolve_credential(profile.model, model.credential_toml)
    binding = {
        "selection": dict(selection),
        "profile": profile.public(),
        "credential_toml": str(model.credential_toml) if model.credential_toml else None,
    }
    binding["sha256"] = digest(binding)
    return binding, key


def public_execution(state):
    binding = state.get("agent_execution")
    if not binding:
        return None
    model = binding["profile"]["model"]
    return {
        **binding["selection"],
        "provider": model["provider"],
        "model": model["model"],
        "profile_sha256": binding["sha256"],
    }
