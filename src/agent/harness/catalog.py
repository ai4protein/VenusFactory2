"""Trusted host catalog: scientific mode, executor and inference stay independent.

Clients select stable IDs, never endpoints, import paths or credentials. Catalog
readiness is static admission, not a claim of model or scientific conformance.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from .contracts import ModelBinding
from .registry import builtin_registry


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    binding: ModelBinding
    credential_toml: Path | None = None
    inference_controls: dict | None = None
    max_output_tokens: int | None = None

    def public(self):
        return {
            "id": self.id,
            "label": self.label,
            "provider": self.binding.provider,
            "model": self.binding.model,
            "protocol": self.binding.protocol,
            "free": free_model(self.binding),
            **({"inference_controls": dict(self.inference_controls)} if self.inference_controls is not None else {}),
        }


def resolve_credential(model, credential_toml=None):
    """Resolve only this binding; a legacy runtime's default model is irrelevant."""
    key = os.environ.get(model.credential_env)
    if key:
        return key
    if credential_toml:
        cfg = tomllib.loads(Path(credential_toml).read_text())
        provider = cfg.get("providers", {}).get(model.provider, {})
        if str(provider.get("base_url", "")).rstrip("/") != model.base_url.rstrip("/"):
            raise ValueError("Credential source endpoint differs from requested provider")
        key = provider.get("api_key")
        if isinstance(key, str) and key:
            return key
    raise ValueError("Selected model credential is unavailable")


class ExecutionCatalog:
    def __init__(self, root, default_profile, *, path=None, legacy_credential=None, registry=None, free_only=False, include_configured=False):
        self.free_only = free_only
        self.hide_nonfree = include_configured and free_only
        self.registry = registry or builtin_registry()
        self.default_profile = default_profile
        self.models = {}
        self.enabled_harnesses = None
        configured_default = None
        if path is not None:
            data = json.loads(Path(path).read_text())
            if set(data) - {"models", "defaults", "harnesses"} or not isinstance(data["models"], list):
                raise ValueError("Invalid execution catalog")
            configured_default = data.get("defaults", {}).get(default_profile.mode)
            self.enabled_harnesses = data.get("harnesses", {}).get(default_profile.mode)
            known = {a.id for a in self.registry.specs()}
            if self.enabled_harnesses is not None and (not isinstance(self.enabled_harnesses, list) or not self.enabled_harnesses or any(h not in known for h in self.enabled_harnesses) or len(set(self.enabled_harnesses)) != len(self.enabled_harnesses)):
                raise ValueError("Invalid configured harnesses")
            for row in data["models"]:
                if row.get("enabled", True) is not True:
                    continue
                if set(row) - {"id", "label", "binding", "credential_toml", "inference_controls", "enabled", "pricing", "max_output_tokens", "protocols"}:
                    raise ValueError("Unknown model catalog field")
                mid = row["id"]
                if not isinstance(mid, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", mid) or mid in self.models:
                    raise ValueError("Invalid or duplicate model identity")
                bridge = row.get("credential_toml")
                bridge = (Path(root) / bridge).resolve() if bridge else None
                controls = row.get("inference_controls")
                if controls is not None:
                    if not isinstance(controls, dict) or set(controls) - {
                        "thinking",
                        "temperature",
                        "reasoning_effort",
                    }:
                        raise ValueError("Only explicit inference controls belong in a model option")
                    replace(default_profile.budget, **controls)
                    controls = dict(controls)
                self.models[mid] = ModelOption(mid, str(row["label"]), ModelBinding(**row["binding"]), bridge, controls)
        if include_configured:
            # Official endpoints are opt-in by model ID, never enabled by an
            # unrelated pre-existing API key or an example catalog placeholder.
            placeholders = {'openai-official', 'deepseek-official', 'google-official', 'alibaba-official'}
            self.models = {mid: m for mid, m in self.models.items() if mid not in placeholders}
            if not free_only:
                from .provider_config import configured_models
                for row in configured_models():
                    self.models[row['id']] = ModelOption(
                        row['id'], row['label'], row['binding'],
                        inference_controls={'thinking': None, 'temperature': 1.0, 'reasoning_effort': None},
                        max_output_tokens=row['max_output_tokens'],
                    )
        matching = [m.id for m in self.models.values() if m.binding == default_profile.model]
        if matching:
            self.default_model = matching[0]
        else:
            mid = "profile-default"
            if mid in self.models:
                raise ValueError("Reserved default model identity conflicts with catalog")
            self.models[mid] = ModelOption(
                mid,
                default_profile.model.model,
                default_profile.model,
                Path(legacy_credential) if legacy_credential else None,
            )
            self.default_model = mid

        if configured_default:
            if set(configured_default) != {"harness", "model_id"} or configured_default["model_id"] not in self.models or (self.enabled_harnesses is not None and configured_default["harness"] not in self.enabled_harnesses):
                raise ValueError("Invalid configured default")
            self.default_model = configured_default["model_id"]
            self.default_profile = replace(default_profile, harness=configured_default["harness"], model=self.models[self.default_model].binding)

    def select(self, selection=None):
        selection = selection or {"harness": self.default_profile.harness, "model_id": self.default_model}
        if set(selection) != {"harness", "model_id"} or selection["model_id"] not in self.models:
            raise ValueError("Select a registered executor and model")
        if self.enabled_harnesses is not None and selection["harness"] not in self.enabled_harnesses:
            raise ValueError("Executor is disabled in the host configuration")
        model = self.models[selection["model_id"]]
        if selection["harness"] not in {s.id for s in self.registry.specs()}:
            raise ValueError("Unknown executor")
        budget = replace(self.default_profile.budget, **(model.inference_controls or {}))
        if model.max_output_tokens is not None:
            budget = replace(budget, output_tokens=min(budget.output_tokens, model.max_output_tokens))
        binding = model.binding
        # OpenRouter documents both protocols for the same exact free model.
        # The selected protocol is frozen before admission, never inferred in flight.
        if selection["harness"] in {"codex", "expert-codex"} and binding.provider in {"openrouter", "openai"}:
            binding = replace(binding, protocol="responses")
            budget = replace(budget, thinking=None, reasoning_effort=None)
        return replace(self.default_profile, harness=selection["harness"], model=binding, budget=budget), model

    def public(self, runtimes):
        combinations = []
        for option in self.models.values():
            for adapter in self.registry.specs():
                if self.enabled_harnesses is not None and adapter.id not in self.enabled_harnesses:
                    continue
                try:
                    profile, _ = self.select({"harness": adapter.id, "model_id": option.id})
                except ValueError:
                    reasons = ["execution_policy_incompatible"]
                else:
                    check = self.registry.preflight(profile, runtimes, include_runtime_digest=False)
                    reasons = list(check["reasons"])
                try:
                    resolve_credential(option.binding, option.credential_toml)
                except (OSError, ValueError, KeyError):
                    reasons.insert(0, "credential_unavailable")
                if self.free_only and not free_model(option.binding):
                    reasons.append("free_model_required")
                combinations.append(
                    {"harness": adapter.id, "model_id": option.id, "ready": not reasons, "reasons": reasons}
                )
        return {
            "mode": self.default_profile.mode,
            "default": {"harness": self.default_profile.harness, "model_id": self.default_model},
            "models": [m.public() for m in self.models.values() if not self.hide_nonfree or free_model(m.binding)],
            "harnesses": [
                {"id": s.id, "modes": list(s.modes), "protocols": list(s.protocols), "status": s.status}
                for s in self.registry.specs()
                if self.enabled_harnesses is None or s.id in self.enabled_harnesses
            ],
            "combinations": combinations,
            "readiness": "static_admission_not_live_conformance",
        }


def free_model(binding):
    from .free_catalog import configured_free
    return configured_free(binding.model, binding.base_url, binding.protocol)
