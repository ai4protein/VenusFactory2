"""Lazy, explicit adapter discovery. Unavailable combinations never fall back."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import file_hash


@dataclass(frozen=True)
class AdapterSpec:
    id: str
    factory: str | None
    modes: tuple[str, ...]
    protocols: tuple[str, ...]
    controls: tuple[str, ...]
    runtime_key: str
    status: str = "implemented"


class HarnessRegistry:
    def __init__(self):
        self._specs = {}

    def register(self, spec):
        if spec.id in self._specs:
            raise ValueError("Duplicate harness identity")
        self._specs[spec.id] = spec

    def specs(self):
        return tuple(self._specs.values())

    def preflight(self, profile, runtimes):
        spec = self._specs.get(profile.harness)
        reasons = []
        if spec is None:
            return {"ready": False, "reasons": ["unknown_harness"], "profile": profile.public()}
        if profile.mode not in spec.modes:
            reasons.append("mode_not_implemented_for_harness")
        if profile.model.protocol not in spec.protocols:
            reasons.append("model_protocol_incompatible")
        if not set(profile.required_controls) <= set(spec.controls):
            reasons.append("required_controls_unavailable")
        if spec.factory is None:
            reasons.append("adapter_not_implemented")
        path = Path(runtimes.get(spec.runtime_key, "/nonexistent-runtime"))
        if not path.is_file():
            reasons.append("runtime_missing")
        return {
            "ready": not reasons,
            "reasons": reasons,
            "profile": profile.public(),
            "adapter": asdict(spec),
            "runtime": {"path": str(path), "sha256": file_hash(path) if path.is_file() else None},
        }

    def resolve(self, profile, runtimes):
        check = self.preflight(profile, runtimes)
        if not check["ready"]:
            raise ValueError("Harness profile rejected: " + ", ".join(check["reasons"]))
        module, name = self._specs[profile.harness].factory.split(":")
        return getattr(importlib.import_module(module), name)(runtimes), check


def builtin_registry():
    registry = HarnessRegistry()
    controls = ("mcp", "filesystem_isolation", "request_gateway", "events")
    registry.register(
        AdapterSpec(
            "kimi-code",
            "agent.harness.adapters:KimiAdapter",
            ("science_agent",),
            ("chat_completions",),
            controls,
            "kimi_bin",
        )
    )
    registry.register(
        AdapterSpec(
            "deepseek-harness",
            "agent.harness.adapters:DeepSeekAdapter",
            ("science_agent",),
            ("chat_completions",),
            controls,
            "dsh_bin",
        )
    )
    registry.register(
        AdapterSpec(
            "role-graph",
            "agent.harness.adapters:ScientistAdapter",
            ("science_expert",),
            ("chat_completions",),
            controls,
            "python_bin",
        )
    )
    registry.register(
        AdapterSpec(
            "langgraph",
            "agent.harness.adapters:LangGraphAdapter",
            ("science_agent", "science_expert"),
            ("chat_completions",),
            controls,
            "python_bin",
        )
    )
    for name, runtime in [("expert-kimi-code", "kimi_bin"), ("expert-deepseek-harness", "dsh_bin")]:
        registry.register(
            AdapterSpec(
                name,
                "agent.harness.native_expert_team:NativeExpertTeamAdapter",
                ("science_expert",),
                ("chat_completions",),
                controls,
                runtime,
            )
        )
    # Reserved compatibility target; do not mistake a registry entry for support.
    registry.register(
        AdapterSpec("codex", None, ("science_agent",), ("responses",), (), "codex_bin", status="integration_pending")
    )
    return registry
