"""Serializable public contracts. Credentials are resolved only by the host."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelBinding:
    provider: str
    model: str
    base_url: str
    protocol: str = "chat_completions"
    credential_env: str = "VENUS_RESEARCH_API_KEY"
    context_tokens: int = 131072

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Model endpoint must have no embedded credentials, query or fragment")
        if self.protocol not in {"chat_completions", "responses"}:
            raise ValueError("Unknown model wire protocol")
        if not self.provider or not self.model:
            raise ValueError("Provider and exact model alias are required")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.credential_env):
            raise ValueError("Use an environment-variable reference, never an inline credential")
        if type(self.context_tokens) is not int or self.context_tokens <= 0:
            raise ValueError("Invalid context capacity")


@dataclass(frozen=True)
class Budget:
    wall_seconds: float = 900
    request_seconds: float = 300
    model_calls: int = 24
    tool_calls: int = 12
    output_tokens: int = 12288
    temperature: float | None = 0.1
    thinking: str | None = "enabled"
    reasoning_effort: str | None = None

    def __post_init__(self):
        for name in ["wall_seconds", "request_seconds"]:
            n = getattr(self, name)
            if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n <= 0:
                raise ValueError("Invalid time budget")
        for name in ["model_calls", "tool_calls", "output_tokens"]:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("Invalid count budget")
        if self.temperature is not None and (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("Invalid temperature")
        if self.thinking not in {None, "enabled", "disabled"}:
            raise ValueError("Unsupported common thinking policy")
        if self.reasoning_effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
            raise ValueError("Unsupported explicit reasoning effort")


@dataclass(frozen=True)
class Profile:
    mode: str
    harness: str
    model: ModelBinding
    budget: Budget = field(default_factory=Budget)
    required_controls: tuple[str, ...] = ("mcp", "filesystem_isolation", "request_gateway", "events")
    request_recovery: str = "none"

    def __post_init__(self):
        if self.mode not in {"science_agent", "science_expert"}:
            raise ValueError("Unknown scientific mode")

        if self.request_recovery not in {"none", "dmx_maas_once_v1", "empty_length_once_v1"}:
            raise ValueError("Unknown request recovery policy")
        if self.request_recovery == "dmx_maas_once_v1" and (
            self.model.provider != "dmx" or urlsplit(self.model.base_url).hostname != "www.dmxapi.cn"
        ):
            raise ValueError("MaaS recovery requires the observed DMX provider endpoint")

        if self.request_recovery == "empty_length_once_v1" and self.model.protocol != "chat_completions":
            raise ValueError("Empty-length recovery requires chat_completions")

    def public(self):
        value = asdict(self)
        if value["budget"]["reasoning_effort"] is None:
            value["budget"].pop("reasoning_effort")
        if self.request_recovery == "none":
            value.pop("request_recovery")
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["model"] = ModelBinding(**value["model"])
        value["budget"] = Budget(**value.get("budget", {}))
        if "required_controls" in value:
            value["required_controls"] = tuple(value["required_controls"])
        return cls(**value)


@dataclass(frozen=True)
class Task:
    workspace: Path
    capability: Path
    output: Path
    prompt: str
    instructions: str
    tools: frozenset[str]
    mcp_entry: str = "generated_tools.py"
    phase: str = "develop"
    interaction: str | None = None

    def __post_init__(self):
        if (not self.tools and self.phase != "chat") or any(not re.fullmatch(r"[a-zA-Z0-9_]+", t) for t in self.tools):
            raise ValueError("Scientific tools need explicit stable names")
        if Path(self.mcp_entry).name != self.mcp_entry or not self.mcp_entry.endswith(".py"):
            raise ValueError("MCP entry must be a bundle-local Python filename")
        if self.phase not in {"develop", "reuse", "probe", "chat"}:
            raise ValueError("Unknown phase")
        if self.interaction is not None and (
            self.interaction != "research_execution" or self.phase != "chat"
            or self.mcp_entry != "web_tools.py" or not self.tools
        ):
            raise ValueError("Research interaction requires explicit Web MCP tools")

    @property
    def interaction_mode(self):
        # Phase continues to select the existing transport and sandbox policy.
        return self.interaction or ("chat" if self.phase == "chat" else "bounded")

    def binding(self):
        return {
            "prompt_sha256": digest(self.prompt),
            "instructions_sha256": digest(self.instructions),
            "tools": sorted(self.tools),
            "mcp_entry": self.mcp_entry,
            "phase": self.phase,
            **({"interaction": self.interaction} if self.interaction is not None else {}),
            "inputs": {
                str(p.relative_to(self.workspace)): file_hash(p)
                for p in sorted(self.workspace.rglob("*"))
                if p.is_file()
            },
            "capabilities": {
                str(p.relative_to(self.capability)): file_hash(p)
                for p in sorted(self.capability.rglob("*"))
                if p.is_file()
            },
        }
