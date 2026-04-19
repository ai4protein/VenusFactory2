"""
Typed configuration system for VenusFactory.

Three-layer resolution: env vars -> defaults -> runtime overrides.
Uses dataclasses with ``__post_init__`` validation, following the
OpenAI Agents SDK pattern.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int = 0) -> int:
    raw = _env(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes")


def _env_float(key: str, default: float = 0.0) -> float:
    raw = _env(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    mode: str = "local"
    host: str = "0.0.0.0"
    port: int = 7861
    cors_origins: list[str] = field(default_factory=list)
    allow_remote_settings: bool = False
    frontend_dist: str = "frontend/dist"
    dev_mode: bool = False
    frontend_dev_url: str = "http://127.0.0.1:5173"

    def __post_init__(self) -> None:
        if self.mode not in ("local", "online"):
            object.__setattr__(self, "mode", "local")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"ServerConfig.port must be 1–65535, got {self.port}")

    @property
    def is_online(self) -> bool:
        return self.mode == "online"

    @classmethod
    def from_env(cls) -> ServerConfig:
        cors_raw = _env("WEBUI_V2_CORS_ORIGINS",
                        "http://127.0.0.1:7861,http://localhost:7861,"
                        "http://127.0.0.1:5173,http://localhost:5173")
        origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
        return cls(
            mode=_env("WEBUI_V2_MODE", "local").lower(),
            host=_env("WEBUI_V2_HOST", "0.0.0.0"),
            port=_env_int("WEBUI_V2_PORT", 7861),
            cors_origins=origins,
            allow_remote_settings=_env_bool("WEBUI_V2_ALLOW_REMOTE_SETTINGS"),
            frontend_dist=_env("WEBUI_V2_FRONTEND_DIST", "frontend/dist"),
            dev_mode=_env_bool("WEBUI_V2_DEV_MODE"),
            frontend_dev_url=_env("WEBUI_V2_FRONTEND_DEV_URL", "http://127.0.0.1:5173"),
        )


# ---------------------------------------------------------------------------
# LLM / Chat configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://www.dmxapi.cn/v1"
    model_name: str = "gemini-2.5-pro"
    temperature: float = 0.2
    max_tokens: int = 8192
    code_max_tokens: int = 10000

    def __post_init__(self) -> None:
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"LLMConfig.temperature must be 0–2, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"LLMConfig.max_tokens must be >= 1, got {self.max_tokens}")

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            api_key=_env("OPENAI_API_KEY"),
            base_url=_env("CHAT_BASE_URL", "https://www.dmxapi.cn/v1"),
            model_name=_env("CHAT_MODEL_NAME", "gemini-2.5-pro"),
            temperature=_env_float("CHAT_TEMPERATURE", 0.2),
            max_tokens=_env_int("CHAT_MAX_TOKENS", 8192),
            code_max_tokens=_env_int("CHAT_CODE_MAX_TOKENS", 10000),
        )

    def with_overrides(self, **kwargs: Any) -> LLMConfig:
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **filtered)


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    max_messages: float = float("inf")
    max_tool_calls: float = float("inf")
    max_step_retries: int = 2
    tool_execution_timeout: int = 600
    search_max_results: int = 3

    def __post_init__(self) -> None:
        if self.max_step_retries < 0:
            raise ValueError(
                f"AgentConfig.max_step_retries must be >= 0, got {self.max_step_retries}"
            )
        if self.tool_execution_timeout < 1:
            raise ValueError(
                f"AgentConfig.tool_execution_timeout must be >= 1, got {self.tool_execution_timeout}"
            )

    @classmethod
    def from_env(cls, *, online: bool = False) -> AgentConfig:
        if online:
            max_messages = 200.0
            max_tool_calls = 500.0
        else:
            max_messages = float("inf")
            max_tool_calls = float("inf")
        return cls(
            max_messages=max_messages,
            max_tool_calls=max_tool_calls,
            max_step_retries=_env_int("MAX_STEP_RETRIES", 2),
            tool_execution_timeout=_env_int("TOOL_EXECUTION_TIMEOUT", 600),
            search_max_results=_env_int("SEARCH_MAX_RESULTS", 3),
        )


# ---------------------------------------------------------------------------
# Online-mode rate limiting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OnlineLimitsConfig:
    fasta_limit: int = 50
    sequence_design_limit: int = 50
    daily_chat_limit: int = 10
    session_token_ttl_hours: int = 24
    local_download_batch_limit: int = 0
    online_download_batch_limit: int = 50

    @classmethod
    def from_env(cls) -> OnlineLimitsConfig:
        return cls(
            fasta_limit=_env_int("WEBUI_V2_ONLINE_FASTA_LIMIT", 50),
            sequence_design_limit=_env_int("WEBUI_V2_ONLINE_SEQUENCE_DESIGN_LIMIT", 50),
            daily_chat_limit=max(1, _env_int("WEBUI_V2_ONLINE_DAILY_CHAT_LIMIT", 10)),
            session_token_ttl_hours=max(1, _env_int("WEBUI_V2_SESSION_TOKEN_TTL_HOURS", 24)),
            local_download_batch_limit=_env_int("WEBUI_V2_LOCAL_DOWNLOAD_BATCH_LIMIT", 0),
            online_download_batch_limit=_env_int("WEBUI_V2_ONLINE_DOWNLOAD_BATCH_LIMIT", 50),
        )


# ---------------------------------------------------------------------------
# Storage / temp outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageConfig:
    temp_outputs_dir: str = "temp_outputs"
    enable_ttl_cleanup: bool = False
    uploads_ttl_days: int = 7
    work_ttl_days: int = 2
    results_ttl_days: int = 14
    sessions_ttl_days: int = 3
    manifests_ttl_days: int = 30
    cache_ttl_days: int = 7

    @classmethod
    def from_env(cls) -> StorageConfig:
        return cls(
            temp_outputs_dir=_env("TEMP_OUTPUTS_DIR", "temp_outputs"),
            enable_ttl_cleanup=_env_bool("TEMP_OUTPUTS_ENABLE_TTL_CLEANUP"),
            uploads_ttl_days=_env_int("TEMP_OUTPUTS_UPLOADS_TTL_DAYS", 7),
            work_ttl_days=_env_int("TEMP_OUTPUTS_WORK_TTL_DAYS", 2),
            results_ttl_days=_env_int("TEMP_OUTPUTS_RESULTS_TTL_DAYS", 14),
            sessions_ttl_days=_env_int("TEMP_OUTPUTS_SESSIONS_TTL_DAYS", 3),
            manifests_ttl_days=_env_int("TEMP_OUTPUTS_MANIFESTS_TTL_DAYS", 30),
            cache_ttl_days=_env_int("TEMP_OUTPUTS_CACHE_TTL_DAYS", 7),
        )


# ---------------------------------------------------------------------------
# MCP configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedbackConfig:
    webhook_url: str = ""
    webhook_secret: str = ""
    collect_conversations: bool = True
    webhook_timeout: int = 10

    @classmethod
    def from_env(cls) -> FeedbackConfig:
        return cls(
            webhook_url=_env("VENUS_FEEDBACK_WEBHOOK_URL"),
            webhook_secret=_env("VENUS_FEEDBACK_WEBHOOK_SECRET"),
            collect_conversations=_env_bool("VENUS_FEEDBACK_COLLECT_CONVERSATIONS", True),
            webhook_timeout=_env_int("VENUS_FEEDBACK_WEBHOOK_TIMEOUT", 10),
        )


@dataclass(frozen=True)
class MCPConfig:
    host: str = "0.0.0.0"
    port: int = 8080

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"MCPConfig.port must be 1–65535, got {self.port}")

    @classmethod
    def from_env(cls) -> MCPConfig:
        return cls(
            host=_env("MCP_HTTP_HOST", "0.0.0.0"),
            port=_env_int("MCP_HTTP_PORT", 8080),
        )


# ---------------------------------------------------------------------------
# Top-level config aggregator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VenusConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    online_limits: OnlineLimitsConfig = field(default_factory=OnlineLimitsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    @classmethod
    def from_env(cls) -> VenusConfig:
        server = ServerConfig.from_env()
        return cls(
            server=server,
            llm=LLMConfig.from_env(),
            agent=AgentConfig.from_env(online=server.is_online),
            online_limits=OnlineLimitsConfig.from_env(),
            storage=StorageConfig.from_env(),
            mcp=MCPConfig.from_env(),
            feedback=FeedbackConfig.from_env(),
        )


# Singleton – lazily built on first access
_config: VenusConfig | None = None


def get_config() -> VenusConfig:
    """Return the global config singleton, building from env on first call."""
    global _config
    if _config is None:
        _config = VenusConfig.from_env()
    return _config


def set_config(config: VenusConfig) -> None:
    """Replace the global config (useful for testing)."""
    global _config
    _config = config
