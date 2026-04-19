"""Tests for the typed configuration system."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from config import (
    ServerConfig,
    LLMConfig,
    AgentConfig,
    OnlineLimitsConfig,
    StorageConfig,
    MCPConfig,
    VenusConfig,
    get_config,
    set_config,
)


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.mode == "local"
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 7861
        assert cfg.is_online is False

    def test_online_mode(self):
        cfg = ServerConfig(mode="online")
        assert cfg.is_online is True

    def test_invalid_mode_coerced(self):
        cfg = ServerConfig(mode="invalid")
        assert cfg.mode == "local"

    def test_invalid_port(self):
        with pytest.raises(ValueError, match="port"):
            ServerConfig(port=0)

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("WEBUI_V2_MODE", "online")
        monkeypatch.setenv("WEBUI_V2_PORT", "9000")
        cfg = ServerConfig.from_env()
        assert cfg.is_online
        assert cfg.port == 9000

    def test_frozen(self):
        cfg = ServerConfig()
        with pytest.raises(AttributeError):
            cfg.mode = "online"  # type: ignore[misc]


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 8192

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(temperature=3.0)

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(max_tokens=0)

    def test_with_overrides(self):
        base = LLMConfig(model_name="gpt-4")
        overridden = base.with_overrides(model_name="claude-3", temperature=0.5)
        assert overridden.model_name == "claude-3"
        assert overridden.temperature == 0.5
        assert base.model_name == "gpt-4"

    def test_with_overrides_none_ignored(self):
        base = LLMConfig(model_name="gpt-4")
        overridden = base.with_overrides(model_name=None)
        assert overridden.model_name == "gpt-4"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CHAT_MODEL_NAME", "test-model")
        cfg = LLMConfig.from_env()
        assert cfg.api_key == "sk-test"
        assert cfg.model_name == "test-model"


class TestAgentConfig:
    def test_defaults_local(self):
        cfg = AgentConfig.from_env(online=False)
        assert cfg.max_messages == float("inf")
        assert cfg.max_tool_calls == float("inf")

    def test_defaults_online(self):
        cfg = AgentConfig.from_env(online=True)
        assert cfg.max_messages == 200.0
        assert cfg.max_tool_calls == 500.0

    def test_invalid_retries(self):
        with pytest.raises(ValueError, match="max_step_retries"):
            AgentConfig(max_step_retries=-1)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="tool_execution_timeout"):
            AgentConfig(tool_execution_timeout=0)

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_STEP_RETRIES", "5")
        monkeypatch.setenv("TOOL_EXECUTION_TIMEOUT", "120")
        cfg = AgentConfig.from_env()
        assert cfg.max_step_retries == 5
        assert cfg.tool_execution_timeout == 120


class TestOnlineLimitsConfig:
    def test_defaults(self):
        cfg = OnlineLimitsConfig()
        assert cfg.daily_chat_limit == 10
        assert cfg.session_token_ttl_hours == 24

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("WEBUI_V2_ONLINE_DAILY_CHAT_LIMIT", "50")
        cfg = OnlineLimitsConfig.from_env()
        assert cfg.daily_chat_limit == 50


class TestStorageConfig:
    def test_defaults(self):
        cfg = StorageConfig()
        assert cfg.temp_outputs_dir == "temp_outputs"
        assert cfg.enable_ttl_cleanup is False


class TestMCPConfig:
    def test_defaults(self):
        cfg = MCPConfig()
        assert cfg.port == 8080

    def test_invalid_port(self):
        with pytest.raises(ValueError, match="port"):
            MCPConfig(port=-1)


class TestVenusConfig:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("WEBUI_V2_MODE", "local")
        cfg = VenusConfig.from_env()
        assert cfg.server.mode == "local"
        assert cfg.agent.max_messages == float("inf")

    def test_online_cascades(self, monkeypatch):
        monkeypatch.setenv("WEBUI_V2_MODE", "online")
        cfg = VenusConfig.from_env()
        assert cfg.server.is_online
        assert cfg.agent.max_messages == 200.0


class TestGlobalConfig:
    def test_set_and_get(self):
        custom = VenusConfig(server=ServerConfig(port=9999))
        set_config(custom)
        assert get_config().server.port == 9999
        set_config(None)  # type: ignore[arg-type]
