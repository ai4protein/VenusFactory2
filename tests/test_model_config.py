"""model-config.yaml is re-read every call and supports multiple endpoints."""
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.model_config import load_model_config, runtime_model_public_fields


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "model-config.yaml"
    monkeypatch.setenv("VENUS_MODEL_CONFIG", str(path))
    return path


def test_example_file_has_dmx_gpt_claude():
    example = Path(__file__).resolve().parents[1] / "model-config.yaml.example"
    raw = yaml.safe_load(example.read_text(encoding="utf-8"))
    assert raw["active"] == "dmx"
    endpoints = raw["endpoints"]
    assert set(endpoints) >= {"dmx", "gpt", "claude"}
    assert endpoints["claude"]["api_compatible"] == "anthropic"
    assert endpoints["gpt"]["base_url"].startswith("https://api.openai.com")
    assert endpoints["dmx"]["base_url"] == "https://www.dmxapi.cn/v1"


def test_switch_active_endpoint(config_file: Path):
    config_file.write_text(
        yaml.safe_dump(
            {
                "active": "claude",
                "endpoints": {
                    "dmx": {
                        "model": "glm-4-flash",
                        "base_url": "https://www.dmxapi.cn/v1",
                        "api_compatible": "openai",
                        "api_key_env": "DMXAPI_API_KEY",
                    },
                    "claude": {
                        "model": "claude-3-7-sonnet-20250219",
                        "label": "Claude",
                        "provider": "anthropic",
                        "base_url": "https://api.anthropic.com/v1",
                        "api_compatible": "anthropic",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                    "gpt": {
                        "model": "gpt-4o",
                        "base_url": "https://api.openai.com/v1",
                        "api_compatible": "openai",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_model_config()
    assert cfg.endpoint_id == "claude"
    assert cfg.model == "claude-3-7-sonnet-20250219"
    assert cfg.base_url == "https://api.anthropic.com/v1"
    assert cfg.api_compatible == "anthropic"
    assert set(cfg.endpoint_ids()) == {"dmx", "claude", "gpt"}


def test_reload_picks_up_model_and_base_url(config_file: Path):
    config_file.write_text(
        "active: dmx\nendpoints:\n  dmx:\n    model: glm-4-flash\n    base_url: https://www.dmxapi.cn/v1\n",
        encoding="utf-8",
    )
    first = load_model_config()
    assert first.model == "glm-4-flash"
    assert first.base_url == "https://www.dmxapi.cn/v1"

    config_file.write_text(
        "active: gpt\nendpoints:\n  gpt:\n    model: gpt-4o-mini\n    base_url: https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    second = load_model_config()
    assert second.endpoint_id == "gpt"
    assert second.model == "gpt-4o-mini"
    assert second.base_url == "https://api.openai.com/v1"


def test_legacy_flat_file_still_works(config_file: Path):
    config_file.write_text(
        "model: glm-4-flash\nbase_url: https://www.dmxapi.cn/v1\nprovider: dmx\napi_compatible: openai\n",
        encoding="utf-8",
    )
    cfg = load_model_config()
    assert cfg.model == "glm-4-flash"
    assert cfg.provider == "dmx"
    assert cfg.base_url == "https://www.dmxapi.cn/v1"


def test_public_fields_hide_secrets(config_file: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-should-not-leak")
    config_file.write_text(
        "active: gpt\nendpoints:\n  gpt:\n    model: gpt-4o\n    base_url: https://api.openai.com/v1\n    api_key_env: OPENAI_API_KEY\n    api_key: sk-inline-secret\n",
        encoding="utf-8",
    )
    public = runtime_model_public_fields()
    blob = yaml.safe_dump(public)
    assert "sk-secret-should-not-leak" not in blob
    assert "sk-inline-secret" not in blob
    assert public["runtime_endpoint"] == "gpt"
    assert public["runtime_endpoints"][0]["has_key"] is True
