import asyncio
import hashlib
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from backend.web_v2 import settings_api as settings


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / '.env'
    example = tmp_path / '.env.example'
    example.write_text('# === Models ===\nCHAT_MAX_TOKENS=4096\nDMXAPI_API_KEY=\nWEBUI_V2_DEV_MODE=false\n')
    path.write_text('# User configuration\nCHAT_MAX_TOKENS=2048\nDMXAPI_API_KEY=test-secret\nPRIVATE_ROUTING=keep-me\n')
    monkeypatch.setattr(settings, '_ENV_PATH', path)
    monkeypatch.setattr(settings, '_ENV_EXAMPLE_PATH', example)
    monkeypatch.delenv('WEBUI_V2_ALLOW_REMOTE_SETTINGS', raising=False)
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    return path


def request(host='127.0.0.1'):
    return Request({'type': 'http', 'client': (host, 1234), 'method': 'POST', 'path': '/api/settings/env', 'headers': []})


def save(entries, revision=None, host='127.0.0.1'):
    return asyncio.run(settings.save_env_entries(settings.UpdateEnvRequest(entries=entries, revision=revision), request(host)))


def test_partial_update_preserves_credentials_unknown_keys_and_comments(env):
    result = save([{'key': 'CHAT_MAX_TOKENS', 'value': '3072'}])
    assert env.read_text() == '# User configuration\nCHAT_MAX_TOKENS=3072\nDMXAPI_API_KEY=test-secret\nPRIVATE_ROUTING=keep-me\n'
    assert result['revision'] == hashlib.sha256(env.read_bytes()).hexdigest()


def test_stale_save_is_rejected_without_touching_file(env):
    loaded = asyncio.run(settings.get_env_entries(request()))
    env.write_text(env.read_text() + 'ADDED_LATER=yes\n')
    before = env.read_bytes()
    with pytest.raises(HTTPException) as error:
        save([{'key': 'CHAT_MAX_TOKENS', 'value': '1'}], loaded['revision'])
    assert error.value.status_code == 409
    assert env.read_bytes() == before


@pytest.mark.parametrize('value', ['good\nINJECTED=yes', 'good\rINJECTED=yes', 'a\x00b'])
def test_multiline_values_cannot_inject_settings(env, value):
    before = env.read_bytes()
    with pytest.raises(HTTPException) as error:
        save([{'key': 'CHAT_MAX_TOKENS', 'value': value}])
    assert error.value.status_code == 400
    assert env.read_bytes() == before


def test_empty_value_clears_only_requested_key_and_new_key_is_appended(env):
    save([{'key': 'DMXAPI_API_KEY', 'value': ''}, {'key': 'WEBUI_V2_DEV_MODE', 'value': 'true'}])
    assert 'DMXAPI_API_KEY=\n' in env.read_text()
    assert 'PRIVATE_ROUTING=keep-me\n' in env.read_text()
    assert env.read_text().endswith('WEBUI_V2_DEV_MODE=true\n')


def test_unknown_key_and_remote_access_still_rejected(env):
    before = env.read_bytes()
    with pytest.raises(HTTPException) as error:
        save([{'key': 'UNKNOWN', 'value': '1'}])
    assert error.value.status_code == 400
    with pytest.raises(HTTPException) as error:
        save([{'key': 'CHAT_MAX_TOKENS', 'value': '1'}], host='198.51.100.1')
    assert error.value.status_code == 403
    assert env.read_bytes() == before


def test_duplicate_definitions_and_file_permissions(env):
    env.write_text('export CHAT_MAX_TOKENS=2048\nCHAT_MAX_TOKENS=1024\nPRIVATE_ROUTING=keep-me\n')
    env.chmod(0o600)
    save([{'key': 'CHAT_MAX_TOKENS', 'value': '3072'}])
    assert env.read_text().startswith('export CHAT_MAX_TOKENS=3072\nCHAT_MAX_TOKENS=3072\n')
    assert env.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('allow_remote', ['0', '1'])
@pytest.mark.parametrize('host', ['127.0.0.1', '198.51.100.1'])
def test_online_rejects_read_and_save_even_with_remote_access_enabled(env, monkeypatch, allow_remote, host):
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    monkeypatch.setenv('WEBUI_V2_ALLOW_REMOTE_SETTINGS', allow_remote)
    before = env.read_bytes()
    with pytest.raises(HTTPException) as read_error:
        asyncio.run(settings.get_env_entries(request(host)))
    assert read_error.value.status_code == 403
    with pytest.raises(HTTPException) as save_error:
        save([{'key': 'CHAT_MAX_TOKENS', 'value': '1'}], host=host)
    assert save_error.value.status_code == 403
    assert env.read_bytes() == before


@pytest.mark.parametrize('key,value', [
    ('WEBUI_V2_AGENT_CONTEXT_TOKENS','1024'),
    ('WEBUI_V2_AGENT_RECENT_TURNS','0'),
    ('WEBUI_V2_AGENT_COMPACT_PERCENT','100'),
    ('WEBUI_V2_AGENT_SUMMARY_TOKENS','abc'),
    ('WEBUI_V2_AGENT_MAX_TURNS','-1'),
])
def test_context_settings_reject_invalid_values_without_changing_env(env,key,value):
    settings._ENV_EXAMPLE_PATH.write_text(settings._ENV_EXAMPLE_PATH.read_text()+key+'=\n')
    before=env.read_bytes()
    with pytest.raises(HTTPException) as error:save([{'key':key,'value':value}])
    assert error.value.status_code==400 and env.read_bytes()==before


def test_context_settings_save_zero_and_blank_defaults(env):
    key='WEBUI_V2_AGENT_MAX_TURNS'
    settings._ENV_EXAMPLE_PATH.write_text(settings._ENV_EXAMPLE_PATH.read_text()+key+'=0\n')
    save([{'key':key,'value':'0'}])
    assert key+'=0\n' in env.read_text()
    save([{'key':key,'value':''}])
    assert key+'=\n' in env.read_text()
