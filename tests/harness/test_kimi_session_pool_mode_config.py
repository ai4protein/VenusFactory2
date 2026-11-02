from pathlib import Path
from agent.kimi_session_pool import _seed_kimi_config


def test_session_configuration_uses_mode_home(monkeypatch, tmp_path):
    host = tmp_path / "host"
    mode = tmp_path / "mode"
    host.mkdir(); mode.mkdir()
    (host / "mcp.json").write_text("host")
    (mode / "mcp.json").write_text("mode")
    monkeypatch.setenv("KIMI_CONFIG_DIR", str(host))
    monkeypatch.setenv("KIMI_CODE_HOME", str(mode))
    session = tmp_path / "session"
    _seed_kimi_config(str(session))
    assert (session / ".kimi-code/mcp.json").read_text() == "mode"
