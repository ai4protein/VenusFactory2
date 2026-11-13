"""Offline catalog for report protocol tests; no provider calls or biological claims."""
import json

import pytest


def install_catalog(monkeypatch, tmp_path, identities=("writer", "reviewer", "other")):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("VENUS_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    monkeypatch.setenv("VENUS_TEST_REVIEW_KEY", "offline-fixture-only")
    monkeypatch.delenv("VENUS_REPORT_REVIEWER_MODEL_ID", raising=False)
    monkeypatch.delenv("VENUS_CAPABILITY_POLICY_SNAPSHOT", raising=False)
    path = tmp_path / "review-catalog.json"
    path.write_text(json.dumps({"models": [
        {"id": name, "label": name, "binding": {
            "provider": "fixture", "model": name + "-model", "base_url": "https://models.example.test/v1",
            "protocol": "chat_completions", "credential_env": "VENUS_TEST_REVIEW_KEY"},
         "pricing": {"input_per_million": 0, "output_per_million": 0}, "max_output_tokens": 4096}
        for name in identities]}))
    monkeypatch.setenv("VENUS_EXECUTION_CATALOG", str(path))
    return path


@pytest.fixture(autouse=True)
def reviewer_catalog(monkeypatch, tmp_path):
    """Opt-in by importing this fixture in legacy report protocol test modules."""
    return install_catalog(monkeypatch, tmp_path, ("fixture", "free", "free-model", "mock", "unused"))
