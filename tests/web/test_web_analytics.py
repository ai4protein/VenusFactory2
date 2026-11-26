import pytest

from backend.web_v2.web_analytics import (
    GA_MEASUREMENT_ID_ENV, public_analytics_config, validate_measurement_id,
)


@pytest.mark.parametrize("mode", ["local", "", "unknown"])
def test_local_and_unknown_modes_never_expose_a_tracking_id(monkeypatch, mode):
    monkeypatch.setenv(GA_MEASUREMENT_ID_ENV, "G-TEST123456")
    assert public_analytics_config(mode) == {"google_analytics_id": ""}


def test_online_exposes_only_the_public_measurement_id(monkeypatch):
    monkeypatch.setenv(GA_MEASUREMENT_ID_ENV, "  G-TEST123456  ")
    monkeypatch.setenv("OPENAI_API_KEY", "private-fixture")
    assert public_analytics_config("online") == {"google_analytics_id": "G-TEST123456"}


@pytest.mark.parametrize("value", ["", "invalid", 'G-TEST123456"><script>', "G-TEST123456&x=1"])
def test_missing_or_invalid_configuration_disables_tracking(monkeypatch, value):
    monkeypatch.setenv(GA_MEASUREMENT_ID_ENV, value)
    assert public_analytics_config("online") == {"google_analytics_id": ""}


def test_admin_validation_rejects_malformed_ids_and_allows_disabling():
    assert validate_measurement_id("  ") == ""
    assert validate_measurement_id(" G-TEST123456 ") == "G-TEST123456"
    with pytest.raises(ValueError):
        validate_measurement_id("https://example.com/tag.js")


def test_environment_editor_validates_and_clears_tracking_id(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from backend.web_v2 import settings_api as settings
    env_path = tmp_path / ".env"
    template = tmp_path / ".env.example"
    env_path.write_text("UNCHANGED=yes\n")
    template.write_text(GA_MEASUREMENT_ID_ENV + "=\n")
    monkeypatch.setattr(settings, "_ENV_PATH", env_path)
    monkeypatch.setattr(settings, "_ENV_EXAMPLE_PATH", template)
    def save(value):
        settings.write_env_entries(settings.UpdateEnvRequest(
            entries=[{"key": GA_MEASUREMENT_ID_ENV, "value": value}]))
    with pytest.raises(HTTPException) as error:
        save("https://other.example/script")
    assert error.value.status_code == 400
    assert env_path.read_text() == "UNCHANGED=yes\n"
    save(" G-TEST123456 ")
    assert env_path.read_text() == "UNCHANGED=yes\n" + GA_MEASUREMENT_ID_ENV + "=G-TEST123456\n"
    save("")
    assert env_path.read_text() == "UNCHANGED=yes\n" + GA_MEASUREMENT_ID_ENV + "=\n"
