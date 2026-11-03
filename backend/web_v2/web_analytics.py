"""Public, online-only Google Analytics configuration (never API credentials)."""
import os
import re

GA_MEASUREMENT_ID_ENV = "WEBUI_V2_GA_MEASUREMENT_ID"
_MEASUREMENT_ID = re.compile(r"G-[A-Z0-9]{4,32}")


def validate_measurement_id(value: str) -> str:
    value = value.strip()
    if value and not _MEASUREMENT_ID.fullmatch(value):
        raise ValueError("Google Analytics measurement ID must use the G-... format, or be empty.")
    return value


def public_analytics_config(mode: str) -> dict[str, str]:
    measurement_id = ""
    if mode == "online":
        try:
            measurement_id = validate_measurement_id(os.getenv(GA_MEASUREMENT_ID_ENV, ""))
        except ValueError:
            # Invalid deployment configuration must not break the application.
            pass
    return {"google_analytics_id": measurement_id}
