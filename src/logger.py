"""
Centralized logging for VenusFactory.

Single entry point: ``from logger import logger``.
Environment controls:
  VENUS_LOG_LEVEL          – DEBUG / INFO / WARNING / ERROR (default: INFO)
  VENUS_DONT_LOG_MODEL_DATA – 1/true to suppress LLM payloads (default: true)
  VENUS_LOG_FORMAT         – "json" for structured output, anything else for text
"""
from __future__ import annotations

import logging
import os
import sys


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes")


def _env_log_level() -> int:
    raw = os.getenv("VENUS_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, raw, logging.INFO)


DONT_LOG_MODEL_DATA: bool = _env_bool("VENUS_DONT_LOG_MODEL_DATA", default=True)


class _SensitiveFilter(logging.Filter):
    """Suppress records tagged with `model_data=True` when sensitivity mode is on."""

    def filter(self, record: logging.LogRecord) -> bool:
        if DONT_LOG_MODEL_DATA and getattr(record, "model_data", False):
            return False
        return True


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    fmt_style = os.getenv("VENUS_LOG_FORMAT", "text").strip().lower()
    if fmt_style == "json":
        try:
            import json

            class _JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:
                    payload = {
                        "ts": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "msg": record.getMessage(),
                    }
                    if record.exc_info and record.exc_info[1]:
                        payload["exception"] = str(record.exc_info[1])
                    return json.dumps(payload, ensure_ascii=False)

            handler.setFormatter(_JsonFormatter())
        except Exception:
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    return handler


def setup_logging() -> None:
    """Idempotent logging bootstrap – safe to call multiple times."""
    root = logging.getLogger("venus")
    if root.handlers:
        return
    root.setLevel(_env_log_level())
    handler = _build_handler()
    handler.addFilter(_SensitiveFilter())
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``venus`` namespace.

    Usage::

        from logger import get_logger
        logger = get_logger(__name__)
    """
    setup_logging()
    return logging.getLogger(f"venus.{name}")


logger = get_logger("core")
