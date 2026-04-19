"""Async webhook dispatcher for feedback and conversation events."""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Dict

from config import get_config
from logger import get_logger

_logger = get_logger("web_v2.feedback_webhook")


def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()


async def dispatch_webhook(event_type: str, data: Dict[str, Any]) -> bool:
    cfg = get_config().feedback
    if not cfg.webhook_url:
        return False

    payload = json.dumps(
        {"event": event_type, "data": data},
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "X-Venus-Event": event_type,
    }
    if cfg.webhook_secret:
        headers["X-Venus-Signature"] = _sign_payload(payload, cfg.webhook_secret)

    try:
        import httpx

        async with httpx.AsyncClient(timeout=cfg.webhook_timeout) as client:
            resp = await client.post(cfg.webhook_url, content=payload, headers=headers)
            if resp.status_code >= 400:
                _logger.warning("Webhook %s returned %d", event_type, resp.status_code)
                return False
            return True
    except ImportError:
        import urllib.request

        req = urllib.request.Request(
            cfg.webhook_url, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.webhook_timeout) as resp:
                return resp.status < 400
        except Exception as e:
            _logger.warning("Webhook %s failed (urllib): %s", event_type, e)
            return False
    except Exception as e:
        _logger.warning("Webhook %s failed: %s", event_type, e)
        return False
