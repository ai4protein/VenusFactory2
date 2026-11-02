"""Current outbound research authorization, independent of historical profiles.

Missing or malformed authorization fails closed. Reading profiles, replaying
numerical evidence and validating historical experiments do not consult it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

POLICY_PATH = Path(__file__).resolve().parents[4] / "configs/rsi/model_access.json"


class ModelAccessDenied(ValueError):
    """A live model request is not authorized by the current research policy."""


def authorize_model(*, model, base_url, protocol, output_tokens):
    from agent.harness.provider_config import authorize_local
    try:
        configured = authorize_local(model, base_url, protocol, output_tokens)
    except (ValueError, TypeError, AttributeError):
        raise ModelAccessDenied("Invalid local provider configuration or request limits") from None
    if configured is not None:
        return configured
    try:
        raw = POLICY_PATH.read_bytes()
        policy = json.loads(raw)
        if policy.get("schema") == 2:
            if set(policy) != {"schema", "policy_id", "catalog"}:
                raise ValueError
            from agent.harness.free_catalog import free_routes, catalog_path
            import os
            path = catalog_path() if os.environ.get("VENUS_EXECUTION_CATALOG") else (POLICY_PATH.parent / policy["catalog"]).resolve()
            routes, catalog_sha = free_routes(path)
            raw += catalog_sha.encode()
            policy = {"schema": 1, "policy_id": policy["policy_id"], "pricing_source": "server_execution_catalog", "allowed": routes}
        if (
            set(policy) != {"schema", "policy_id", "pricing_source", "allowed"}
            or type(policy["schema"]) is not int
            or policy["schema"] != 1
            or not isinstance(policy["policy_id"], str)
            or not policy["policy_id"]
            or not isinstance(policy["pricing_source"], str)
            or not isinstance(policy["allowed"], list)
            or not policy["allowed"]
        ):
            raise ValueError
        identities = set()
        for rule in policy["allowed"]:
            if set(rule) != {"model", "base_url", "protocol", "max_output_tokens"}:
                raise ValueError
            url = urlsplit(rule["base_url"])
            if (
                url.scheme != "https"
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or url.port not in {None, 443}
                or not isinstance(rule["model"], str)
                or not rule["model"]
                or rule["protocol"] not in {"chat_completions", "responses"}
                or type(rule["max_output_tokens"]) is not int
                or rule["max_output_tokens"] < 1
            ):
                raise ValueError
            identity = (rule["model"], rule["base_url"].rstrip("/"), rule["protocol"])
            if identity in identities:
                raise ValueError
            identities.add(identity)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise ModelAccessDenied("Research model authorization is missing or invalid") from None
    if not isinstance(base_url, str) or type(output_tokens) is not int or output_tokens < 1:
        raise ModelAccessDenied("Invalid research model request limits")
    matching = [
        rule
        for rule in policy["allowed"]
        if (rule["model"], rule["base_url"].rstrip("/"), rule["protocol"]) == (model, base_url.rstrip("/"), protocol)
    ]
    if not matching:
        raise ModelAccessDenied("Research model, endpoint or protocol is not authorized")
    from .openrouter_policy import free_model_id, is_openrouter

    if is_openrouter(base_url) and (protocol not in {"chat_completions", "responses"} or not free_model_id(model)):
        raise ModelAccessDenied("Only explicit free OpenRouter routes are supported")
    if output_tokens > matching[0]["max_output_tokens"]:
        raise ModelAccessDenied("Research model output exceeds the authorized ceiling")
    return {
        "policy_id": policy["policy_id"],
        "policy_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "live_research_request",
    }


def authorize_profile(profile):
    return authorize_model(
        model=profile.model.model,
        base_url=profile.model.base_url,
        protocol=profile.model.protocol,
        output_tokens=profile.budget.output_tokens,
    )
