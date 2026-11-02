"""Explicit, immutable Scientist strategies and evidence of transmitted text.

Policies are data in the admitted capability bundle, never executable plugins.
Absence preserves the legacy four-role instructions. A malformed policy cannot
silently fall back. Request witnesses contain hashes, not scientific input text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .contracts import digest

FILENAME = "scientist_policy.json"
ROLES = ("PI", "CB", "MLS", "SC")
_BLOCK = re.compile(
    r"(?:^|\n)<venus-scientist-policy role=\"(PI|CB|MLS|SC)\" sha256=\"([a-f0-9]{64})\">\n"
    r"(.*?)\n</venus-scientist-policy>(?=\n|$)",
    re.DOTALL,
)


def validate_policy(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "policy_id", "common", "roles"}:
        raise ValueError("Invalid Scientist policy fields")
    if value["schema_version"] != "scientist_policy.v1":
        raise ValueError("Unknown Scientist policy schema")
    if not isinstance(value["policy_id"], str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,95}", value["policy_id"]):
        raise ValueError("A versioned Scientist policy identity is required")
    if not isinstance(value["roles"], dict) or set(value["roles"]) != set(ROLES):
        raise ValueError("All four Scientist roles are required")
    for text in [value["common"], *value["roles"].values()]:
        if not isinstance(text, str) or not text.strip() or len(text) > 16000 or "venus-scientist-policy" in text:
            raise ValueError("Scientist instructions must be bounded plain text")
    return value


def load_policy(capability):
    path = Path(capability) / FILENAME
    if path.is_symlink():
        raise ValueError("Symlinked Scientist policy")
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > 100000:
        raise ValueError("Invalid Scientist policy file")
    return validate_policy(json.loads(path.read_text()))


def policy_binding(policy):
    return {"policy_id": policy["policy_id"], "sha256": digest(policy)}


def role_instruction(policy, role):
    validate_policy(policy)
    if role not in ROLES:
        raise ValueError("Unknown Scientist role")
    body = policy["common"] + "\n\n" + policy["roles"][role]
    return f'<venus-scientist-policy role="{role}" sha256="{digest(policy)}">\n' + body + "\n</venus-scientist-policy>"


def _texts(content):
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") in {"text", "input_text"}
                and isinstance(item.get("text"), str)
            ):
                yield item["text"]


def request_policy_witness(payload):
    """Inspect only instruction/user fields; exclude model output and tool results."""
    texts = [("instructions", t) for t in _texts(payload.get("instructions"))]
    for key in ("messages", "input"):
        messages = payload.get(key, [])
        if isinstance(messages, str):
            texts.append((key + ":user", messages))
        elif isinstance(messages, list):
            for item in messages:
                if isinstance(item, dict) and item.get("role") in {"system", "developer", "user"}:
                    texts.extend((key + ":" + item["role"], t) for t in _texts(item.get("content")))
    blocks = []
    for field, text in texts:
        for match in _BLOCK.finditer(text):
            role, sha, body = match.groups()
            block = {"role": role, "policy_sha256": sha, "body_sha256": digest(body), "field": field}
            if block not in blocks:
                blocks.append(block)
    return {"scientist_policy_blocks": blocks} if blocks else {}


def verify_request_policy(request, policy, role):
    expected = request_policy_witness({"instructions": role_instruction(policy, role)})["scientist_policy_blocks"][0]
    keys = ("role", "policy_sha256", "body_sha256")
    if not any(
        all(block.get(k) == expected[k] for k in keys)
        for block in request.get("scientist_policy_blocks", [])
        if isinstance(block, dict)
    ):
        raise ValueError("Scientist policy was not witnessed in the actual model request")
