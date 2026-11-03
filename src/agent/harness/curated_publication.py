"""Operator-only publication. This module is deliberately not an Agent/MCP tool."""
import re
import time
from pathlib import Path

from .contracts import digest
from .research_workspace import LIBRARY, publish_proposals


def publication_record(manifest, receipt, goal_id, *, author_email, curator):
    email = str(author_email).strip()
    curator = str(curator).strip()
    if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,63}", email) or len(email) > 254:
        raise ValueError("Provide a valid author email")
    if not curator or len(curator) > 200 or any(ord(c) < 32 for c in curator):
        raise ValueError("Provide the curator identity")
    data = {"candidate_id": manifest["id"], "receipt_sha256": receipt["receipt_sha256"],
            "goal_id": goal_id, "author_email": email, "curator": curator,
            "published_at": time.time(), "policy": "operator_selected.v1"}
    return {**data, "sha256": digest(data)}


def validate_publication(value, manifest, receipt, goal_id):
    if not isinstance(value, dict) or value.get("policy") != "operator_selected.v1":
        raise ValueError("Public registration requires an operator publication decision")
    if (value.get("candidate_id"), value.get("receipt_sha256"), value.get("goal_id")) != (manifest["id"], receipt["receipt_sha256"], goal_id):
        raise ValueError("Publication decision does not match the reviewed candidate")
    if digest({k: v for k, v in value.items() if k != "sha256"}) != value.get("sha256"):
        raise ValueError("Publication decision changed")
    publication_record(manifest, receipt, goal_id, author_email=value.get("author_email", ""), curator=value.get("curator", ""))
    return value


def publish_candidate(bundle, goal_id, *, author_email, curator, library=LIBRARY):
    from mcp_tools.tools.runtime_tool_policy import is_online_mode
    from agent.rsi.engine.context import get_resources
    if is_online_mode() or get_resources() is not None:
        raise ValueError("Publication is a local operator action, unavailable to research sessions")
    import json
    from .capability_review import admitted
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    receipt = admitted(bundle, manifest)
    decision = publication_record(manifest, receipt, goal_id, author_email=author_email, curator=curator)
    return publish_proposals([bundle], goal_id, library=library, publication=decision)[0]
