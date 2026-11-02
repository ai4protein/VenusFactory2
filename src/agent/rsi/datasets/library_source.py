"""Resolve a source handle only inside the host-admitted task projection."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from agent.rsi.engine.experience import validate_projection
    from agent.rsi.engine.protein_model import digest
except ImportError:  # frozen standalone MCP bundle
    from experience import validate_projection
    from protein_model import digest

REFIT_TOOL = "evaluate_library_model_code"


def resolve_library_source(capability, task, experience_id):
    """Return unfitted source and its exact admission binding; never load weights."""
    if task.get("phase") != "develop" or "library_experience_sha256" not in task:
        raise ValueError("No library source admitted for this episode")
    if not isinstance(experience_id, str) or len(experience_id) != 64 or set(experience_id) - set("0123456789abcdef"):
        raise ValueError("Invalid library experience ID")
    path = Path(capability) / "library_experience.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked library projection")
    packet = json.loads(path.read_text())
    if digest(packet) != task["library_experience_sha256"]:
        raise ValueError("Frozen library experience changed")
    validate_projection(packet)
    if packet["condition"] != "source":
        raise ValueError("Library source is unavailable in this condition")
    matches = [entry for entry in packet["entries"] if entry["experience_id"] == experience_id]
    if len(matches) != 1:
        raise ValueError("Source handle is not in the admitted library projection")
    entry = matches[0]
    return entry["source_code"], {
        "kind": "admitted_library_source",
        "experience_id": experience_id,
        "projection_sha256": digest(packet),
        "source_sha256": entry["source_sha256"],
    }


def validate_source_origin(capability, task, source, origin):
    if not isinstance(origin, dict):
        raise ValueError("Invalid library source origin")
    expected_source, expected_origin = resolve_library_source(capability, task, origin.get("experience_id"))
    if source != expected_source or origin != expected_origin:
        raise ValueError("Library source origin differs from admitted projection")
    return origin
