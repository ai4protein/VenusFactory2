"""Harness-independent MCP access to the unified platform/scientific skill registry.

Thin adapter over ``agent.skills`` envelopes — same payloads as LangChain
``read_skill`` used by Science Expert.
"""
from __future__ import annotations

import json
from typing import Optional

from fastmcp import FastMCP

from agent.skills import build_list_skills_response, build_read_skill_response

mcp = FastMCP("Venus_Skill_MCP")


@mcp.tool(name="read_skill")
def mcp_read_skill(skill_id: str, relative_path: Optional[str] = None) -> str:
    """Read a platform or scientific-agent-skills package file (default SKILL.md).

    Use the public ID from ``list_skills``: native and upstream packages share
    one registry (e.g. ``nature_figure``, ``scanpy``, ``scikit-learn``).
    Native IDs win collisions; upstream equivalents use ``sas_<id>`` such as
    ``sas_biopython``. Reading a skill does not execute its bundled scripts.

    Optional ``relative_path`` loads progressive-disclosure files such as
    ``manifest.yaml``, ``references/legacy_guide.md``, or
    ``_shared_nature/core/ethics.md``.

    Returns JSON: success, skill_id, skill_root, relative_path, content, available_ids.
    """
    return json.dumps(
        build_read_skill_response(skill_id, relative_path),
        ensure_ascii=False,
    )


@mcp.tool(name="list_skills")
def mcp_list_skills() -> str:
    """Discover platform and scientific-agent-skills in one catalog.

    Prefer this before ``read_skill`` when you are unsure which skill_id to load.
    """
    return json.dumps(build_list_skills_response(), ensure_ascii=False)
