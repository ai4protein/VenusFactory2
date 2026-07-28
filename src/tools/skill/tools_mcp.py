"""Skill MCP tools for Science Agent (kimi-code).

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
    """Read a VenusFactory skill package file (default SKILL.md).

    ``skill_id`` is the directory name under ``src/agent/skills/``
    (e.g. ``nature_figure``, ``zero_shot_mutation_workflow``, ``uniprot_database``).

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
    """List VenusFactory skill_ids with short descriptions for routing.

    Prefer this before ``read_skill`` when you are unsure which skill_id to load.
    """
    return json.dumps(build_list_skills_response(), ensure_ascii=False)
