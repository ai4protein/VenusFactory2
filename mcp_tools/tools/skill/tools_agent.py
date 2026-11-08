"""
Skill tools for Science Expert (CB/MLS): LangChain ``read_skill`` adapter.

Envelope logic lives in ``agent.skills`` so MCP (Science Agent) and LangChain
share one implementation.
"""
import json
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    from agent.skills import build_read_skill_response
except ImportError:
    def build_read_skill_response(skill_id: str, relative_path: Optional[str] = None):
        return {
            "success": False,
            "error": "Skills middleware unavailable",
            "available_ids": [],
        }


class ReadSkillInput(BaseModel):
    skill_id: str = Field(
        ...,
        description="Public ID from the unified platform/scientific skill catalog, e.g. rdkit, nature_figure, scanpy, sas_biopython",
    )
    relative_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional path inside the skill package relative to its root. "
            "Omit or use SKILL.md for the main skill doc. "
            "For nature_* routers: manifest.yaml, static/core/contract.md, references/api.md, etc. "
            "Path traversal (..) is rejected."
        ),
    )


@tool("read_skill", args_schema=ReadSkillInput)
def read_skill_tool(skill_id: str, relative_path: Optional[str] = None) -> str:
    """
    Read a platform or scientific-agent-skills package. Default is SKILL.md.
    Use when the Computational Biologist or the plan asks you to follow a skill.
    skill_id must match Available skills. Upstream collisions use sas_<id>.
    For progressive disclosure (nature_figure / nature_writing / nature_polishing, or thick database skills),
    call again with relative_path to load manifest.yaml, static/, or references/ files.
    Returns JSON: success, skill_id, skill_root, relative_path, content, available_ids.
    """
    return json.dumps(
        build_read_skill_response(skill_id, relative_path),
        ensure_ascii=False,
    )


def get_python_repl_tool():
    """Compatibility import: host REPLs are retired; execution is session-bound."""
    return None


@tool("list_skills")
def list_skills_tool(query: str = "", limit: int = 10, offset: int = 0) -> str:
    """Search reference skills by name or task. Returns short descriptions and pagination; read_skill loads a selected guide."""
    from agent.skills import build_list_skills_response
    return json.dumps(build_list_skills_response(query, limit, offset), ensure_ascii=False)
