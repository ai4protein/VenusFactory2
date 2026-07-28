"""
Skill tools for Science Expert (CB/MLS): LangChain ``read_skill`` adapter.

Envelope logic lives in ``agent.skills`` so MCP (Science Agent) and LangChain
share one implementation.
"""
import json
from typing import Optional

from langchain.tools import tool
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
        description="Skill directory name under src/agent/skills/, e.g. rdkit, fda, nature_figure, zero_shot_mutation_workflow",
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
    Read a file from a VenusFactory skill package. Default is SKILL.md.
    Use when the Computational Biologist or the plan asks you to follow a skill.
    skill_id is the directory name under src/agent/skills/ (must match Available skills skill_id).
    For progressive disclosure (nature_figure / nature_writing / nature_polishing, or thick database skills),
    call again with relative_path to load manifest.yaml, static/, or references/ files.
    Returns JSON: success, skill_id, skill_root, relative_path, content, available_ids.
    """
    return json.dumps(
        build_read_skill_response(skill_id, relative_path),
        ensure_ascii=False,
    )


# Optional: Python REPL for interactive code execution (stdout/stderr and plot paths visible in chat)
_python_repl_tool = None


def get_python_repl_tool():
    """Return PythonREPLTool if langchain_experimental is available, else None."""
    global _python_repl_tool
    if _python_repl_tool is not None:
        return _python_repl_tool
    try:
        from langchain_experimental.tools import PythonREPLTool
        _python_repl_tool = PythonREPLTool(
            name="python_repl",
            description="Execute Python code in a REPL. Use for quick scripts, plotting (e.g. matplotlib), or trying skill examples. Stdout, stderr, and any saved figure paths will be visible in the chat. Do not use for long-running or file-heavy tasks; prefer agent_generated_code for those.",
        )
    except ImportError:
        pass
    return _python_repl_tool
