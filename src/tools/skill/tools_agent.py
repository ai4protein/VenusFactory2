"""
Skill tools for CB/MLS: read_skill (get SKILL.md content), optional python_repl (execute code).
Used so CB can see skill metadata and instruct MLS to read and execute skills; code/plots visible in chat.
"""
import json
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

# Import from agent.skills (middleware)
try:
    from agent.skills import get_skill_content, list_skill_ids
except ImportError:
    def get_skill_content(skill_id: str) -> Optional[str]:
        return None
    def list_skill_ids():
        return []


class ReadSkillInput(BaseModel):
    skill_id: str = Field(..., description="Skill directory name, e.g. rdkit, fda, brenda_database")


@tool("read_skill", args_schema=ReadSkillInput)
def read_skill_tool(skill_id: str) -> str:
    """
    Read the full SKILL.md content for a skill. Use this when the Computational Biologist or the plan asks you to follow a skill (e.g. rdkit, fda, brenda_database). skill_id is the directory name under src/agent/skills/, e.g. rdkit, fda, brenda_database. Returns the full skill document so you can write and run code (agent_generated_code or python_repl) according to the skill's instructions. Output is returned as JSON with success, content, and available_ids.
    """
    available = list_skill_ids()
    if not skill_id or skill_id not in available:
        return json.dumps({
            "success": False,
            "error": f"Unknown skill_id. Available: {available}",
            "available_ids": available,
        }, ensure_ascii=False)
    content = get_skill_content(skill_id)
    if content is None:
        return json.dumps({
            "success": False,
            "error": f"Could not read skill: {skill_id}",
            "available_ids": available,
        }, ensure_ascii=False)
    return json.dumps({
        "success": True,
        "skill_id": skill_id,
        "content": content,
        "available_ids": available,
    }, ensure_ascii=False)


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
