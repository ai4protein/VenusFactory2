"""Dual-mode skill wiring: Science Expert (graph) + Science Agent (kimi/MCP)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_expert_hub_has_read_skill():
    from tools.tools_agent_hub import get_tools

    names = {getattr(t, "name", None) for t in get_tools()}
    assert "read_skill" in names


def test_mcp_skill_tools_match_expert_loader():
    from tools.skill.tools_mcp import mcp_list_skills, mcp_read_skill
    from agent.skills import list_skill_ids, get_skill_content

    listed = json.loads(mcp_list_skills())
    assert listed["success"] is True
    assert "nature_figure" in listed["available_ids"]
    assert set(listed["available_ids"]) == set(list_skill_ids())

    payload = json.loads(mcp_read_skill("nature_figure"))
    assert payload["success"] is True
    assert payload["content"] == get_skill_content("nature_figure")

    progressive = json.loads(
        mcp_read_skill("nature_writing", relative_path="_shared_nature/core/ethics.md")
    )
    assert progressive["success"] is True
    assert "ethic" in progressive["content"].lower() or len(progressive["content"]) > 20


def test_read_skill_envelope_identical_across_adapters():
    """LangChain and MCP read_skill must share the same core envelope fields."""
    from tools.skill.tools_agent import read_skill_tool
    from tools.skill.tools_mcp import mcp_list_skills, mcp_read_skill
    from agent.skills import build_list_skills_response, build_read_skill_response

    skill_id = "nature_figure"
    core = build_read_skill_response(skill_id)
    expert = json.loads(read_skill_tool.invoke({"skill_id": skill_id}))
    agent = json.loads(mcp_read_skill(skill_id))

    assert expert == core
    assert agent == core
    for key in ("success", "skill_id", "skill_root", "relative_path", "content", "available_ids"):
        assert key in expert

    unknown_core = build_read_skill_response("not_a_real_skill_xyz")
    unknown_expert = json.loads(read_skill_tool.invoke({"skill_id": "not_a_real_skill_xyz"}))
    unknown_agent = json.loads(mcp_read_skill("not_a_real_skill_xyz"))
    assert unknown_expert == unknown_core == unknown_agent
    assert unknown_core["success"] is False
    assert "available_ids" in unknown_core

    listed = json.loads(mcp_list_skills())
    assert listed == build_list_skills_response()


def test_resolve_engine_routes_chat_mode():
    from web_v2.chat_api._shared import _resolve_chat_mode, _resolve_engine

    assert _resolve_engine(None, None, "science_agent") == "kimi-code"
    assert _resolve_engine(None, None, "science_expert") == "graph"
    assert _resolve_engine("graph", None, "science_agent") == "kimi-code"
    assert _resolve_engine("kimi-code", None, "science_expert") == "graph"
    assert _resolve_engine(None, None, None) == "graph"
    assert _resolve_engine("kimi-code", None, None) == "kimi-code"

    assert _resolve_chat_mode("kimi-code", None) == "science_agent"
    assert _resolve_chat_mode("graph", None) == "science_expert"
    assert _resolve_chat_mode("graph", "science_agent") == "science_agent"


def test_kimi_project_skill_symlinks():
    from agent.kimi_skills import ensure_kimi_project_skills

    dest = ensure_kimi_project_skills(ROOT)
    assert dest.is_dir()
    nature = dest / "nature_figure"
    assert nature.is_symlink() or nature.is_dir()
    assert (nature / "SKILL.md").is_file()
    assert (dest / "zero_shot_mutation_workflow" / "SKILL.md").is_file()


def test_kimi_system_prompt_embeds_catalog_and_self_directed_policy():
    from web_v2.chat_api._stream_kimi import _build_system_prompt

    prompt = _build_system_prompt("en")
    assert "mcp__venusfactory__read_skill" in prompt
    assert "mcp__venusfactory__list_skills" in prompt
    assert "you decide" in prompt.lower() or "Self-directed" in prompt
    assert "no** mandatory skill-first" in prompt.lower() or "no mandatory skill-first" in prompt.lower()
    # Catalog is inlined so Agent can see skills without an extra tool call
    assert "nature_figure" in prompt
    assert "zero_shot_mutation_workflow" in prompt
    assert "skill_id:" in prompt


def test_online_skill_builtin_still_denied():
    from agent.kimi_security import decide

    d = decide(
        {"tool_name": "Skill", "tool_input_display": {}},
        session_dir="/tmp/does-not-matter",
        mode="online",
    )
    assert d.allowed is False


def test_online_mcp_read_skill_trusted_when_allowlisted():
    from agent.kimi_security import decide, install_trusted_mcp_tools

    install_trusted_mcp_tools(["mcp__venusfactory__read_skill", "mcp__venusfactory__list_skills"])
    d = decide(
        {"tool_name": "mcp__venusfactory__read_skill", "tool_input_display": {"skill_id": "nature_figure"}},
        session_dir="/tmp/does-not-matter",
        mode="online",
    )
    assert d.allowed is True


def test_mcp_server_mounts_skill_mcp():
    import mcp_server

    assert hasattr(mcp_server, "skill_mcp")
    assert mcp_server.skill_mcp is not None
