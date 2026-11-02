"""Persistent Science must not turn an empty PI plan into ordinary chat."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agent.graph.research.plan import research_plan_node


class Sections:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def state():
    return {
        "protein_context": SimpleNamespace(get_context_summary=lambda: "Public retrospective modelling task"),
        "messages": [
            HumanMessage(content="Research objective: reuse the accepted model and verify the remaining criterion.")
        ],
        "history": [
            {"role": "assistant", "kind": "thinking", "role_id": "principal_investigator", "content": "Planning"}
        ],
        "ui_lang": "en",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["[]", "No sections were returned", RuntimeError("upstream failure")])
async def test_explicit_scientific_workflow_preserves_expert_handoff(output):
    sections = Sections(output)
    result = await research_plan_node(
        state(),
        {
            "configurable": {
                "chains": {
                    "pi_sections": sections,
                    "scientific_workflow_required": True,
                }
            }
        },
    )
    assert sections.calls == 1
    assert result["status"] == "resume_plan"
    assert result["waiting_for"] is None
    assert result["research_sections"] == []
    assert result["history"][-1]["role_id"] == "principal_investigator"
    assert result["history"][-1].get("kind") != "thinking"
    assert "Research objective:" in result["pi_report"]


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, False, "true"])
async def test_unbound_conversation_keeps_ordinary_chat_fallback(flag):
    sections = Sections("[]")
    result = await research_plan_node(
        state(),
        {
            "configurable": {
                "chains": {
                    "pi_sections": sections,
                    "scientific_workflow_required": flag,
                }
            }
        },
    )
    assert sections.calls == 1
    assert result["status"] == "chat_mode"


@pytest.mark.asyncio
async def test_valid_pi_sections_still_enter_research():
    sections = Sections('[{"section_name":"Relevant evidence","search_queries":["NucB"]}]')
    result = await research_plan_node(
        state(),
        {
            "configurable": {
                "chains": {
                    "pi_sections": sections,
                    "scientific_workflow_required": True,
                }
            }
        },
    )
    assert sections.calls == 1
    assert result["status"] == "research_planning_done"
    assert result["research_sections"][0]["section_name"] == "Relevant evidence"
