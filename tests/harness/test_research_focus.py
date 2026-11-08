import pytest
from types import SimpleNamespace
from agent.harness.research_focus import selected_tools, admitted_tools
from agent.harness.tool_disclosure import ToolDisclosure
from agent.harness.web_native_worker import web_tool_name

@pytest.mark.parametrize("harness",["kimi-code","codex","deepseek-harness","claude-code"])
def test_only_current_research_stage_is_exposed(harness):
    names=["submit_research_checkpoint","prepare_research_dataset","train_research_model","propose_research_capability"]
    tools=[SimpleNamespace(name=n,description=n) for n in names]
    context={"plan":{"deliverable":"model"},"model_development":{"next_action":{"stage":"evaluate_baseline"}}}
    focus=selected_tools(context,tools)
    assert focus==["prepare_research_dataset","train_research_model"]
    mapping={web_tool_name(t.name,harness):t for t in tools}
    direct=admitted_tools({"research_focus":focus},mapping)
    disclosure=ToolDisclosure(mapping,schema=lambda t:{},allowed=lambda t:True,direct=direct)
    assert direct<disclosure.names and "propose_research_capability" not in disclosure.names
    blocked=ToolDisclosure(mapping,schema=lambda t:{},allowed=lambda t:t.name!="train_research_model",direct=direct)
    assert "train_research_model" not in {t["name"] for t in blocked.surface()}

def test_nonresearch_calls_do_not_gain_direct_tools():
    assert admitted_tools({"research_focus":["train_research_model"]},{"train_research_model":object()})==set()
    with pytest.raises(ValueError):admitted_tools({"research_focus":["run_session_shell"]},{})
