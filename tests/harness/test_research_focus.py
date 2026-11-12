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



@pytest.mark.parametrize("deliverable", ["analysis", "report", "tool"])
def test_non_model_work_uses_real_capabilities_without_training_contract(deliverable):
    from agent.harness.research_focus import instructions

    # Residual model state must not override the current explicit deliverable.
    context = {
        "plan": {"deliverable": deliverable},
        "model_development": {
            "prepared_dataset": {"evaluation_file": "evaluation.json"},
            "next_action": {"stage": "evaluate_baseline"},
            "repair_source": {"source": "old_model.py"},
        },
    }
    text = instructions(context, 6)
    for model_contract in ("prepare_research_dataset", "train_research_model",
                           "fit(training, seed)", "predict(model, inputs)",
                           "model_development.next_action", "get_research_review_requirements"):
        assert model_contract not in text
    assert "category, then name and description" in text
    assert "inspect the selected interface" in text
    assert "actual completed results and output files" in text
    assert "host analysis_evidence" in text and "cite its path" in text
    assert "preserve all scientific acceptance criteria" in text
    assert "do not substitute demos or invented data" in text
    assert "6 execution calls" in text
    assert "When two calls remain, checkpoint immediately." in text


def test_reusable_tool_still_requires_tested_interface_and_private_review():
    from agent.harness.research_focus import instructions

    tool = instructions({"plan": {"deliverable": "tool"}}, 8)
    assert "input/output contract and boundary cases" in tool
    assert "independent tool Review before registration" in tool
    assert "operator selection and an author email" in tool
    for deliverable in ("analysis", "report"):
        text = instructions({"plan": {"deliverable": deliverable}}, 8)
        assert "before registration" not in text
        assert "independent tool Review" not in text


@pytest.mark.parametrize("deliverable", ["model", None, "", "unknown-future-kind"])
@pytest.mark.parametrize("prepared", [False, True])
def test_model_and_legacy_unspecified_work_keep_full_development_requirements(deliverable, prepared):
    from agent.harness.research_focus import instructions

    context = {"plan": {} if deliverable is None else {"deliverable": deliverable}}
    if prepared:
        context["model_development"] = {"prepared_dataset": {"evaluation_file": "evaluation.json"}}
    text = instructions(context, 12)
    assert "fit(training, seed) and predict(model, inputs)" in text
    assert "model_development.repair_source" in text
    assert "measure a child generation on the same split" in text
    assert "Keep failed trials as evidence" in text
    assert "Preserve observed targets and biological groups" in text
    assert "online ESM2-8M only" in text
    assert "submit independent Review" in text
    assert "public release requires operator selection and an author email" in text
    assert "12 execution calls" in text
    if prepared:
        assert "Preserve the fixed split and do not prepare it again" in text
        assert "training rows only" in text
        assert "use prepare_research_dataset for measured tabular data" not in text
    else:
        assert "use prepare_research_dataset for measured tabular data" in text
