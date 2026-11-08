import copy
import pytest
from agent.harness.research_execution_feedback import feedback_state

NOTE = ("Execution budget: 4 tool calls remain; 15 read-only discovery calls remain. "
        "Save a checkpoint when the current step is ready. "
        "Checkpoint saves have a separate allowance: 2 remain.")

def call(tool,args,*,failed,output):
    return [{"kind":"tool_call_start","tool_call_id":"test",
             "tool_name":"mcp__venusfactory__"+tool,"args":args},
            {"kind":"tool_result","tool_call_id":"test","is_error":failed,"output":output}]

@pytest.mark.parametrize("tool,source_field", [
    ("train_research_model","source_file"),("evaluate_research_model","source")])
@pytest.mark.parametrize("parent,explicit", [("","baseline"),("measured-parent","candidate")])
@pytest.mark.parametrize("initial", [None,"auto"])
def test_equivalent_variant_retry_clears_the_same_work_target(tmp_path,tool,source_field,parent,explicit,initial):
    args={source_field:"src/model.py","parent_experiment_id":parent}
    if initial is not None:args["variant"]=initial
    first=feedback_state(call(tool,args,failed=True,output="Bad fitted state"),tmp_path)
    original=copy.deepcopy(first)
    repaired={source_field:"src/model.py","parent_experiment_id":parent,"variant":explicit}
    result=feedback_state(call(tool,repaired,failed=False,output="Verified experiment"),tmp_path,first)
    assert result["pending"]==[] and result["errors"]==[]
    assert first==original

@pytest.mark.parametrize("changed", [
    {"source_file":"src/other.py","variant":"baseline"},
    {"source_file":"src/model.py","variant":"ablation"},
    {"source_file":"src/model.py","parent_experiment_id":"different","variant":"candidate"}])
def test_another_experiment_does_not_clear_an_unresolved_failure(tmp_path,changed):
    first=feedback_state(call("train_research_model",{"source_file":"src/model.py"},
                              failed=True,output="Fitting failed"),tmp_path)
    result=feedback_state(call("train_research_model",changed,failed=False,output="Succeeded elsewhere"),tmp_path,first)
    assert result["errors"]==first["errors"]

def test_trace_root_cause_survives_budget_text_and_context_truncation(tmp_path):
    detail="ValueError: Training failed: "+("trace line"+chr(10))*200+"TypeError: list indices must be integers"
    state=feedback_state(call("train_research_model",{"source_file":"src/model.py"},
                              failed=True,output=detail+chr(10)+NOTE),tmp_path)
    message=state["errors"][0]["error"]
    assert len(message)<=1200
    assert message.endswith("TypeError: list indices must be integers")
    assert "Execution budget:" not in message

def test_prior_feedback_is_cleaned_without_mutating_historical_record(tmp_path):
    previous={"pending":[{"key":"prior","tool":"train_research_model",
                           "error":"KeyError: inputs"+chr(10)+NOTE}]}
    snapshot=copy.deepcopy(previous)
    result=feedback_state([],tmp_path,previous)
    assert result["errors"][0]["error"]=="KeyError: inputs"
    assert previous==snapshot

def test_actual_budget_denial_remains_an_error(tmp_path):
    detail="DisclosureError: Execution budget reached."
    result=feedback_state(call("write_session_file",{},failed=True,output=detail+chr(10)+NOTE),tmp_path)
    assert result["errors"][0]["error"]==detail


@pytest.mark.parametrize("broker", [False, True])
def test_rewritten_file_resolves_old_write_failure_without_clearing_other_work(tmp_path, broker):
    def write(path, content, failed):
        name, args = "write_session_file", {"path": path, "content": content}
        if broker:
            name, args = "call_tool", {"name": name, "arguments": args}
        return call(name, args, failed=failed, output="Execution budget reached" if failed else "Saved")
    previous = feedback_state(write("src/model.py", "old candidate", True), tmp_path)
    snapshot = copy.deepcopy(previous)
    unrelated = feedback_state(write("README.md", "notes", False), tmp_path, previous)
    assert unrelated["errors"] == previous["errors"]
    still_failed = feedback_state(write("src/model.py", "corrected candidate", True), tmp_path, previous)
    assert len(still_failed["pending"]) == 1
    repaired = feedback_state(write("src/model.py", "corrected candidate", False), tmp_path, previous)
    assert repaired == {"pending": [], "errors": []}
    assert previous == snapshot
    assert "old candidate" not in str(previous)


def test_successful_checkpoint_resolves_rejected_save_but_not_failed_training(tmp_path):
    events = call("submit_research_checkpoint", {
        "summary": "first attempt", "completed_steps": ["unknown"], "artifacts": ["report.md"]
    }, failed=True, output="Unknown plan step")
    previous = feedback_state(events, tmp_path)
    previous = feedback_state(call("train_research_model", {"source_file": "src/model.py"},
        failed=True, output="Fitting failed"), tmp_path, previous)
    snapshot = copy.deepcopy(previous)
    result = feedback_state(call("submit_research_checkpoint", {
        "summary": "Partial work", "completed_steps": [], "artifacts": ["report.md"]
    }, failed=False, output="Checkpoint saved"), tmp_path, previous)
    assert len(result["pending"]) == 1
    assert result["pending"][0]["tool"].endswith("train_research_model")
    assert result["errors"][0]["error"] == "Fitting failed"
    assert previous == snapshot
