from pathlib import Path

from pydantic import BaseModel, ValidationError

from agent.harness.tool_diagnostics import contract_error


def test_long_trace_preserves_final_cause_without_exposing_host_or_credentials(tmp_path):
    secret = "sk-" + "x"*40
    trace = "Capability execution failed:\n" + ("  File /home/service/private/engine.py, line 3\n"*80)
    trace += "FileNotFoundError: /home/service/private/weights.bin api_key=" + secret
    message = contract_error(ValueError(trace), tmp_path)
    assert len(message) <= 1620
    assert "Capability execution failed" in message and "FileNotFoundError" in message
    assert "trace shortened" in message
    assert "/home/service" not in message and secret not in message


def test_validation_error_does_not_echo_large_source_in_arguments(tmp_path):
    class WriteFile(BaseModel):
        path: str
        content: str
    code = "private source\n"*20000
    try:
        WriteFile.model_validate({"content":code})
    except ValidationError as exc:
        message = contract_error(exc, tmp_path)
    assert "path: Field required" in message
    assert "private source" not in message
    assert len(message) < 150


def test_owned_missing_file_is_virtualized_and_actionable(tmp_path):
    error = FileNotFoundError(2, "No such file", str(tmp_path/"weights.json"))
    message = contract_error(error, tmp_path)
    assert "/workspace/weights.json" in message
    assert str(tmp_path) not in message

def test_next_iteration_feedback_keeps_the_root_cause(tmp_path):
    from agent.harness.autonomous_research import execution_feedback
    events = [
        {"kind":"tool_call_start", "tool_call_id":"one",
         "tool_name":"evaluate_research_model", "args":{}},
        {"kind":"tool_result", "tool_call_id":"one", "is_error":True,
         "output":"Capability execution failed: " + "stack frame\n"*400 +
                  "AttributeError: cannot load the fitted feature transformer"},
    ]
    feedback = execution_feedback(events, tmp_path)
    assert len(feedback[0]["error"]) <= 1200
    assert "AttributeError: cannot load the fitted feature transformer" in feedback[0]["error"]
