import json
import pytest

from agent.harness.tool_diagnostics import python_failure
from agent.harness.research_workspace import ResearchWorkspace


def test_last_exception_survives_warnings_and_chained_trace(tmp_path):
    text = ("UserWarning: serial execution fallback\nTraceback:\n"
            "TypeError: old cause\nDuring handling of the above exception:\n"
            "Traceback:\nValueError: incompatible feature matrix\nExpected one row per observation")
    message = python_failure(text, tmp_path)
    assert message == "ValueError: incompatible feature matrix\nExpected one row per observation"
    assert "UserWarning" not in message and "old cause" not in message


def test_non_python_failure_still_has_bounded_diagnostics(tmp_path):
    text = "process stopped\n" + "x" * 3000 + "\nExit code: 137"
    message = python_failure(text, tmp_path, 400)
    assert len(message) <= 400 and message.endswith("Exit code: 137")
    assert "process stopped" in message


@pytest.mark.asyncio
async def test_real_failed_fit_returns_contract_and_retains_full_receipt(tmp_path):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []},
                                development_root=tmp_path / "development")
    table = "id,group,x,target\n" + "".join(f"{i},g{i},{i},{2*i+1}\n" for i in range(60))
    await service.write_research_file("data.csv", table)
    prepared = await service.prepare_research_dataset(
        "data.csv", "target", ["x"], "group", "Synthetic software fixture", "Independent fixture groups")
    code = """import warnings
warnings.warn("benign execution warning")
def fit(training, seed):
    import pandas as pd
    return pd.DataFrame(training)
def predict(model, inputs):
    return {"value": 0.0}
"""
    await service.write_research_file("broken.py", code)
    with pytest.raises(ValueError) as caught:
        await service.train_research_model(
            "Fixture", "broken.py", prepared["evaluation_file"], "Validate feedback only")
    message = str(caught.value)
    assert "All arrays must be of the same length" in message
    assert "training['train']" in message
    assert "At broken.py:5:" in message and "return pd.DataFrame(training)" in message
    assert "benign execution warning" not in message and "Traceback" not in message
    assert len(message) < 850
    receipts = list(service.development.root.glob("training/*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "failed"
    execution = receipt["execution"]
    assert "Traceback" in (execution.get("stderr") or execution.get("stderr_tail", ""))
    assert not service.development.records()


def test_class_methods_report_missing_module_entrypoints_without_running_code():
    from agent.harness.research_training import validate_algorithm
    source = """class Predictor:
    def fit(self, training, seed):
        return self
    def predict(self, inputs):
        return 1.0
"""
    with pytest.raises(ValueError, match="module-level functions, not class methods"):
        validate_algorithm(source)



def test_training_trace_maps_original_line_and_ignores_external_or_wrapper_frames(tmp_path):
    source = "import numpy as np\ndef fit(training, seed):\n    return np.mean(training['train'])\n"
    trace = (
        'Traceback (most recent call last):'
        '\n  File "/workspace/runtime.py", line 3, in fit\n'
        "    FAKE LINE FROM STDERR\n"
        '  File "/opt/venus-runtime/lib/numpy/core.py", line 4, in mean\n'
        "TypeError: incompatible input\n"
        "The above exception was the direct cause of the following exception:\n"
        '  File "train.py", line 12, in <module>\n'
        "ValueError: TypeError: incompatible input. Read training['train'].\n"
    )
    message = python_failure(trace, tmp_path, source_file="src/model.py", source_text=source)
    assert message.startswith("ValueError: TypeError: incompatible input")
    assert "At src/model.py:3: return np.mean(training['train'])" in message
    assert "FAKE LINE" not in message and "/opt/" not in message and "train.py" not in message


@pytest.mark.parametrize("trace_file,line", [
    ("/other-user/runtime.py", 2), ("/workspace/runtime.py", 100), ("train.py", 2)
])
def test_unrelated_trace_frames_are_not_attributed_to_authored_source(tmp_path, trace_file, line):
    source = "def fit(training, seed):\n    return helper(training)\n"
    trace = f'  File "{trace_file}", line {line}, in fit\nValueError: failure'
    assert python_failure(trace, tmp_path, source_file="src/model.py", source_text=source) == "ValueError: failure"


def test_line_feedback_redacts_credentials_and_respects_total_budget(tmp_path):
    secret = "sk-" + "z"*40
    source = f"def fit(training, seed):\n    return fit_remote(training, api_key='{secret}')\n"
    trace = '  File "/workspace/runtime.py", line 2, in fit\nValueError: ' + "x"*1000
    message = python_failure(trace, tmp_path, 400, source_file="src/model.py", source_text=source)
    assert len(message) <= 400 and secret not in message
    assert "At src/model.py:2:" in message
