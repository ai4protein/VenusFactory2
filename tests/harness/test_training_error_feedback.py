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
