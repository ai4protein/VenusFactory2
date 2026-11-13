"""One real isolation/state-portability regression; synthetic engineering data only."""

import hashlib
import json
from pathlib import Path

import pytest
from test_research_training import service_at

from agent.harness.capability_review import model_evaluation
from agent.harness.research_workspace import invoke_bundle, run_python

SOURCE = """from pathlib import Path

class CenteredLinearRegressor:
    def fit(self, rows):
        xs = [float(row["inputs"]["x"]) for row in rows]
        ys = [float(row["target"]) for row in rows]
        self.center = sum(xs) / len(xs)
        self.offset = sum(ys) / len(ys)
        self.slope = sum((x-self.center)*(y-self.offset) for x,y in zip(xs,ys)) / sum((x-self.center)**2 for x in xs)
        self.training_count = len(rows)
        return self

    def predict(self, inputs):
        if set(inputs) != {"x"} or type(inputs["x"]) not in (int, float):
            raise ValueError("Provide numeric x")
        return self.offset + self.slope * (inputs["x"] - self.center)

def fit(training, seed):
    assert training["task"] == "regression"
    assert not {"validation", "test"}.intersection(training)
    assert not Path("evaluation.json").exists()
    assert not Path("private-other-session.txt").exists()
    rows = training["train"]
    assert len(rows) == 8 and all(row["id"].startswith("train") for row in rows)
    return CenteredLinearRegressor().fit(rows)

def predict(model, inputs):
    return {"value": model.predict(inputs)}
"""


@pytest.mark.asyncio
async def test_helper_class_with_explicit_module_functions_fits_and_replays_in_real_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "sandboxes"))
    service = await service_at(tmp_path)
    executions = []

    async def observed_runner(workspace, control, script, timeout=120, **kwargs):
        result = await run_python(workspace, control, script, timeout, **kwargs)
        executions.append({"workspace": Path(workspace), "script": script, "result": result})
        assert result["backend"] == "bwrap" and result["exit_code"] == 0
        assert "gpu" not in result
        return result

    service.runner = observed_runner
    await service.write_research_file("helper_model.py", SOURCE)
    trained = await service.train_research_model(
        "Helper-class engineering fixture",
        "helper_model.py",
        "evaluation.json",
        "Verify explicit module entrypoints and portable fitted helper state; no biological claim.",
        variant="baseline",
        device="cpu",
    )
    training, experiment = trained["training"], trained["experiment"]
    receipt_paths = list(service.development.root.glob("training/*/receipt.json"))
    assert len(receipt_paths) == 1
    receipt = json.loads(receipt_paths[0].read_text())
    assert receipt["status"] == receipt["stage"] == "completed"
    assert receipt["experiment_id"] == experiment["id"]
    assert receipt["execution"]["backend"] == "bwrap" and receipt["execution"]["exit_code"] == 0
    fit_stats = json.loads(receipt["execution"]["stdout"].strip().splitlines()[-1])
    assert fit_stats["trained_rows"] == 8 and fit_stats["state_bytes"] > 0
    assert experiment["fresh_process_replay"] and not experiment["test_used_for_feedback"]
    assert experiment["validation"]["score"] == 0
    assert not experiment["validation"]["diagnostics"]["constant_predictions"]

    state_file = next(name for name in training["fitted_files"] if name.endswith("model-state.json"))
    state_path = service.workspace / state_file
    state = json.loads(state_path.read_text())
    assert state["format"] == "pickle-v5"  # The fitted custom class, not a manually copied parameter dict.
    weights = state_path.parent / state["file"]
    assert weights.stat().st_size == state["bytes"]
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == state["sha256"]

    manifest = {"source": training["source"], "files": experiment["files"]}
    start = len(executions)
    for index in range(2):
        result = await invoke_bundle(
            service.workspace, manifest, {"x": 20}, tmp_path / f"fresh-prediction-{index}", runner=observed_runner
        )
        assert result == {"value": 41.0}
    prediction_runs = executions[start:]
    assert len(prediction_runs) == 2
    assert len({row["result"]["run_id"] for row in prediction_runs}) == 2
    assert len({row["workspace"] for row in prediction_runs}) == 2
    for execution in prediction_runs:
        assert not list(execution["workspace"].rglob("training.json"))
        assert not list(execution["workspace"].rglob("evaluation.json"))

    spec = {
        "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
        "output_schema": {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]},
        "evaluation_file": training["evaluation_file"],
        "prediction_key": "value",
        "reproduction_script": training["reproduction_script"],
        "fitted_files": training["fitted_files"],
    }
    review_manifest = {
        **manifest,
        "review_spec": spec,
        "evidence_files": dict.fromkeys(training["evidence_files"], "private"),
    }
    technical = await model_evaluation(
        service.workspace, review_manifest, tmp_path / "independent-refit", observed_runner
    )
    assert technical["training_reproduced"] and technical["score"] == 0
    assert technical["diagnostics"]["split"] == "test"
    assert technical["split_counts"] == {"train": 8, "validation": 3, "test": 3}
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == state["sha256"]
    assert (service.workspace / "helper_model.py").read_text() == SOURCE
    assert not service.proposals  # This engineering test does not perform scientific admission or publication.
