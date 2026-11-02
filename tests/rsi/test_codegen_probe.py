"""Code-delivery probes must execute independently and reject unusable outputs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import run_codegen_probe as probe

SOURCE = """
import math

def fit(sequences, targets, seed):
    x = [s.count("C") / len(s) for s in sequences]
    means = [sum(v for v, y in zip(x, targets) if y == k) / targets.count(k) for k in [0, 1]]
    return {"means": means}

def predict(model, sequences):
    rows = []
    for s in sequences:
        x = s.count("C") / len(s)
        values = [math.exp(-10 * (x - c) ** 2) for c in model["means"]]
        rows.append([v / sum(values) for v in values])
    return rows
"""


def test_code_is_fitted_twice_and_reused_in_fresh_sandbox_process():
    result = probe.validate_source(SOURCE)
    assert result["same_seed_state_equal"]
    assert result["fresh_process_predictions_valid"]
    assert result["scientific_generalization_test"] is False


@pytest.mark.parametrize(
    "prediction",
    [
        [[0.5, 0.5]] * 4,
        [[0.1, 1.0], [0.9, 0.9], [0.3, 0.7], [0.5, 0.5]],
        [[0.0, 1.0], [1.0, 0.0]],
    ],
)
def test_invalid_probabilities_or_constant_outputs_fail_host_assessment(monkeypatch, prediction):
    monkeypatch.setattr(
        probe,
        "run_code",
        lambda source, request: {"model": {}} if request["operation"] == "fit" else {"predictions": prediction},
    )
    with pytest.raises(ValueError, match="Invalid or constant"):
        probe.validate_source(SOURCE)


@pytest.mark.asyncio
async def test_native_episode_contract_and_training_rows_are_admitted(tmp_path):
    from agent.rsi.generated_classifier import validate_rows
    from agent.rsi.scientific_episode import episode_contract
    from agent.rsi.tool_contracts import validate_generated_result

    task = probe.prepare_task(tmp_path / "fixture")
    rows, scope = probe.synthetic_case()
    validate_rows(rows, scope, labels=True)
    contract = episode_contract(task.capability, task.phase, task.tools)
    assert contract.context_tool == "inspect_model_code_task"
    assert contract.artifact_file == "candidate.json"
    import sys

    from fastmcp import Client

    config = {
        "mcpServers": {
            "probe": {
                "command": sys.executable,
                "args": ["-u", str(task.capability / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(task.workspace)},
            }
        }
    }
    async with Client(config) as client:
        metadata = (await client.call_tool("inspect_model_code_task", {})).data
    assert metadata["development_state"]["attempts_used"] == 0
    validate_generated_result(
        "inspect_model_code_task",
        metadata,
        workspace=task.workspace,
        capability=task.capability,
    )

    missing_catalog = dict(metadata)
    missing_catalog.pop("residue_reference_catalog")
    with pytest.raises(ValueError, match="bound artifact"):
        validate_generated_result(
            "inspect_model_code_task", missing_catalog, workspace=task.workspace, capability=task.capability
        )
