import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/rsi/cases/wait_clean_training_evaluation.py"
SPEC = importlib.util.spec_from_file_location("clean_evaluation_queue_fixture", PATH)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)
IDENTITY = {"pid": 123, "start_ticks": "456", "boot_id": "fixture"}


def completed():
    return {
        "process": IDENTITY,
        "status": "completed",
        "training_completed": True,
        "evaluation_completed": False,
        "stages": [
            {"algorithm": name, "epochs": epochs, "status": "completed", "exit_code": 0}
            for name, epochs in (("triplet", 2000), ("supconH", 1500))
        ],
    }


def test_completed_dependency_can_be_observed_after_process_exit():
    assert queue.classify_training(completed(), IDENTITY, False) == "ready"


def test_transient_observation_failure_is_not_terminal():
    assert queue.classify_training(None, IDENTITY, True) == "observation_unavailable"
    assert queue.classify_training(None, IDENTITY, False) == "predecessor_unavailable"
    assert queue.classify_training({"status": "running", "process": IDENTITY}, IDENTITY, True) == "waiting_for_training"
    assert (
        queue.classify_training({"status": "running", "process": IDENTITY}, IDENTITY, False)
        == "predecessor_unavailable"
    )


def test_explicit_failure_is_preserved_without_restart():
    with pytest.raises(RuntimeError, match="no automatic restart"):
        queue.classify_training({"status": "failed", "process": IDENTITY}, IDENTITY, True)


@pytest.mark.parametrize(
    "mutation",
    [
        "identity",
        "training_flag",
        "evaluation_flag",
        "stage_missing",
        "stage_order",
        "epoch_count",
        "exit_code",
        "bool_exit",
        "stage_failed",
        "unknown_state",
    ],
)
def test_invalid_dependency_cannot_start_evaluation(mutation):
    status = completed()
    if mutation == "identity":
        status["process"] = {**IDENTITY, "start_ticks": "other"}
    elif mutation == "training_flag":
        status["training_completed"] = False
    elif mutation == "evaluation_flag":
        status["evaluation_completed"] = True
    elif mutation == "stage_missing":
        status["stages"].pop()
    elif mutation == "stage_order":
        status["stages"].reverse()
    elif mutation == "epoch_count":
        status["stages"][0]["epochs"] = 20
    elif mutation == "exit_code":
        status["stages"][0]["exit_code"] = 1
    elif mutation == "bool_exit":
        status["stages"][0]["exit_code"] = False
    elif mutation == "stage_failed":
        status["stages"][0]["status"] = "failed"
    else:
        status["status"] = "unknown"
    with pytest.raises(ValueError):
        queue.classify_training(status, IDENTITY, True)
