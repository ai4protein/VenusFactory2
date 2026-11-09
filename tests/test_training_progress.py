"""Training log replay keeps curves, test metrics and snapshots consistent."""
import math

from backend.web_v2.training_progress import TrainingProgress, parse_training_line


def test_replay_handles_sparse_repeated_and_out_of_order_epochs():
    lines = [
        "2026-11-09 10:00:00 - Trainer - INFO - Epoch 2 Train Loss: 1e-1",
        "Epoch 2 Val Loss: 0.2",
        "Epoch 2 Val Spearman: 0.6",
        "Epoch 4 Train Loss: 0.07",
        "Epoch 4 Val Accuracy: 0.8",
        "Epoch 2 Val Spearman: 0.65",
        "Epoch 3 Val Spearman: 0.7",
        "Epoch 3 Train Loss: 0.08",
        "Epoch 4 Val Loss: 0.15",
        "Test Loss: 0.12",
        "Test Accuracy: 0.85",
    ]
    expected = {
        "epochs": [2, 4, 3],
        "train_loss": [0.1, 0.07, 0.08],
        "val_loss": [0.2, 0.15, None],
        "val_metrics": {"spearman": [0.65, None, 0.7], "accuracy": [None, 0.8, None]},
    }
    runs = []
    for _ in range(2):  # A reconnect rebuilds the view from the durable log.
        state = TrainingProgress()
        events = [event for line in lines for event in parse_training_line(line, state)]
        assert state.metrics() == expected
        assert state.test_metrics == {"loss": 0.12, "accuracy": 0.85}
        runs.append(events)
    assert runs[0] == runs[1]


def test_emitted_snapshots_do_not_change_with_later_lines():
    state = TrainingProgress()
    first = parse_training_line("Epoch 0 Val Accuracy: 0.7", state)[0]
    test = parse_training_line("Test Loss: 0.5", state)[0]
    parse_training_line("Epoch 1 Train Loss: 0.1", state)
    parse_training_line("Epoch 0 Val Accuracy: 0.9", state)
    parse_training_line("Test Accuracy: 0.8", state)
    assert first["data"] == {
        "epochs": [0], "train_loss": [None], "val_loss": [None],
        "val_metrics": {"accuracy": [0.7]},
    }
    assert test["data"] == {"metrics": {"loss": 0.5}}


def test_parser_retains_nonfinite_measurements_for_transport_to_handle():
    state = TrainingProgress()
    parse_training_line("Epoch 0 Train Loss: NaN", state)
    parse_training_line("Epoch 0 Val Loss: inf", state)
    parse_training_line("Test Pearson: -inf", state)
    assert math.isnan(state.train_loss[0])
    assert state.val_loss == [float("inf")]
    assert state.test_metrics == {"pearson": float("-inf")}


def test_other_log_lines_do_not_fabricate_progress():
    state = TrainingProgress()
    for line in ["", "Loading model", "Epoch 1 started", "Test failed", "Epoch x Train Loss: bad"]:
        assert parse_training_line(line, state) == []
    assert state.metrics() == {"epochs": [], "train_loss": [], "val_loss": [], "val_metrics": {}}
    assert state.test_metrics == {}
