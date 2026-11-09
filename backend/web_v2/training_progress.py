"""Reconstruct training metrics from durable logs without I/O or job dependencies."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)"
_LOG_PREFIX = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - [A-Za-z]+ - INFO - (.*)")
_EPOCH_METRIC = re.compile(
    rf"Epoch (\d+) (Train Loss|Val Loss|Val [A-Za-z0-9_\-]+): {_NUMBER}", re.IGNORECASE
)
_TEST_METRIC = re.compile(rf"Test ([A-Za-z0-9_\-]+):\s*{_NUMBER}", re.IGNORECASE)


@dataclass
class TrainingProgress:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float | None] = field(default_factory=list)
    val_loss: list[float | None] = field(default_factory=list)
    val_metrics: dict[str, list[float | None]] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    def metrics(self) -> dict:
        """Return a snapshot independent of subsequent log updates."""
        return {
            "epochs": list(self.epochs),
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "val_metrics": {name: list(values) for name, values in self.val_metrics.items()},
        }

    def epoch_slot(self, epoch: int) -> int:
        if epoch not in self.epochs:
            self.epochs.append(epoch)
            self.train_loss.append(None)
            self.val_loss.append(None)
            for values in self.val_metrics.values():
                values.append(None)
        return self.epochs.index(epoch)


def parse_training_line(line: str, state: TrainingProgress) -> list[dict]:
    prefix = _LOG_PREFIX.search(line)
    content = prefix.group(1).strip() if prefix else line.strip()
    match = _EPOCH_METRIC.search(content)
    if match:
        epoch, name, value = int(match[1]), match[2].lower(), float(match[3])
        index = state.epoch_slot(epoch)
        if name == "train loss":
            state.train_loss[index] = value
        elif name == "val loss":
            state.val_loss[index] = value
        else:
            metric = state.val_metrics.setdefault(name.removeprefix("val "), [None] * len(state.epochs))
            metric[index] = value
        return [{"event": "training_metrics", "data": state.metrics()}]

    match = _TEST_METRIC.search(content)
    if match:
        state.test_metrics[match[1].lower()] = float(match[2])
        return [{"event": "test_results", "data": {"metrics": dict(state.test_metrics)}}]
    return []
