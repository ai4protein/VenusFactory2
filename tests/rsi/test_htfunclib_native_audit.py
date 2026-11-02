import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[2] / "scripts/rsi/cases/audit_htfunclib_case.py"
spec = importlib.util.spec_from_file_location("htfunclib_native_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Booster:
    def __init__(self, values, iterations=10000):
        self.values, self.iterations = values, iterations

    def current_iteration(self):
        return self.iterations

    def predict(self, features, num_threads):
        assert num_threads == 4
        return self.values


def test_native_predictions_replay_binary_and_multiclass():
    assert audit.native_predictions(Booster([0.1, 0.9]), [0, 1], [False, True], [False, True]).tolist() == [False, True]
    values = [[0.1, 0.2, 0.3, 0.4], [0.8, 0.1, 0.05, 0.05]]
    assert audit.native_predictions(Booster(values), [0, 1], ["C", "G", "G/C", "U"], ["U", "C"]).tolist() == ["U", "C"]


@pytest.mark.parametrize("values", [[0.1], [np.nan, 0.9], [-0.1, 0.9], [[0.1, 0.9], [0.2, 0.8]]])
def test_native_predictions_reject_bad_binary_output(values):
    with pytest.raises(ValueError):
        audit.native_predictions(Booster(values), [0, 1], [False, True], [False, True])


def test_truncated_model_or_changed_predictions_fail():
    with pytest.raises(ValueError, match="boosting schedule"):
        audit.native_predictions(Booster([0.1, 0.9], 9999), [0, 1], [False, True], [False, True])
    with pytest.raises(ValueError, match="disagree"):
        audit.native_predictions(Booster([0.1, 0.9]), [0, 1], [False, True], [True, True])


def test_multiclass_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="probability matrix"):
        audit.native_predictions(Booster([[0.2, 0.2, 0.2]]), [0], ["a", "b", "c"], ["a"])
