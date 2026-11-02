import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/rsi/cases/run_alde_author_reproduction.py"
    spec = importlib.util.spec_from_file_location("alde_reproduction", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_trajectory_audit_preserves_but_reports_source_duplicates():
    check = module().check_indices
    audit = check(list(range(96)) + list(range(384)), 1000)
    assert audit["query_events"] == 480 and audit["unique_sequences"] == 384
    assert audit["later_repeated_queries"] == 96
    assert check(list(range(480)), 1000)["later_repeated_queries"] == 0
    for bad in [list(range(479)), list(range(479)) + [1000], list(range(479)) + [1.5]]:
        with pytest.raises(ValueError):
            check(bad, 1000)


def test_independent_curve_uses_all_observed_queries_and_rejects_bad_indices():
    import numpy as np

    path = Path(__file__).resolve().parents[2] / "scripts/rsi/cases/audit_alde_reproduction.py"
    spec = importlib.util.spec_from_file_location("alde_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    labels = np.arange(1, 501, dtype=float)
    assert np.allclose(audit.curve(list(range(480)), labels), np.array([96, 192, 288, 384, 480]) / 500)
    duplicate = list(range(96)) + list(range(384))
    assert audit.curve(duplicate, labels)[-1] == 384 / 500
    with pytest.raises(ValueError):
        audit.curve(list(range(479)) + [-1], labels)
