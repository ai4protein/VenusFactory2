"""Numerical and strict reference-contract checks for McbA reproduction."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge


def script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / "scripts/rsi" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = script("run_mcba_multisubstrate")
auditor = script("audit_mcba_multisubstrate")


@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
def test_independent_svd_with_ill_scaled_augmentation(alpha):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(35, 76))
    x[:, 0] *= 1e4
    y = rng.normal(size=35) + 7
    test = rng.normal(size=(120, 76))
    test[:, 0] *= 1e4
    coef, intercept = auditor.ridge_svd(x, y, alpha)
    reference = Ridge(alpha=alpha).fit(x, y).predict(test)
    np.testing.assert_allclose(test @ coef + intercept, reference, rtol=1e-7, atol=1e-5)


def cell(text):
    return {"outputs": [{"data": {"text/plain": [text]}}]}


def test_missing_or_duplicate_cached_rank_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        runner.ranking_reference(cell("0 AAAA 1.2"), 2)
    with pytest.raises(ValueError, match="Incomplete"):
        runner.ranking_reference(cell("0 AAAA 1.2\n1 AAAA 1.1"), 2)


def test_truncated_architecture_cache_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        runner.metric_reference(cell("0 Simple: One Hot 0.1 0.2\n0 Ridge(alpha=1.0)"))


def test_signed_author_dcg_is_not_nonnegative_relevance_ndcg():
    y = np.array([-1.0, 0.0, 1.0])
    assert auditor.signed_ndcg(y, y) == pytest.approx(1.0)
    assert auditor.signed_ndcg(y, -y) == pytest.approx(-1.0)


def test_numerical_discrepancy_is_retained_at_original_tolerance():
    reference = np.array([0.0, 100.0, -1.0])
    reconstructed = reference + np.array([1.1e-5, 1.5e-5, 2e-5])
    result = auditor.prediction_comparison(reconstructed, reference, ["AAAA", "CCCC", "DDDD"])
    assert result["passed"] is False
    assert result["outside_tolerance"] == 2
    assert [x["id"] for x in result["examples"]] == ["AAAA", "DDDD"]
    assert result["max_absolute_error"] == pytest.approx(2e-5)
    assert result["max_tolerance_ratio"] > 1


@pytest.mark.parametrize("prediction", [np.array([np.nan]), np.array([np.inf]), np.zeros(2)])
def test_invalid_prediction_evidence_still_aborts(prediction):
    with pytest.raises(ValueError):
        auditor.prediction_comparison(prediction, np.zeros(1), ["AAAA"])


def test_discrepancies_and_unsuccessful_allocations_cannot_be_called_passed():
    successful = {
        "id": "rank",
        "status": "passed",
        "numerical_prediction_replay_passed": True,
        "historical_reference_passed": True,
    }
    numerical = {
        "id": "comparison",
        "status": "numerical_or_reference_discrepancy",
        "numerical_prediction_replay_passed": False,
        "historical_reference_passed": None,
    }
    result = auditor.summarize([successful, numerical])
    assert result["status"] == "verified_with_discrepancies"
    assert not result["independent_prediction_replay_passed"]
    assert result["historical_reference_counts"] == {"passed": 1, "failed": 0, "unavailable": 1}
    failed = {"id": "failed", "status": "runtime_error", "scientific_result": False}
    result = auditor.summarize([successful, failed])
    assert result["status"] == "incomplete_reproduction"
    assert result["unsuccessful_tasks"] == ["failed"]
    assert not result["independent_prediction_replay_passed"]
