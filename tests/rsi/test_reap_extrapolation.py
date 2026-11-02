"""Ensure methods share identical future-held-out extrapolation allocations."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("reap_extrapolation", Path(__file__).resolve().parents[2] / "scripts/rsi/cases/run_reap_extrapolation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return [{"family": "family", "sequence": f"sequence_{i}", "mutation_order": 1 if i < 20 else 2} for i in range(25)]


def test_every_method_shares_single_mutant_split_and_higher_order_future():
    plan = module.allocations(rows())
    assert len(plan) == 50
    assert plan == module.allocations(rows())
    for offset in range(0, 50, 5):
        group = plan[offset:offset+5]
        for arm in group:
            assert arm["train"] == group[0]["train"]
            assert arm["validation"] == group[0]["validation"]
            assert arm["future"] == list(range(20, 25))
            assert set(arm["train"]) | set(arm["validation"]) == set(range(20))
            assert not set(arm["train"]) & set(arm["validation"])


def test_full_sequence_overlap_rejected_even_with_different_mutation_names():
    data = rows()
    data[20]["sequence"] = data[0]["sequence"]
    with pytest.raises(ValueError, match="overlap"):
        module.allocations(data)


def test_missing_extrapolation_cohort_rejected():
    with pytest.raises(ValueError, match="Empty split"):
        module.allocations(rows()[:20])
