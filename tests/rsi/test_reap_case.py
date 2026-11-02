"""Reject ambiguity and leakage at the REAP public-data boundary."""
import math
from copy import deepcopy

import pytest

from agent.rsi.cases.reap_case import (
    apply_mutations,
    join_campaign,
    mutation_key,
    summarize_replicates,
)


def fixture():
    sequence = {"family": "P450_BM3_FL62", "round": 0, "role": "campaign", "sequence_consistent": True, "key": "A1C", "sequence": "CDEF", "sequence_sha256": "hash", "source": {"sheet": "Zero-shot"}}
    plot = {"family": "P450_BM3_FL62", "round": 0, "key": "A1C", "name": "A1C", "mean": 2., "control": False, "source": {"sheet": "Fig.3a"}}
    assay = {"id": "a1", "family": "P450_BM3_FL62", "key": "A1C", "mean": 2., "replicates": [1., 2., 3.], "sample_sd": 1., "source": {"sheet": "raw"}}
    return [sequence], [plot], [assay]


def test_complete_unique_triplet_is_admitted_with_context():
    rows, excluded, controls, remaining = join_campaign(*fixture())
    assert len(rows) == 1 and not excluded and not controls and not remaining
    assert rows[0]["replicates"] == [1., 2., 3.]
    assert rows[0]["assay_context"] == "purified_initial_selection"


def test_equal_means_from_distinct_triplets_remain_ambiguous():
    seq, plots, assays = fixture()
    assays.append({**assays[0], "id": "a2", "replicates": [0., 2., 4.], "sample_sd": 2.})
    rows, excluded, _, remaining = join_campaign(seq, plots, assays)
    assert not rows and len(excluded) == 1 and len(remaining) == 2


@pytest.mark.parametrize("change", [{"role": "independent_saturation"}, {"round": 1}, {"sequence_consistent": False}])
def test_independent_future_or_wrong_sequence_cannot_supply_training(change):
    seq, plots, assays = fixture()
    seq[0].update(change)
    rows, excluded, _, _ = join_campaign(seq, plots, assays)
    assert not rows and len(excluded) == 1


def test_plot_and_raw_label_disagreement_is_not_repaired_by_row_order():
    seq, plots, assays = fixture()
    assays[0]["key"] = "A1D"
    rows, excluded, _, remaining = join_campaign(seq, plots, assays)
    assert not rows and len(excluded) == 1 and len(remaining) == 1


def test_shared_assay_cannot_be_used_twice():
    seq, plots, assays = fixture()
    plots.append(deepcopy(plots[0]))
    rows, excluded, _, _ = join_campaign(seq, plots, assays)
    assert len(rows) == 1 and len(excluded) == 1


@pytest.mark.parametrize("values,mean,sd", [([1., 2.], 1.5, .5), ([1., math.nan, 3.], 2., 1.), ([1., 2., 3.], 2.1, 1.), ([1., 2., 3.], 2., 0.81649658)])
def test_missing_or_inconsistent_replicates_rejected(values, mean, sd):
    with pytest.raises(ValueError):
        summarize_replicates(values, mean, sd)


def test_coordinate_transform_and_reference_residue_are_verified():
    assert apply_mutations("MACDEF", "A60C", -59) == "MCCDEF"
    assert apply_mutations("ACDE", "A1C", -1) == "CCDE"
    with pytest.raises(ValueError, match="Reference residue"):
        apply_mutations("ACDE", "C1A", -1)
    assert mutation_key("E9A-A1C") == "A1C-E9A"
    with pytest.raises(ValueError, match="Duplicate position"):
        mutation_key("A1C-A1D")


def test_independent_triplet_anchor_resolves_equal_mean_without_averaging():
    seq, plots, assays = fixture()
    assays.append({**assays[0], "id": "a2", "replicates": [0., 2., 4.], "sample_sd": 2.})
    plots[0]["replicate_anchor"] = {"replicates": [1., 2., 3.], "source": {"sheet": "purified"}}
    rows, excluded, _, remaining = join_campaign(seq, plots, assays)
    assert len(rows) == 1 and rows[0]["id"] == "a1" and not excluded
    assert [a["id"] for a in remaining] == ["a2"]
    plots[0]["replicate_anchor"]["replicates"] = [1., 1., 4.]
    rows, excluded, _, _ = join_campaign(seq, plots, assays)
    assert not rows and len(excluded) == 1
