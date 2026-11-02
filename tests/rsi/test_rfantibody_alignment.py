import math

import pytest

from agent.rsi.cases.rfantibody_alignment import compare_alignment_rows


def row(name, backbone, pae, rmsd):
    return {"id": name, "backbone_id": backbone, "interaction_pae": pae, "independent_target_aligned_ca_rmsd": rmsd}


def test_alignment_changes_are_not_hidden_by_global_pae_or_sequence_denominators():
    rows = [row("a0", "a", 4.0, 1.99), row("a1", "a", 6.0, 1.0), row("b0", "b", 11.0, 1.99), row("b1", "b", 12.0, 3.0)]
    result = compare_alignment_rows(rows, {"a0": 2.01, "a1": 1.0, "b0": 2.01, "b1": 3.0})
    assert result["structures"] == 4 and result["backbones"] == 2
    assert result["rmsd_threshold_changes"] == 2
    assert result["combined_filter_changes"] == 1
    assert result["current_sequence_passes"] == 2
    assert result["legacy_alignment_sequence_passes"] == 1
    assert result["current_selected_backbone_passes"] == 1
    assert result["legacy_alignment_selected_backbone_passes"] == 0
    assert result["current_any_passing_backbones"] == result["legacy_alignment_any_passing_backbones"] == 1


def test_alignment_tie_breaking_is_applied_separately_without_changing_pae():
    result = compare_alignment_rows([row("a", "p", 5.0, 1.2), row("b", "p", 5.0, 1.3)], {"a": 1.5, "b": 1.1})
    assert result["selected_design_changes"] == 1
    assert result["backbone_records"][0]["current_selected_id"] == "a"
    assert result["backbone_records"][0]["legacy_alignment_selected_id"] == "b"
    assert result["combined_filter_changes"] == 0


def test_strict_thresholds_are_not_rounded_to_pass():
    result = compare_alignment_rows([row("a", "p", 9.99, 2.0), row("b", "q", 10.0, 0.1)], {"a": 1.999, "b": 0.1})
    assert result["current_sequence_passes"] == 0
    assert result["legacy_alignment_sequence_passes"] == 1


@pytest.mark.parametrize(
    "values", [{}, {"a": 1.0, "b": 1.0}, {"a": math.nan}, {"a": math.inf}, {"a": -1.0}, {"a": True}]
)
def test_invalid_or_unmatched_legacy_metrics_cannot_produce_a_summary(values):
    with pytest.raises(ValueError):
        compare_alignment_rows([row("a", "p", 5.0, 1.0)], values)


def test_duplicate_or_empty_allocations_are_rejected():
    with pytest.raises(ValueError):
        compare_alignment_rows([], {})
    with pytest.raises(ValueError):
        compare_alignment_rows([row("a", "p", 5.0, 1.0)] * 2, {"a": 1.0})
