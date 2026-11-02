"""Ordering and tie semantics in scientific selection metrics."""

import numpy as np
import pytest

from agent.rsi.cases.catapro_ranking import ranking_metrics, reaction_groups, summarize_groups


def test_perfect_and_reversed_selection():
    good = ranking_metrics([3, 1, 2], [30, 10, 20])
    bad = ranking_metrics([3, 1, 2], [10, 30, 20])
    assert good["pairs"] == good["concordant"] == 3
    assert good["scc"] == good["strict_pair_accuracy"] == 1
    assert bad["discordant"] == 3
    assert bad["scc"] == -1 and bad["strict_pair_accuracy"] == 0


def test_ties_do_not_get_credit_for_selecting_a_better_mutant():
    m = ranking_metrics([0, 0, 2, 3], [0, 1, 1, 3])
    assert m["pairs"] == 6 and m["concordant"] == 4
    assert m["prediction_ties"] == m["target_ties"] == 1
    assert m["both_ties"] == 0
    assert m["strict_pair_accuracy"] == 4 / 6
    assert m["comparable_pair_accuracy"] == 4 / 5


def test_constant_groups_are_explicit_not_dropped_or_invented():
    m = ranking_metrics([1, 1, 1], [2, 2, 2])
    assert m["scc"] is None and m["comparable_pair_accuracy"] is None
    assert m["strict_pair_accuracy"] == 0 and m["both_ties"] == 3
    s = summarize_groups([{"uniprot_id": "A", "metrics": m}])
    assert s["groups"] == s["undefined_scc_groups"] == 1
    assert s["macro_scc"] is None


@pytest.mark.parametrize("pred,target", [([1], [2]), ([1, 2], [2]), ([[1, 2]], [[1, 2]]), ([1, np.nan], [1, 2])])
def test_invalid_vectors_fail(pred, target):
    with pytest.raises(ValueError):
        ranking_metrics(pred, target)


def test_groups_keep_wild_type_and_mutants_separate_substrates_and_reject_duplicate_ids():
    rows = [
        {"": "a", "UniProtID": "U", "Smiles": "C"},
        {"": "b", "UniProtID": "U", "Smiles": "C"},
        {"": "c", "UniProtID": "U", "Smiles": "O"},
    ]
    assert reaction_groups(rows) == {"U&*&C": [0, 1], "U&*&O": [2]}
    with pytest.raises(ValueError, match="duplicate"):
        reaction_groups(rows + rows[:1])
    with pytest.raises(ValueError, match="identity"):
        reaction_groups([{**rows[0], "UniProtID": ""}])


def test_macro_averaging_does_not_treat_substrates_as_independent_proteins():
    groups = [
        {"uniprot_id": "U", "metrics": ranking_metrics([1, 2], [1, 2])},
        {"uniprot_id": "U", "metrics": ranking_metrics([1, 2, 3], [3, 2, 1])},
    ]
    s = summarize_groups(groups)
    assert s["groups"] == 2 and s["unique_uniprot_ids"] == 1
    assert s["macro_strict_pair_accuracy"] == 0.5 and s["macro_scc"] == pytest.approx(0, abs=1e-15)
