import pytest
from mcp_tools.models.factory.training.data.split_policy import holdout_indices, assert_disjoint_sequences


def test_single_table_holdouts_are_disjoint_and_keep_duplicate_sequences_together():
    sequences = ["ACDE" + str(i) for i in range(40)] * 2
    labels = [i % 2 for i in range(40)] * 2
    indices = holdout_indices(sequences, ["validation", "test"], labels)
    sets = {name: {sequences[i] for i in rows} for name, rows in indices.items()}
    assert_disjoint_sequences(sets)
    assert sum(len(v) for v in indices.values()) == len(sequences)
    assert {k: len(v) for k, v in indices.items()} == {"train": 64, "validation": 8, "test": 8}
    for rows in indices.values(): assert {labels[i] for i in rows} == {0, 1}
    assert indices == holdout_indices(sequences, ["validation", "test"], labels)


def test_missing_test_is_drawn_from_training_and_external_validation_is_preserved():
    sequences = ["SEQ" + str(i) for i in range(20)]
    indices = holdout_indices(sequences, ["test"])
    assert set(indices) == {"train", "test"}
    assert not set(indices["train"]) & set(indices["test"])


def test_reused_or_too_small_tables_fail_explicitly():
    with pytest.raises(ValueError, match="10 distinct"): holdout_indices(["ACD"] * 10, ["validation", "test"])
    with pytest.raises(ValueError, match="identical"): assert_disjoint_sequences({"train": [" ACD "], "test": ["acd"]})
