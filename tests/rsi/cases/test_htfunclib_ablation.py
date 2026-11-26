import pandas as pd
import pytest

from agent.rsi import htfunclib_ablation as design


def test_group_split_keeps_repeated_sequences_together_without_dropping_records():
    data = pd.DataFrame(
        {
            "sequence": ["A" + str(i // 3) for i in range(90)],
            "y": [i % 2 for i in range(90)],
            "negative": [1 - i % 2 for i in range(90)],
        }
    )
    ids, overlap = design.partition(data, 7, "sequence_group_split")
    assert overlap["shared_sequence_identities"] == 0
    assert set(ids["train"]) | set(ids["test"]) == set(range(90))
    assert not set(ids["train"]) & set(ids["test"])
    _, row_overlap = design.partition(data, 7, "author_row_split")
    assert row_overlap["shared_sequence_identities"] > 0


def test_schedule_and_aggregation_keep_all_six_fits():
    assert len(design.schedule()) == 6
    assert [r["mode"] for r in design.schedule()][2] == "sequence_group_split"
    with pytest.raises(ValueError, match="All six"):
        design.aggregate([])
    with pytest.raises(ValueError, match="All six"):
        design.aggregate([{**r, "status": "failed"} for r in design.schedule()])


def test_unregistered_seed_or_mode_cannot_tune_on_test():
    with pytest.raises(ValueError, match="Unregistered"):
        design.partition(None, 42, "sequence_group_split")
    with pytest.raises(ValueError, match="Unregistered"):
        design.partition(None, 7, "new_split")
