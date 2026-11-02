import pytest

from agent.rsi.landscape_cases import partition


def fixture_rows():
    return [{"id": str(i), "sequence": str(i), "target": 0.0 if i % 2 else float(i)} for i in range(12)]


def test_partitions_do_not_depend_on_label_and_retain_zeros():
    rows = fixture_rows()
    a = partition(rows, "GB1", (3, 4, 5))
    changed = [{**r, "target": 100 - r["target"]} for r in rows]
    b = partition(changed, "GB1", (3, 4, 5))
    assert [[r["id"] for r in x] for x in a] == [[r["id"] for r in x] for x in b]
    assert len({r["id"] for x in a for r in x}) == 12
    assert any(r["target"] == 0 for x in a for r in x)


def test_duplicate_or_nonfinite_labels_rejected():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="duplicate"):
        partition(rows + [rows[0]], "GB1", (3, 4, 5))
    rows[0]["target"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        partition(rows, "GB1", (3, 4, 5))


def test_fresh_partitions_exclude_all_prior_sequences_before_sampling():
    rows = fixture_rows()
    previous = {r["sequence"] for r in rows[:3]}
    splits = partition(rows, "GB1", (3, 3, 3), seed=20261012, exclude_sequences=previous)
    assert not previous & {r["sequence"] for part in splits for r in part}
    with pytest.raises(ValueError, match="unseen"):
        partition(rows, "GB1", (4, 4, 4), exclude_sequences=previous)
