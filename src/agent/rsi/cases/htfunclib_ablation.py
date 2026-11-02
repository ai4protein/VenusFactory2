"""Prespecified sequence-group split stress test for the published EpiNNet."""

from __future__ import annotations

from agent.rsi import htfunclib_epinnet as case

SEEDS = (7, 8, 9)
MODES = ("author_row_split", "sequence_group_split")


def schedule():
    return [
        {"id": str(seed) + "_" + mode, "split_seed": seed, "mode": mode}
        for seed in SEEDS
        for mode in (MODES if seed != 8 else MODES[::-1])
    ]


def partition(data, seed, mode):
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if seed not in SEEDS or mode not in MODES:
        raise ValueError("Unregistered split allocation")
    selected = pd.concat([data[data.y == 1], data[data.negative == 1]])
    if selected.index.has_duplicates:
        raise ValueError("Positive and negative selections overlap")
    if mode == "author_row_split":
        train, test = train_test_split(selected.index.tolist(), test_size=0.2, random_state=seed)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_index, test_index = next(splitter.split(selected, groups=selected.sequence))
        train, test = selected.iloc[train_index].index.tolist(), selected.iloc[test_index].index.tolist()
    result = {"train": list(train), "test": list(test)}
    if set(train) & set(test) or set(train) | set(test) != set(selected.index):
        raise ValueError("Incomplete or overlapping row partition")
    overlap = case.overlap_diagnostics(data, result)
    if mode == "sequence_group_split" and overlap["shared_sequence_identities"]:
        raise ValueError("Sequence-group leakage")
    return result, overlap


def aggregate(rows):
    import statistics

    expected = schedule()
    if [r["id"] for r in rows] != [r["id"] for r in expected] or any(r.get("status") != "completed" for r in rows):
        raise ValueError("All six prespecified fits must complete before aggregation")
    summaries = {}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        summaries[mode] = {
            "fits": len(subset),
            "balanced_accuracy_values": [r["metrics"]["balanced_accuracy"] for r in subset],
            "mean_balanced_accuracy": statistics.mean(r["metrics"]["balanced_accuracy"] for r in subset),
            "mean_positive_recall": statistics.mean(r["metrics"]["per_class_recall"][1] for r in subset),
            "test_record_counts": [r["test_rows"] for r in subset],
            "shared_sequence_counts": [r["overlap"]["shared_sequence_identities"] for r in subset],
        }
    return {
        "modes": summaries,
        "group_minus_row_mean_balanced_accuracy": summaries[MODES[1]]["mean_balanced_accuracy"]
        - summaries[MODES[0]]["mean_balanced_accuracy"],
        "interpretation": "Descriptive partition stress test on one GFP score landscape; different test populations, three split seeds, no causal leakage effect or cross-protein claim.",
    }
