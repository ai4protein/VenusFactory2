"""Deterministic sequence-grouped holdouts for uploaded training tables."""
from collections import defaultdict
import hashlib


def sequence_key(value):
    key = "".join(str(value).split()).upper()
    if not key:
        raise ValueError("Protein sequences must not be empty.")
    return key


def holdout_indices(sequences, missing, labels=None):
    if not set(missing) <= {"validation", "test"}:
        raise ValueError("Unknown holdout split.")
    groups = defaultdict(list)
    for index, sequence in enumerate(sequences):
        groups[sequence_key(sequence)].append(index)
    if len(groups) < 10:
        raise ValueError("Automatic splitting needs at least 10 distinct protein sequences. Add more data or provide separate validation and test tables.")
    strata = defaultdict(list)
    for key, indices in groups.items():
        values = {str(labels[i]) for i in indices} if labels is not None else {"all"}
        if len(values) != 1:
            raise ValueError("The same protein sequence has conflicting class labels.")
        strata[next(iter(values))].append(key)
    result = {"train": [], **{name: [] for name in missing}}
    for keys in strata.values():
        keys.sort(key=lambda key: hashlib.sha256(("venus-holdout-v1:" + key).encode()).digest())
        count = max(1, round(len(keys) * .1))
        if len(keys) - count * len(missing) < 1:
            raise ValueError("Each class needs enough distinct sequences for separate training and held-out data.")
        offset = 0
        for name in missing:
            for key in keys[offset:offset + count]:
                result[name].extend(groups[key])
            offset += count
        for key in keys[offset:]:
            result["train"].extend(groups[key])
    return {name: sorted(indices) for name, indices in result.items()}


def assert_disjoint_sequences(splits):
    seen = {}
    for name, sequences in splits.items():
        keys = {sequence_key(sequence) for sequence in sequences}
        for previous, other in seen.items():
            if keys & other:
                raise ValueError(f"{previous} and {name} contain identical protein sequences. Keep each sequence in one data split.")
        seen[name] = keys
