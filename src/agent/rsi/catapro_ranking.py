"""Reaction-level evaluation of bound, previously computed CataPro predictions."""

import math
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def ranking_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.ndim != 1 or prediction.shape != target.shape or len(target) < 2:
        raise ValueError("Ranking needs two matching vectors with at least two rows")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Ranking inputs must be finite")
    concordant = discordant = target_ties = prediction_ties = both_ties = 0
    for i in range(len(target) - 1):
        a = np.sign(prediction[i] - prediction[i + 1 :])
        b = np.sign(target[i] - target[i + 1 :])
        concordant += int(np.sum(a * b > 0))
        discordant += int(np.sum(a * b < 0))
        target_ties += int(np.sum(b == 0))
        prediction_ties += int(np.sum(a == 0))
        both_ties += int(np.sum((a == 0) & (b == 0)))
    pairs = len(target) * (len(target) - 1) // 2
    scc = None
    if np.ptp(prediction) > 0 and np.ptp(target) > 0:
        scc = float(spearmanr(prediction, target).statistic)
    return {
        "rows": len(target),
        "pairs": pairs,
        "scc": scc,
        "concordant": concordant,
        "discordant": discordant,
        "target_ties": target_ties,
        "prediction_ties": prediction_ties,
        "both_ties": both_ties,
        "strict_pair_accuracy": concordant / pairs,
        "comparable_pair_accuracy": concordant / (pairs - target_ties) if pairs > target_ties else None,
    }


def reaction_groups(rows):
    groups = defaultdict(list)
    seen = set()
    for i, row in enumerate(rows):
        identity = row[""]
        if not identity or identity in seen:
            raise ValueError("Missing or duplicate sample ID")
        seen.add(identity)
        if not row["UniProtID"] or not row["Smiles"]:
            raise ValueError("Missing reaction identity")
        groups[row["UniProtID"] + "&*&" + row["Smiles"]].append(i)
    return dict(groups)


def summarize_groups(groups):
    scc = [r["metrics"]["scc"] for r in groups if r["metrics"]["scc"] is not None]
    acc = [r["metrics"]["strict_pair_accuracy"] for r in groups]
    return {
        "groups": len(groups),
        "rows": sum(r["metrics"]["rows"] for r in groups),
        "unique_uniprot_ids": len({r["uniprot_id"] for r in groups}),
        "undefined_scc_groups": len(groups) - len(scc),
        "macro_scc": math.fsum(scc) / len(scc) if scc else None,
        "macro_strict_pair_accuracy": math.fsum(acc) / len(acc) if acc else None,
    }
