"""Reconstruct and audit the author's GFP EpiNNet-to-library calculation.

The published score table and notebook are pinned public inputs. Author pickle
files and shell jobs are never loaded or executed. Repeated sequence records in
the original train/test split are retained and disclosed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

from .htfunclib_case import COMMIT, ENVIRONMENT, classification_metrics, sha256

CASE_ID = "htfunclib.epinnet.v1"
SOURCE_FILES = {
    "02_htFuncLib_aggregation_and_analysis.ipynb": "ee794c1bbec1ffddd2bfec5250eb9ba0cbc104038795bb25b979e1d066895714",
    "bubb_design/score_df.csv": "bf82a01482ea654df784aed2676735ceffb7cd8da53bdc2bacfcd3b732fb5fd9",
    "test_10k/all_10k_test.sc": "e0f7dda00597d4c1e4e9c924ed8afa68dae862a7cfa7ac537dba342e0d29bef9",
}
ORDER_SHA256 = {
    "all": "2a39112ba3931c9681b724f8b2f4ed04a0e7420614a005b323b40e377fc71d1d",
    "train": "0e955cfc5c603e6883b9cfd2b24c7d678be0e46aaf36a55584d9ae3c74a2f073",
    "test": "5511b443bc285d1e46873862043b7d0b3ed172f58c319cbb5efbd3c8d2e04df0",
}
PARAMETERS = {
    "hidden_layer_sizes": 26,
    "activation": "logistic",
    "learning_rate": "invscaling",
    "random_state": 42,
    "verbose": True,
    "max_iter": 20000,
}
LIMITATIONS = [
    "Starts from published Rosetta scores; atomistic design jobs are not rerun.",
    "The author's row split has 326 sequence identities in both training and test sets; it is not unseen-sequence generalization.",
    "Rosetta energy labels are computational proxy labels, not measured fluorescence.",
    "The single-feature basis-vector ranking is the author heuristic, not a physical single-mutant prediction.",
    "The top-40 library choice is fixed to the notebook, not selected by our test outcomes.",
    "The archived 10k-panel scores are reanalysed author outputs, not newly calculated or agent-generated scores.",
    "The author's random 10k job generator is unseeded and samples positions with replacement.",
    "The sklearn runtime was not fully locked by the author; reconstruction versions and any discrepancies are retained.",
    "One author-seed replay does not establish autonomous new-model development, cross-protein transfer or full-paper reproduction.",
]


def order_digest(ids):
    import numpy as np

    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def source_contract(source):
    source = Path(source)
    for name, digest in SOURCE_FILES.items():
        path = source / name
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError("EpiNNet source changed: " + name)
    notebook = json.loads((source / "02_htFuncLib_aggregation_and_analysis.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][6]["source"]))
    positions = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALL_POSS" for t in node.targets)
    )
    positions = [str(p) + "A" for p in positions]
    call = ast.parse("".join(notebook["cells"][34]["source"])).body[0].value
    parameters = {
        kw.arg: len(positions) if kw.arg == "hidden_layer_sizes" else ast.literal_eval(kw.value) for kw in call.keywords
    }
    if parameters != PARAMETERS:
        raise ValueError("EpiNNet author training allocation changed")

    def output_text(index):
        return "".join(
            "".join(o.get("text", o.get("data", {}).get("text/plain", [])))
            for o in notebook["cells"][index].get("outputs", [])
        )

    reference_ranking = {}
    for position, aa, _p0, p1 in re.findall(
        r"^\s*\d+\s+(\d+A)\s+([A-Z])\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$", output_text(46), re.MULTILINE
    ):
        reference_ranking[position + "_" + aa] = float(p1)
    if len(reference_ranking) != 91:
        raise ValueError("Incomplete author ranking output")
    reference_library = dict(re.findall(r"^(\d+A)\s+([A-Z]+)\s*$", output_text(61), re.MULTILINE))
    if set(reference_library) != set(positions) or math.prod(map(len, reference_library.values())) != 414720:
        raise ValueError("Incomplete author library output")
    return positions, reference_ranking, reference_library


def overlap_diagnostics(data, partitions):
    training = data.loc[partitions["train"]]
    test = data.loc[partitions["test"]]
    shared = set(training["sequence"]) & set(test["sequence"])
    labeled = data.loc[partitions["train"] + partitions["test"]]
    conflicting = labeled.groupby("sequence")["y"].nunique()
    return {
        "training_unique_sequences": int(training["sequence"].nunique()),
        "test_unique_sequences": int(test["sequence"].nunique()),
        "shared_sequence_identities": len(shared),
        "test_records_with_training_sequence": int(test["sequence"].isin(shared).sum()),
        "test_records_with_unseen_sequence": int((~test["sequence"].isin(shared)).sum()),
        "shared_source_row_ids": len(set(training.index) & set(test.index)),
        "labeled_sequences_with_conflicting_energy_labels": int((conflicting > 1).sum()),
    }


def prepare(source, frozen_orders=None):
    """Frozen orders make audit independent of sorting changes in newer NumPy."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    positions, reference_ranking, reference_library = source_contract(source)
    data = pd.read_csv(Path(source) / "bubb_design/score_df.csv")
    if len(data) != 9574 or data.index.has_duplicates or not np.isfinite(data.total_score).all():
        raise ValueError("Unexpected Rosetta input table")
    if frozen_orders is None:
        data = data.sort_values("total_score")
    else:
        if set(frozen_orders) != set(ORDER_SHA256):
            raise ValueError("Missing frozen split")
        for name, ids in frozen_orders.items():
            if order_digest(ids) != ORDER_SHA256[name]:
                raise ValueError("Frozen author row order changed")
        data = data.loc[frozen_orders["all"]]
    wt_rows = data[data.num_muts == 0]
    if len(wt_rows) != 48 or float(wt_rows.iloc[0].total_score) != -723.939:
        raise ValueError("Wild-type threshold changed")
    data["y"] = (data.total_score < wt_rows.iloc[0].total_score).astype(int)
    data["negative"] = np.r_[np.zeros(4787), np.ones(4787)].astype(int)
    data["sequence"] = data[positions].agg("".join, axis=1)
    x = pd.get_dummies(data[positions])
    x = x.loc[:, ~x.columns.duplicated()]
    selected = pd.concat([data[data.y == 1], data[data.negative == 1]])
    if frozen_orders is None:
        train, test = train_test_split(selected, test_size=0.2, random_state=42)
        partitions = {"all": data.index.tolist(), "train": train.index.tolist(), "test": test.index.tolist()}
    else:
        partitions = frozen_orders
    if {k: order_digest(v) for k, v in partitions.items()} != ORDER_SHA256:
        raise ValueError("Author row order or split changed")
    if (
        x.shape != (9574, 91)
        or int(data.y.sum()) != 1444
        or len(selected) != 6231
        or len(partitions["train"]) != 4984
        or len(partitions["test"]) != 1247
    ):
        raise ValueError("Author feature or label population changed")
    overlap = overlap_diagnostics(data, partitions)
    if overlap["shared_sequence_identities"] != 326 or overlap["shared_source_row_ids"] != 0:
        raise ValueError("Author sequence overlap changed")
    protocol = {
        "case": CASE_ID,
        "source_commit": COMMIT,
        "source_files_sha256": SOURCE_FILES,
        "source_license": "AGPL-3.0",
        "paper_doi": "10.1038/s41467-023-38099-z",
        "parameters": PARAMETERS,
        "environment": ENVIRONMENT,
        "rows": len(data),
        "positions": positions,
        "feature_columns": list(x),
        "partitions": partitions,
        "order_sha256": ORDER_SHA256,
        "training_rows": 4984,
        "test_rows": 1247,
        "energy_positive_cutoff": -723.939,
        "negative_rule": "worst half of published sorted score rows",
        "wt_threshold_row": int(wt_rows.index[0]),
        "wt_after_restore_row": int(wt_rows.index[1]),
        "wt_after_restore_energy": float(wt_rows.iloc[1].total_score),
        "overlap": overlap,
        "author_reference": {
            "test_confusion_matrix": [[920, 12], [16, 299]],
            "all_confusion_matrix": [[7103, 1027], [47, 1397]],
            "training_iterations": 151,
            "ranking_positive_probability": reference_ranking,
            "chosen_library": reference_library,
            "library_size": 414720,
            "printed_probability_tolerance": 5.1e-7,
        },
        "scope_limitations": LIMITATIONS,
        "selection_prefix_rows": 40,
        "full_paper_reproduced": False,
        "new_rosetta_jobs_executed": 0,
    }
    return protocol, data, x


def forward(weights, features):
    import numpy as np
    from scipy.special import expit

    x = np.asarray(features, dtype=float)
    required = {"w0", "b0", "w1", "b1"}
    if set(weights) != required:
        raise ValueError("Unexpected numerical weight schema")
    shapes = {"w0": (91, 26), "b0": (26,), "w1": (26, 1), "b1": (1,)}
    if any(weights[k].shape != shapes[k] or not np.isfinite(weights[k]).all() for k in required):
        raise ValueError("Invalid numerical weight shape or values")
    if x.ndim != 2 or x.shape[1] != 91 or not np.isfinite(x).all():
        raise ValueError("Invalid one-hot input matrix")
    hidden = expit(x @ weights["w0"] + weights["b0"])
    positive = expit(hidden @ weights["w1"] + weights["b1"]).ravel()
    return np.column_stack([1 - positive, positive])


def rank_and_select(protocol, data, probabilities):
    import numpy as np
    import pandas as pd

    if np.asarray(probabilities).shape != (91,) or not np.isfinite(probabilities).all():
        raise ValueError("Incomplete feature-ranking predictions")
    ranking = pd.DataFrame(
        {
            "feature": protocol["feature_columns"],
            "positive_probability": probabilities,
        }
    )
    ranking[["position", "amino_acid"]] = ranking.feature.str.split("_", expand=True)
    ranking = ranking.sort_values("positive_probability", ascending=False)
    wt = data.loc[protocol["wt_threshold_row"]]
    records, chosen = [], None
    for count in range(len(ranking)):  # Author notebook omits the final full-prefix row.
        space = {}
        prefix = ranking.iloc[:count]
        for p in protocol["positions"]:
            aas = prefix.loc[prefix.position == p, "amino_acid"].tolist()
            space[p] = str(wt[p]) + "".join(aa for aa in aas if aa != wt[p])
        subset = data
        for position, aas in space.items():
            subset = subset[subset[position].isin(list(aas))]
        record = {
            "prefix_rows": count,
            "library_size": math.prod(map(len, space.values())),
            "good": int(subset.y.sum()),
            "bad": int(subset.negative.sum()),
            "total": len(subset),
        }
        record["rest"] = record["total"] - record["good"] - record["bad"]
        records.append(record)
        if count == 40:
            chosen = space
    return ranking, records, chosen


def archived_rosetta_panel(source, threshold):
    import pandas as pd

    table = pd.read_csv(Path(source) / "test_10k/all_10k_test.sc", sep=r"\s+")
    counts = {str(k): int(v) for k, v in table.seq_space.value_counts().items()}
    if counts != {"Filtered": 8786, "EpiNNet": 8065}:
        raise ValueError("Archived Rosetta panel population changed")
    result = {"origin": "reanalysis_of_published_scores", "new_jobs": 0, "score_rows": len(table), "groups": {}}
    for group, frame in table.groupby("seq_space"):
        result["groups"][group] = {
            "rows": len(frame),
            "unique_descriptions": int(frame.description.nunique()),
            "below_wt": int((frame.total_score < threshold).sum()),
            "equal_wt": int((frame.total_score == threshold).sum()),
            "above_wt": int((frame.total_score > threshold).sum()),
            "mean_score": float(frame.total_score.mean()),
        }
    return result


def reference_comparison(protocol, metrics, ranking, chosen, iterations):
    reference = protocol["author_reference"]
    differences = [
        abs(float(row.positive_probability) - reference["ranking_positive_probability"][row.feature])
        for row in ranking.itertuples()
    ]
    return {
        "test_confusion_matrix_matches": metrics["test"]["confusion_matrix"] == reference["test_confusion_matrix"],
        "all_confusion_matrix_matches": metrics["all"]["confusion_matrix"] == reference["all_confusion_matrix"],
        "training_iterations": iterations,
        "author_training_iterations": 151,
        "ranking_probability_max_abs_difference": max(differences),
        "ranking_within_printed_precision": max(differences) <= reference["printed_probability_tolerance"],
        "selected_library_matches": chosen == reference["chosen_library"],
        "selected_library_size": math.prod(map(len, chosen.values())),
        "author_selected_library_size": reference["library_size"],
    }


def canonical_epinnet(report):
    if (
        report.get("status") != "completed"
        or report.get("case") != CASE_ID
        or report.get("execution_origin") != "new_author_epinnet_training"
        or report.get("full_paper_reproduced") is not False
        or report.get("agent_generated_model") is not False
        or report.get("new_rosetta_jobs_executed") != 0
        or report.get("source_commit") != COMMIT
        or report.get("completed_stages")
        != ["data_reconstruction", "training", "mutation_ranking", "library_selection", "published_score_reanalysis"]
    ):
        raise ValueError("Incomplete or unsupported EpiNNet evidence")
    if (
        report.get("overlap", {}).get("shared_sequence_identities") != 326
        or report.get("native_probability_max_abs_error", 1) > 1e-10
    ):
        raise ValueError("EpiNNet leakage or numerical evidence missing")
    return {
        "computational_scope": "htfunclib_epinnet_from_published_rosetta_scores_to_library",
        "execution_origin": report["execution_origin"],
        "source_commit": COMMIT,
        "training_rows": 4984,
        "test_rows": 1247,
        "one_hot_features": 91,
        "overlap": report["overlap"],
        "reference_comparison": report["reference_comparison"],
        "test_metrics": report["metrics"]["test"],
        "agent_generated_model": False,
        "new_rosetta_jobs_executed": 0,
        "full_paper_reproduced": False,
        "new_algorithm_or_agent_superiority_established": False,
        "scope_limitations": LIMITATIONS,
    }


def verify_artifacts(report, output, source):
    import numpy as np
    import pandas as pd

    output = Path(output)
    canonical_epinnet(report)
    if sha256(output / "protocol.json") != report["protocol_sha256"]:
        raise ValueError("EpiNNet protocol changed")
    protocol = json.loads((output / "protocol.json").read_text())
    expected, data, x = prepare(source, protocol["partitions"])
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError("EpiNNet admission changed: " + key)
    required = {
        "weights.npz",
        "loss_curve.json",
        "ranking.csv",
        "library_path.json",
        "chosen_library.json",
        "all.predictions.csv",
        "train.predictions.csv",
        "test.predictions.csv",
        "htfunclib_case.py",
        "htfunclib_epinnet.py",
        "run_htfunclib_epinnet.py",
    }
    if set(report["artifact_sha256"]) != required:
        raise ValueError("Missing EpiNNet numerical artifacts")
    for name, digest in report["artifact_sha256"].items():
        if (output / name).is_symlink() or sha256(output / name) != digest:
            raise ValueError("EpiNNet artifact changed: " + name)
    with np.load(output / "weights.npz", allow_pickle=False) as weights:
        metrics = {}
        for name, ids in protocol["partitions"].items():
            frame = pd.read_csv(output / (name + ".predictions.csv"))
            target = data.loc[ids]
            probabilities = forward(weights, x.loc[ids])
            if (
                list(frame) != ["source_row", "sequence", "label", "probability", "prediction"]
                or frame.source_row.tolist() != ids
                or frame.sequence.tolist() != target.sequence.tolist()
                or not np.array_equal(frame.label.to_numpy(), target.y.to_numpy())
                or not np.isfinite(frame.probability).all()
                or not np.allclose(frame.probability, probabilities[:, 1], rtol=0, atol=1e-10)
                or not np.array_equal(frame.prediction.to_numpy(), (probabilities[:, 1] > 0.5).astype(int))
            ):
                raise ValueError("EpiNNet source identity or native probability mismatch")
            metrics[name] = classification_metrics(frame.label, frame.prediction, [0, 1])
        ranking, library_path, chosen = rank_and_select(protocol, data, forward(weights, np.eye(91))[:, 1])
    stored_ranking = pd.read_csv(output / "ranking.csv")
    if (
        stored_ranking.feature.tolist() != ranking.feature.tolist()
        or not np.allclose(stored_ranking.positive_probability, ranking.positive_probability, rtol=0, atol=1e-10)
        or json.loads((output / "library_path.json").read_text()) != library_path
        or json.loads((output / "chosen_library.json").read_text()) != chosen
        or metrics != report["metrics"]
        or report["overlap"] != protocol["overlap"]
    ):
        raise ValueError("EpiNNet metric, ranking or library mismatch")
    losses = json.loads((output / "loss_curve.json").read_text())
    if not losses or len(losses) != report["training_iterations"] or not np.isfinite(losses).all():
        raise ValueError("EpiNNet convergence trace missing")
    comparison = reference_comparison(protocol, metrics, ranking, chosen, len(losses))
    actual = report["reference_comparison"]
    if any(actual[k] != v for k, v in comparison.items() if k != "ranking_probability_max_abs_difference"):
        raise ValueError("EpiNNet reference comparison changed")
    if (
        abs(actual["ranking_probability_max_abs_difference"] - comparison["ranking_probability_max_abs_difference"])
        > 1e-10
    ):
        raise ValueError("EpiNNet reference numerical error changed")
    panel = archived_rosetta_panel(source, protocol["wt_after_restore_energy"])
    observed_panel = report["published_score_panel"]
    if (
        set(observed_panel) != set(panel)
        or any(observed_panel[k] != panel[k] for k in panel if k != "groups")
        or set(observed_panel["groups"]) != set(panel["groups"])
    ):
        raise ValueError("Published Rosetta reanalysis changed")
    for group, row in panel["groups"].items():
        observed = observed_panel["groups"][group]
        if set(observed) != set(row) or any(
            not math.isclose(observed[k], v, rel_tol=0, abs_tol=1e-10) if k == "mean_score" else observed[k] != v
            for k, v in row.items()
        ):
            raise ValueError("Published Rosetta reanalysis changed")
    return canonical_epinnet(report)
