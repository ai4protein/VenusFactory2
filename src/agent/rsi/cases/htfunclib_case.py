"""Pinned GFP htFuncLib processed-input classifier reproduction.

Source: Fleishman-Lab/htFuncLib, AGPL-3.0, commit recorded below.
Author notebooks remain separately attributed data, not imported executable code.
This module intentionally supports Python 3.8 for the isolated author environment.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tarfile
import warnings
from pathlib import Path

CASE_ID = "htfunclib.rf.v1"
COMMIT = "55e7fa555028b8ea7b0a2e96c85b89c75f82183b"
NOTEBOOK_SHA256 = "b6d07fed6f46c5181fc3ea2b5dd85985dbc43e7253ed6d76c103dbc00e844c7f"
DATA_SHA256 = {
    "X_train": "b525653e31676fe15e883474bd56ce613c11ee53e3c208fd3fb179a7806ec1f1",
    "X_valid": "7d1e229e2b74636d28be03825e62382326a114d1e41c23f46591df86ca4f3784",
    "X_test": "30fd2e8ffa3ff7e07e119e9e78425d6f36ca57f9caad91f5c867cea83467d7b2",
    "y_train": "7c77fda5db15beb9f27010b09baba8a029d7dc8be2fe22ee612549aadcd960cf",
    "y_valid": "ec77dd8792aa819ae119b8ee7a07a54133db0c13ffeb90a8d365e6d6365f9b44",
    "y_test": "9ce6f21c076b20fccfb9e0e51be50c175aceca64b7d6b32846eb56654e31c4ae",
    "ngs_df": "388c0648a6a406d52409faf566146a0a2285e296072f100c7fa8062449124c46",
}
SPLIT_ROWS = {"train": 116116, "valid": 29030, "test": 36287}
ENVIRONMENT = {
    "numpy": "1.20.2",
    "pandas": "1.2.3",
    "lightgbm": "3.2.1",
    "scipy": "1.6.3",
    "scikit-learn": "0.24.2",
    "joblib": "1.0.1",
    "threadpoolctl": "2.1.0",
}
LIMITATIONS = [
    "Starts from released annotated NGS data; raw sequencing, upstream Rosetta and feature generation are not reproduced.",
    "Random sequence split within one GFP library is not a cross-protein or prospective validation.",
    "Amino-acid and category encodings are fitted to all released rows, preserving the author notebook.",
    "Author Boolean-to-string filter retains all rows for the second classifier; preserved, not silently corrected.",
    "Author binary classification_report names are reversed; metrics here label False as U and True as functional.",
    "CPU threads are fixed at four; the author notebook leaves thread count implicit.",
    "Author README pins sklearn metapackage 0.0, not its runtime; sklearn/scipy versions are reconstruction choices.",
    "One seed, processed-input replay; no new model, agent autonomy, biochemical efficacy or full-paper claim.",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def verify_sources(source):
    source = Path(source)
    pins = {"RF_train.ipynb": NOTEBOOK_SHA256}
    pins.update({n + ".csv.tar.gz": h for n, h in DATA_SHA256.items()})
    for name, digest in pins.items():
        p = source / name
        if p.is_symlink() or sha256(p) != digest:
            raise ValueError("htFuncLib source identity changed: " + name)
    return pins


def read_csv_archive(path, member_name):
    """Read exactly one regular member in memory; never extract archive paths."""
    import pandas as pd

    with tarfile.open(path, "r:gz") as archive:
        entries = archive.getmembers()
        if (
            len(entries) != 1
            or entries[0].name != member_name
            or not entries[0].isfile()
            or entries[0].size > 120_000_000
            or Path(member_name).name != member_name
        ):
            raise ValueError("Unexpected htFuncLib archive member")
        with archive.extractfile(entries[0]) as stream:
            return pd.read_csv(stream)


def notebook_contract(path):
    if sha256(path) != NOTEBOOK_SHA256:
        raise ValueError("Author notebook identity changed")
    notebook = json.loads(Path(path).read_text())
    tree = ast.parse("".join(notebook["cells"][6]["source"]))
    sliced = tree.body[0].value.slice
    if isinstance(sliced, ast.Index):  # Python 3.8.
        sliced = sliced.value
    columns = ast.literal_eval(sliced)
    positions = ast.literal_eval(tree.body[1].value)
    parameters = {}
    for stage, cell in [("functionality", 18), ("colour", 29)]:
        text = "".join(notebook["cells"][cell]["source"])
        tree = ast.parse("\n".join(line for line in text.splitlines() if not line.startswith("%")))
        call = tree.body[0].value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "LGBMClassifier"
            or call.args
        ):
            raise ValueError("Unexpected classifier definition")
        parameters[stage] = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        if parameters[stage]["n_estimators"] != 10000 or parameters[stage]["random_state"] != 42:
            raise ValueError("Author iteration or seed allocation changed")
    return columns, positions, parameters


def compare_partition(actual_x, actual_y, expected_x, expected_y):
    import numpy as np

    if (
        list(actual_x.columns) != list(expected_x.columns)
        or actual_x.shape != expected_x.shape
        or list(actual_y.columns) != ["assign"]
        or not np.array_equal(actual_y["assign"].values, expected_y.values)
    ):
        raise ValueError("Released partition rows, feature order or labels differ")
    a, b = actual_x.to_numpy(dtype=float), expected_x.to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite released features")
    delta = float(np.max(np.abs(a - b)))
    if delta > 1e-12:
        raise ValueError("Released features differ from reconstructed author split")
    return delta


def legacy_second_stage_rows(binary_predictions):
    """Preserve the author's Boolean-array == 'U' behaviour, retaining its warning."""
    import numpy as np

    values = np.asarray(binary_predictions)
    if values.dtype.kind != "b" or values.ndim != 1:
        raise ValueError("Expected one-dimensional binary predictions")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        removed = np.flatnonzero(values == "U")
    return np.delete(np.arange(len(values)), removed), [str(x.message) for x in caught]


def chained_predictions(binary_predictions, colour_predictions):
    import numpy as np

    first, second = np.asarray(binary_predictions), np.asarray(colour_predictions)
    if first.dtype.kind != "b" or first.ndim != 1 or first.shape != second.shape:
        raise ValueError("Chained prediction shape or binary type changed")
    if not set(second.tolist()) <= {"U", "C", "G", "G/C"}:
        raise ValueError("Unknown GFP class")
    return np.where(first, second, "U")


def classification_metrics(truth, prediction, labels):
    """Independent confusion-count metric implementation, with explicit class order."""
    labels = list(labels)
    truth, prediction = list(truth), list(prediction)
    if not truth or len(truth) != len(prediction) or len(set(labels)) != len(labels):
        raise ValueError("Incomplete prediction allocation")
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for actual, predicted in zip(truth, prediction):
        if actual not in index or predicted not in index:
            raise ValueError("Unexpected prediction class")
        matrix[index[actual]][index[predicted]] += 1
    supports = [sum(row) for row in matrix]
    if any(n == 0 for n in supports):
        raise ValueError("A declared class is absent from evaluation")
    recall = [matrix[i][i] / supports[i] for i in range(len(labels))]
    return {
        "labels": labels,
        "rows": len(truth),
        "confusion_matrix": matrix,
        "support": supports,
        "per_class_recall": recall,
        "accuracy": sum(matrix[i][i] for i in range(len(labels))) / len(truth),
        "balanced_accuracy": sum(recall) / len(labels),
    }


def admit(source):
    import numpy as np
    from sklearn.model_selection import train_test_split

    source = Path(source)
    pins = verify_sources(source)
    columns, positions, parameters = notebook_contract(source / "RF_train.ipynb")
    raw = read_csv_archive(source / "ngs_df.csv.tar.gz", "ngs_df.csv")
    if len(raw) != sum(SPLIT_ROWS.values()) or raw["pseudo_prot"].duplicated().any():
        raise ValueError("Unexpected GFP population or duplicate sequence")
    if set(raw["assign"]) != {"U", "C", "G", "G/C"}:
        raise ValueError("Unexpected GFP labels")
    data = raw[columns].copy()
    encodings = {}
    for column in positions:
        mapping = {v: i for i, v in enumerate(sorted(data[column].unique()))}
        encodings[column] = mapping
        data[column] = data[column].map(mapping)
    cat_columns = [c for c in data if c.endswith("_cat")]
    mapping = {v: i for i, v in enumerate(sorted(np.unique(data[cat_columns].values)))}
    for column in cat_columns:
        data[column] = data[column].map(mapping)
    development, test = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
    train, valid = train_test_split(development, test_size=0.2, random_state=42)
    indices = {"train": train, "valid": valid, "test": test}
    features, targets, identities, partitions = {}, {}, {}, {}
    for name, idx in indices.items():
        x = read_csv_archive(source / ("X_" + name + ".csv.tar.gz"), "X_" + name + ".csv")
        y = read_csv_archive(source / ("y_" + name + ".csv.tar.gz"), "y_" + name + ".csv")
        error = compare_partition(x, y, data.drop(columns="assign").iloc[idx], data["assign"].iloc[idx])
        if len(x) != SPLIT_ROWS[name]:
            raise ValueError("Partition size changed")
        features[name], targets[name] = x, y["assign"].to_numpy()
        identities[name] = raw.iloc[idx][["pseudo_prot", "assign"]].reset_index(drop=True)
        partitions[name] = {
            "rows": len(idx),
            "max_abs_feature_difference": error,
            "label_counts": {str(k): int(v) for k, v in y["assign"].value_counts().items()},
            "source_row_order_sha256": hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest(),
        }
    protocol = {
        "case": CASE_ID,
        "source_repository": "https://github.com/Fleishman-Lab/htFuncLib",
        "source_commit": COMMIT,
        "source_license": "AGPL-3.0",
        "paper_doi": "10.1038/s41467-023-38099-z",
        "release_doi": "10.5281/zenodo.7745581",
        "source_files_sha256": pins,
        "rows": len(raw),
        "features": list(features["train"]),
        "partitions": partitions,
        "position_encodings": encodings,
        "category_encoding": mapping,
        "author_parameters": parameters,
        "runtime_parameter_overrides": {"n_jobs": 4},
        "scope_limitations": LIMITATIONS,
        "full_paper_reproduced": False,
        "published_binary_test_balanced_accuracy": 0.8926395148226008,
        "reference_tolerance_absolute": 1e-6,
        "reference_is_not_a_model_selection_target": True,
        "stage_training_limit": "10000 estimators each; no truncation, early stopping or test-based selection",
        "second_stage_filter": "author_boolean_equals_U_retains_all",
    }
    return protocol, features, targets, identities


def canonical_htfunclib(report):
    if (
        report.get("status") != "completed"
        or report.get("execution_origin") != "new_processed_input_training"
        or report.get("case") != CASE_ID
        or report.get("full_paper_reproduced") is not False
        or report.get("agent_generated_model") is not False
        or report.get("source_commit") != COMMIT
    ):
        raise ValueError("Unsupported GFP evidence claim")
    if report.get("completed_stages") != ["functionality", "colour"]:
        raise ValueError("Incomplete GFP training chain")
    if (
        report.get("split_rows") != SPLIT_ROWS
        or report.get("second_stage_retained_rows") != SPLIT_ROWS
        or report.get("completed_boosting_iterations") != {"functionality": 10000, "colour": 10000}
        or report.get("native_prediction_replay_max_abs_error") != {"functionality": 0.0, "colour": 0.0}
    ):
        raise ValueError("GFP scope, iterations or native inference mismatch")
    return {
        "computational_scope": "htfunclib_gfp_processed_input_two_stage_training",
        "execution_origin": report["execution_origin"],
        "source_commit": COMMIT,
        "protein": "GFP",
        "rows": sum(SPLIT_ROWS.values()),
        "split_rows": SPLIT_ROWS,
        "completed_boosting_iterations": report["completed_boosting_iterations"],
        "second_stage_filter_issue_preserved": True,
        "agent_generated_model": False,
        "full_paper_reproduced": False,
        "new_algorithm_or_agent_superiority_established": False,
        "test_metrics": report["metrics"]["test"],
        "binary_reference_comparison": report["binary_reference_comparison"],
        "scope_limitations": LIMITATIONS,
    }


def verify_artifacts(report, output, source):
    """Controller-side numerical verification from bound, per-sequence predictions."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    output = Path(output)
    source = Path(source)
    verify_sources(source)
    _, _, author_parameters = notebook_contract(source / "RF_train.ipynb")
    raw = read_csv_archive(source / "ngs_df.csv.tar.gz", "ngs_df.csv")
    development, test = train_test_split(list(range(len(raw))), test_size=0.2, random_state=42)
    train, valid = train_test_split(development, test_size=0.2, random_state=42)
    row_ids = {"train": train, "valid": valid, "test": test}
    canonical_htfunclib(report)
    if sha256(output / "protocol.json") != report["protocol_sha256"]:
        raise ValueError("GFP protocol changed")
    protocol = json.loads((output / "protocol.json").read_text())
    expected_pins = {"RF_train.ipynb": NOTEBOOK_SHA256}
    expected_pins.update({n + ".csv.tar.gz": h for n, h in DATA_SHA256.items()})
    if (
        protocol["source_commit"] != COMMIT
        or protocol["author_parameters"] != author_parameters
        or protocol["environment"] != ENVIRONMENT
        or protocol["published_binary_test_balanced_accuracy"] != 0.8926395148226008
        or protocol["reference_tolerance_absolute"] != 1e-6
        or protocol["admission_only"] is not False
        or protocol["source_files_sha256"] != expected_pins
        or protocol["runtime_parameter_overrides"] != {"n_jobs": 4}
        or protocol["second_stage_filter"] != "author_boolean_equals_U_retains_all"
        or protocol["scope_limitations"] != LIMITATIONS
    ):
        raise ValueError("GFP admitted protocol changed")
    required = {
        "functionality.txt",
        "colour.txt",
        "functionality.classes.json",
        "colour.classes.json",
        "functionality.evaluations.json",
        "colour.evaluations.json",
        "htfunclib_case.py",
        "run_htfunclib_case.py",
    }
    required.update(n + ".predictions.csv" for n in SPLIT_ROWS)
    required.update(n + ".identities.csv" for n in SPLIT_ROWS)
    if set(report["artifact_sha256"]) != required:
        raise ValueError("Incomplete GFP artifact set")
    for name, digest in report["artifact_sha256"].items():
        p = output / name
        if p.is_symlink() or sha256(p) != digest:
            raise ValueError("GFP artifact changed: " + name)
    all_sequences = set()
    for split, n in SPLIT_ROWS.items():
        frame = pd.read_csv(output / (split + ".predictions.csv"))
        ids = pd.read_csv(output / (split + ".identities.csv"))
        original_ids = raw.iloc[row_ids[split]][["pseudo_prot", "assign"]].reset_index(drop=True)
        if (
            not ids.equals(original_ids)
            or len(frame) != n
            or list(frame) != ["pseudo_prot", "assign", "functionality", "colour", "chained"]
            or not frame[["pseudo_prot", "assign"]].equals(ids)
            or frame["pseudo_prot"].duplicated().any()
            or all_sequences.intersection(frame["pseudo_prot"])
        ):
            raise ValueError("GFP prediction identity or split overlap changed")
        all_sequences.update(frame["pseudo_prot"])
        if not (
            chained_predictions(frame["functionality"].to_numpy(), frame["colour"].to_numpy())
            == frame["chained"].to_numpy()
        ).all():
            raise ValueError("GFP chained prediction rule changed")
        computed = {
            "functionality": classification_metrics(frame["assign"] != "U", frame["functionality"], [False, True]),
            "colour": classification_metrics(frame["assign"], frame["colour"], ["C", "G", "G/C", "U"]),
            "chained": classification_metrics(frame["assign"], frame["chained"], ["C", "G", "G/C", "U"]),
        }
        if computed != report["metrics"][split]:
            raise ValueError("GFP reported metrics disagree with predictions")
    actual = report["metrics"]["test"]["functionality"]["balanced_accuracy"]
    ref = protocol["published_binary_test_balanced_accuracy"]
    expected = {
        "published": ref,
        "actual": actual,
        "absolute_difference": abs(actual - ref),
        "tolerance": 1e-6,
        "within_tolerance": abs(actual - ref) <= 1e-6,
    }
    if report["binary_reference_comparison"] != expected:
        raise ValueError("GFP author reference discrepancy changed")
    return canonical_htfunclib(report)
