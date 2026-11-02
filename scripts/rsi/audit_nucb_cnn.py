"""Independently audit saved NucB inference evidence without retraining models.

Reconstructs cohorts from both public landscapes and ensemble aggregation from
saved per-model probabilities. Does not claim a second neural-network forward run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("non-functional", "activity_greater_than_0", "activity_greater_than_WT", "activity_greater_than_A73R")
ALIASES = dict(zip(("non-functional", "activity > 0", "activity > WT", "activity > A73R"), LABELS))
ALIASES.update({x: x for x in LABELS})


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frame(path):
    data = pd.read_csv(path)
    if "Unnamed: 0" in data:
        require(data["Unnamed: 0"].notna().all() and data["Unnamed: 0"].is_unique, "Invalid serialized row index")
        data = data.drop(columns="Unnamed: 0")
    require(data.sequence.is_unique, "Repeated sequence identity")
    require(data.activity_level.notna().all() and set(data.activity_level) <= set(ALIASES), "Unknown activity label")
    data["activity_level"] = data.activity_level.map(ALIASES)
    for name in ["mutations", "sublibrary_names", "generations"]:
        data[name] = data[name].map(lambda x: tuple(sorted(ast.literal_eval(x))))
    return data


def audit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.run / "report.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    require(report["status"] == "replayed", "Inference run did not pass its declared references")
    for row in protocol["source_manifest"]["files"]:
        require(sha(args.assets / "source" / row["path"]) == row["sha256"], "Author source changed")
    for row in protocol["inputs"]:
        require(sha(args.assets / "gcs" / row["path"]) == row["sha256"], "Public weight/data changed")
    for name, expected in protocol["implementation"].items():
        require(sha(args.run / "implementation" / name) == expected, "Archived runtime changed")
    require(sha(args.git_landscape) == protocol["git_landscape_sha256"], "Git landscape changed")
    module = ast.parse((args.assets / "source/nuclease_design/cnn_model.py").read_text())
    classes = next(
        ast.literal_eval(n.value)
        for n in module.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OUTPUT_CLASSES" for t in n.targets)
    )
    notebook = json.loads((args.assets / "source/notebooks/analyze_cnn.ipynb").read_text())
    text = "\n".join("".join(x.get("data", {}).get("text/plain", [])) for x in notebook["cells"][19]["outputs"])
    expected_table = {}
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) == 5 and tokens[0] in LABELS:
            expected_table[tokens[0]] = list(map(int, tokens[1:]))
    require(set(expected_table) == set(LABELS), "Notebook categories incomplete")
    cohorts, arrays, rows, landscapes = {}, {}, [], {}
    for name, source_path in [("gcs", args.assets / "gcs/processed_data/landscape.csv"), ("git", args.git_landscape)]:
        bound = next(x for x in report["cohorts"] if x["source"] == name)
        for filename, expected in bound["outputs"].items():
            require(sha(args.run / name / filename) == expected, "Saved inference artifact changed")
        original = frame(source_path)
        landscapes[name] = original
        expected_cohort = original[original.generations != ("g4",)].reset_index(drop=True)
        observed_cohort = frame(args.run / name / "cohort.csv")
        pd.testing.assert_frame_equal(observed_cohort, expected_cohort, check_dtype=False)
        cohorts[name] = observed_cohort
        with np.load(args.run / name / "predictions.npz", allow_pickle=False) as saved:
            individual = saved["individual"].astype(np.float64)
            require(individual.shape == (len(observed_cohort), len(classes), 5), "Inference dimensions changed")
            require(
                np.isfinite(individual).all() and (individual >= 0).all() and (individual <= 1).all(),
                "Invalid model probability",
            )
            require(np.allclose(individual.sum(axis=1), 1, rtol=1e-6, atol=1e-6), "Per-model mass invalid")
            # Independent double-precision geometric mean and normalizing constants.
            geometric = np.prod(individual + 1e-6, axis=-1) ** (1 / individual.shape[-1])
            global_prob = geometric / geometric.sum()
            row_prob = geometric / geometric.sum(axis=1, keepdims=True)
            require(
                np.allclose(saved["author"], global_prob, rtol=1e-5, atol=1e-10), "Author ensemble aggregation changed"
            )
            require(
                np.allclose(saved["per_sequence"], row_prob, rtol=1e-5, atol=1e-7),
                "Row normalization sensitivity changed",
            )
            indices = geometric.argmax(axis=1)
            require(np.array_equal(indices, saved["author"].argmax(axis=1)), "Saved predicted classes changed")
            arrays[name] = saved["individual"].copy()
            table = {
                label: [
                    int(((observed_cohort.activity_level == label).to_numpy() & (indices == i)).sum())
                    for i in range(len(classes))
                ]
                for label in LABELS
            }
            require(sum(map(sum, table.values())) == len(observed_cohort), "Cross-table dropped rows")
            require(table == expected_table == bound["cross_tabulation"], "Notebook cross-table differs")
            rows.append(
                {
                    "source": name,
                    "rows": len(observed_cohort),
                    "cohort_reconstructed": True,
                    "aggregation_reconstructed": True,
                    "cross_table": table,
                    "full_landscape_rows": len(original),
                    "source_sha256": sha(source_path),
                    "max_author_error": float(np.abs(saved["author"] - global_prob).max()),
                    "max_per_sequence_error": float(np.abs(saved["per_sequence"] - row_prob).max()),
                }
            )
    pd.testing.assert_frame_equal(
        cohorts["gcs"].set_index("sequence").sort_index(),
        cohorts["git"].set_index("sequence").sort_index(),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        landscapes["gcs"].set_index("sequence").sort_index(),
        landscapes["git"].set_index("sequence").sort_index(),
        check_dtype=False,
    )
    order = pd.Series(range(len(cohorts["git"])), index=cohorts["git"].sequence).loc[cohorts["gcs"].sequence].to_numpy()
    require(
        np.array_equal(arrays["gcs"], arrays["git"][order]), "Per-sequence predictions differ across source versions"
    )
    result = {
        "status": "audit_completed",
        "rows": rows,
        "source_versions_semantically_identical": True,
        "prediction_versions_identical": True,
        "full_landscape_rows_compared": len(landscapes["gcs"]),
        "serialization_metadata_excluded": ["Unnamed: 0"],
        "run_report_sha256": sha(args.run / "report.json"),
        "protocol_sha256": sha(args.run / "protocol.json"),
        "auditor_sha256": sha(Path(__file__)),
        "scope": "Independent source/cohort/hash/aggregation/class-count audit; no new forward run, retraining, efficacy replicate or agent orchestration",
        "normalization_note": "Pinned author ensemble normalizes over the whole batch. Row normalization preserves argmax but changes probability mass; retained as sensitivity only.",
        "label_note": "Final FDR-derived categories and historical CNN classes differ in meaning; this is a cross-tabulation, not classification accuracy.",
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "auditor.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": result["status"], "cohort_rows_each": len(cohorts["gcs"]), "versions_identical": True}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--git-landscape", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    audit(p.parse_args())
