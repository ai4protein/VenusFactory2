"""Conventional expression controls using only an inner training-population split."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr

from agent.rsi import context_controls, generated_context
from .context_controls import score_rows
from .generated_context import development_split
from agent.rsi.datasets.model_tasks import bound_file, ordinary
from agent.rsi.datasets.research_project_catalog import identifier
from .storage import atomic_json, sha256_file


def controls(training, scoring, scope):
    if scope["assay"] != "population_protein_expression" or [c["name"] for c in scope["covariates"]] != [
        "author_log2_tpm",
        "od_rna",
        "od_protein",
        "ploidy",
    ]:
        raise ValueError("Declared yeast expression covariates required")
    outcomes = score_rows(training, scoring, scope, timings=False)
    x = np.asarray([r["values"] for r in training], dtype=float)
    query = np.asarray([r["values"] for r in scoring], dtype=float)
    y = np.asarray([r["target"] for r in training], dtype=float)
    truth = np.asarray([r["target"] for r in scoring], dtype=float)
    null_mse = float(np.mean((truth - y.mean()) ** 2))
    for method, columns in [("transcript_only_ols", [0]), ("culture_metadata_ols", [1, 2, 3])]:
        design = np.column_stack([np.ones(len(x)), x[:, columns]])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = np.column_stack([np.ones(len(query)), query[:, columns]]) @ coefficients
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Nonfinite ordinary-least-squares control")
        mse = float(np.mean((truth - prediction) ** 2))
        correlation = float(spearmanr(truth, prediction).statistic) if np.ptp(truth) and np.ptp(prediction) else None
        outcomes.append(
            {
                "method": method,
                "status": "completed",
                "metrics": {
                    "rmse": math.sqrt(mse),
                    "mae": float(np.mean(np.abs(truth - prediction))),
                    "spearman": correlation,
                    "mse_skill_vs_training_mean": 1 - mse / null_mse if null_mse else None,
                },
                "predictions": [
                    {"id": row["id"], "prediction": float(p)} for row, p in zip(scoring, prediction, strict=True)
                ],
            }
        )
    return outcomes


def verify_metrics(training, scoring, outcomes):
    """Check saved error metrics by scalar arithmetic, including the null predictor."""
    location = math.fsum(r["target"] for r in training) / len(training)
    null_mse = math.fsum((r["target"] - location) ** 2 for r in scoring) / len(scoring)
    for outcome in outcomes:
        if outcome["status"] != "completed":
            continue
        predicted = outcome["predictions"]
        if [p["id"] for p in predicted] != [r["id"] for r in scoring]:
            raise ValueError("Control prediction identities differ")
        errors = [r["target"] - p["prediction"] for r, p in zip(scoring, predicted, strict=True)]
        mse = math.fsum(error**2 for error in errors) / len(errors)
        expected = {"rmse": math.sqrt(mse), "mae": math.fsum(abs(e) for e in errors) / len(errors)}
        if null_mse:
            expected["mse_skill_vs_training_mean"] = 1 - mse / null_mse
        if any(
            not math.isclose(outcome["metrics"][key], value, rel_tol=1e-10, abs_tol=1e-12)
            for key, value in expected.items()
        ):
            raise ValueError("Control numerical metrics differ from independent arithmetic")


def run_catalog(catalog_path, output, progress=None):
    catalog_path, output = Path(catalog_path).absolute(), Path(output).absolute()
    catalog = json.loads(ordinary(catalog_path).read_text())
    entries = [e for e in catalog["entries"] if e["status"] == "admitted"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "yeast_population_training_controls.v1",
        "status": "running",
        "catalog_sha256": sha256_file(catalog_path),
        "tasks": len(entries),
        "completed_tasks": 0,
        "data_boundary": "training_only",
        "outer_validation_scored": False,
        "future_scored": False,
        "new_trained_autonomous_models": 0,
        "external_model_calls": 0,
        "implementation_sha256": sha256_file(Path(__file__)),
        "numeric_dependencies": {
            "context_controls.py": sha256_file(Path(context_controls.__file__)),
            "generated_context.py": sha256_file(Path(generated_context.__file__)),
        },
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    atomic_json(output / "report.json", report)
    counts = Counter()
    try:
        for entry in entries:
            identifier(entry["id"])
            task = json.loads(bound_file(entry["task"]).read_text())
            training = json.loads(bound_file(task["data"]["training"]).read_text())
            fit, scoring = development_split(training)
            outcomes = controls(fit, scoring, task["scope"])
            verify_metrics(fit, scoring, outcomes)
            atomic_json(
                output / "tasks" / (entry["id"] + ".json"),
                {
                    "task": entry["task"],
                    "training": task["data"]["training"],
                    "fit_rows": len(fit),
                    "score_rows": len(scoring),
                    "fit_groups": sorted({r["group"] for r in fit}),
                    "score_groups": sorted({r["group"] for r in scoring}),
                    "outcomes": outcomes,
                    "metric_arithmetic_verified": True,
                },
            )
            counts.update(c["status"] for c in outcomes)
            report.update(completed_tasks=report["completed_tasks"] + 1, controls=dict(counts))
            if report["completed_tasks"] % 25 == 0:
                atomic_json(output / "report.json", report)
                if progress:
                    progress(dict(report))
        report["status"] = "completed"
        atomic_json(output / "report.json", report)
        if progress:
            progress(dict(report))
        return report
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(output / "report.json", report)
        raise
