"""Isolated, version-pinned ProtPredict amino-acid XGBoost comparison worker.

Run with the separately installed Python 3.10 scientific environment. This is a
traditional comparator, not a generated Scientist model. Inputs contain training
labels and optional unlabeled prediction rows; no outer labels are accepted.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

SCHEMA = "protpredict_xgb_request.v1"
VERSIONS = {
    "numpy": "1.22.3",
    "scipy": "1.7.3",
    "scikit-learn": "1.0.2",
    "pandas": "1.4.2",
    "xgboost": "1.6.1",
    "scikit-optimize": "0.9.0",
    "joblib": "1.1.0",
    "threadpoolctl": "2.2.0",
}
BOUNDS = {
    "gamma": (0.0, 1.0),
    "max_depth": (2, 12),
    "subsample": (0.4, 1.0),
    "learning_rate": (0.005, 0.3, "log-uniform"),
    "n_estimators": (10, 1000),
    "colsample_bytree": (0.4, 1.0),
    "reg_lambda": (0.0, 1.0),
    "reg_alpha": (0.0, 0.5),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def matrix(value, minimum, maximum):
    require(isinstance(value, list) and minimum <= len(value) <= maximum, "Invalid number of rows")
    require(all(isinstance(row, list) and len(row) == 19 for row in value), "All 19 measured amino acids required")
    require(all(finite_number(x) and 0 <= x <= 1000 for row in value for x in row), "Invalid amino-acid concentration")


def validate_request(request):
    require(isinstance(request, dict) and request.get("schema") == SCHEMA, "Unsupported worker request")
    action = request.get("action")
    require(action in {"tune", "fit_predict"}, "Unsupported worker action")
    fields = {"schema", "action", "feature_names", "training"}
    fields |= {"search"} if action == "tune" else {"parameters", "scoring"}
    require(set(request) == fields, "Unexpected request fields; outer target labels are forbidden")
    names = request["feature_names"]
    require(
        isinstance(names, list) and len(names) == 19 and all(isinstance(x, str) and x.strip() for x in names),
        "Invalid feature names",
    )
    require(len(set(names)) == 19, "Repeated feature names")
    training = request["training"]
    require(isinstance(training, dict) and set(training) == {"x", "y", "groups"}, "Invalid training fields")
    matrix(training["x"], 40, 600)
    n = len(training["x"])
    require(
        isinstance(training["y"], list)
        and len(training["y"]) == n
        and all(finite_number(y) and y >= 0 for y in training["y"]),
        "Invalid measured training abundance",
    )
    groups = training["groups"]
    require(
        isinstance(groups, list) and len(groups) == n and all(isinstance(x, str) and x.strip() for x in groups),
        "Invalid deleted-genotype groups",
    )
    require(len(set(groups)) == n, "Author KFold control requires one row per distinct training genotype")
    if action == "tune":
        search = request["search"]
        require(isinstance(search, dict) and set(search) == {"iterations", "folds"}, "Explicit bounded search required")
        require(type(search["iterations"]) is int and 2 <= search["iterations"] <= 50, "Search limit is 2..50")
        require(type(search["folds"]) is int and 2 <= search["folds"] <= 5, "Fold limit is 2..5")
    else:
        params = request["parameters"]
        require(isinstance(params, dict) and set(params) == set(BOUNDS), "Use the eight source-declared parameters")
        for name, bounds in BOUNDS.items():
            value = params[name]
            require(finite_number(value) and bounds[0] <= value <= bounds[1], "Parameter outside author search space")
            if name in {"max_depth", "n_estimators"}:
                require(type(value) is int, "Integer tree parameters required")
        scoring = request["scoring"]
        require(isinstance(scoring, dict) and set(scoring) == {"x", "ids"}, "Scoring target labels are forbidden")
        matrix(scoring["x"], 1, 200)
        ids = scoring["ids"]
        require(
            isinstance(ids, list) and len(ids) == len(scoring["x"]) and all(isinstance(x, str) and x for x in ids),
            "Invalid scoring identities",
        )
        require(len(set(ids)) == len(ids), "Repeated scoring identity")
    return request


def verify_runtime():
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(
        versions == VERSIONS and platform.python_version_tuple()[:2] == ("3", "10"),
        "Use the pinned isolated Python 3.10 runtime",
    )
    return {"python": platform.python_version(), "packages": versions}


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_columns(x):
    """Author preprocessing removes exact duplicate columns, preserving first occurrence."""
    import numpy as np

    selected = []
    for i in range(x.shape[1]):
        if not any(np.array_equal(x[:, i], x[:, j]) for j in selected):
            selected.append(i)
    return selected


def python_parameters(parameters):
    return {
        name: int(value) if name in {"max_depth", "n_estimators"} else float(value)
        for name, value in parameters.items()
    }


def execute(request_path, output):
    request_path, output = Path(request_path).absolute(), Path(output).absolute()
    require(
        not any(p.is_symlink() for p in [request_path, *request_path.parents, output, *output.parents]),
        "Symlinked worker path",
    )
    require(not output.exists(), "Worker output already exists")
    before = file_hash(request_path)
    request = validate_request(json.loads(request_path.read_text()))
    runtime = verify_runtime()
    import numpy as np
    from sklearn.model_selection import KFold
    from skopt import BayesSearchCV
    from xgboost import XGBRegressor

    x = np.asarray(request["training"]["x"], dtype=np.float64)
    y = np.asarray(request["training"]["y"], dtype=np.float64)
    selected = unique_columns(x)
    x = x[:, selected]
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "protpredict_xgb_result.v1",
        "status": "running",
        "action": request["action"],
        "request_sha256": before,
        "worker_sha256": file_hash(Path(__file__)),
        "runtime": runtime,
        "training_rows": len(y),
        "feature_names": request["feature_names"],
        "selected_feature_indices": selected,
        "preprocessing": "Exact duplicate feature removal fitted to the complete training set; no scoring rows used",
        "fit_partition": "training",
        "outer_labels_received": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    write(output / "report.json", result)
    try:
        if request["action"] == "tune":
            search = request["search"]
            estimator = XGBRegressor(random_state=0, n_jobs=1)
            optimizer = BayesSearchCV(
                estimator,
                BOUNDS,
                n_iter=search["iterations"],
                scoring="neg_mean_squared_error",
                cv=KFold(search["folds"], shuffle=True, random_state=42),
                random_state=0,
                n_jobs=1,
                refit=True,
            )

            def checkpoint(res):
                write(
                    output / "progress.json",
                    {
                        "iterations_completed": len(res.x_iters),
                        "best_training_cv_mse": float(min(res.func_vals)),
                        "external_model_calls": 0,
                    },
                )

            optimizer.fit(x, y, callback=checkpoint)
            params = python_parameters(optimizer.best_params_)
            trace = []
            for i, candidate in enumerate(optimizer.cv_results_["params"]):
                trace.append(
                    {
                        "iteration": i + 1,
                        "parameters": python_parameters(candidate),
                        "mean_training_cv_mse": -float(optimizer.cv_results_["mean_test_score"][i]),
                        "fold_training_mse": [
                            -float(optimizer.cv_results_[f"split{fold}_test_score"][i])
                            for fold in range(search["folds"])
                        ],
                    }
                )
            write(output / "search_trace.json", trace)
            write(output / "parameters.json", params)
            result.update(
                search=search,
                inner_split_seed=42,
                optimizer_seed=0,
                estimator_seed=0,
                threads=1,
                best_training_cv_mse=-float(optimizer.best_score_),
                parameters={"path": "parameters.json", "sha256": file_hash(output / "parameters.json")},
                search_trace={"path": "search_trace.json", "sha256": file_hash(output / "search_trace.json")},
            )
        else:
            params = request["parameters"]
            estimator = XGBRegressor(random_state=0, n_jobs=1, **params)
            estimator.fit(x, y)
            estimator.save_model(str(output / "model.json"))
            scoring = np.asarray(request["scoring"]["x"], dtype=np.float64)[:, selected]
            predictions = estimator.predict(scoring)
            require(np.isfinite(predictions).all(), "Non-finite comparator prediction")
            rows = [
                {"id": identity, "prediction": float(value)}
                for identity, value in zip(request["scoring"]["ids"], predictions, strict=True)
            ]
            write(output / "predictions.json", rows)
            write(
                output / "predictor.json",
                {
                    "feature_names": request["feature_names"],
                    "selected_feature_indices": selected,
                    "parameters": params,
                    "random_state": 0,
                    "n_jobs": 1,
                    "runtime": runtime,
                },
            )
            result.update(
                model={"path": "model.json", "sha256": file_hash(output / "model.json")},
                predictor={"path": "predictor.json", "sha256": file_hash(output / "predictor.json")},
                predictions={"path": "predictions.json", "sha256": file_hash(output / "predictions.json")},
                scoring_rows=len(rows),
            )
        require(file_hash(request_path) == before, "Worker input changed during computation")
        result["status"] = "completed"
        write(output / "report.json", result)
        return result
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        write(output / "report.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.request, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["status", "action", "training_rows", "external_model_calls", "new_autonomous_models"]
            }
        )
    )


if __name__ == "__main__":
    main()
