"""Fit agent-authored algorithms in a training-only sandbox and measure revisions."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import time
import uuid

from .contracts import digest, file_hash
from .host import write_json
from .research_development import STRATEGIES, evaluation_digest, immutable_copy, require, validate_data
from .tool_diagnostics import python_failure
from utils.scoped_files import read_bytes, write_bytes

from .research_model_state import RUNTIME_SUFFIX, training_script, read_fitted_state


def _unimplemented(statement):
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr):
        return isinstance(statement.value, ast.Constant) and statement.value.value in (None, Ellipsis)
    if isinstance(statement, ast.Return):
        return statement.value is None or (
            isinstance(statement.value, ast.Constant) and statement.value.value in (None, Ellipsis))
    if isinstance(statement, ast.Raise):
        error = statement.exc.func if isinstance(statement.exc, ast.Call) else statement.exc
        return isinstance(error, ast.Name) and error.id == "NotImplementedError"
    return False


def validate_algorithm(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(
            f"Model source is not valid Python: {exc.msg} "
            f"at line {exc.lineno}, column {exc.offset}. "
            "Repair the saved .py source before training."
        ) from None
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("fit", "predict"):
        node = functions.get(name)
        require(node is not None and len(node.args.posonlyargs)+len(node.args.args) == 2
                and not node.args.vararg and not node.args.kwarg and not node.args.kwonlyargs,
                "Implement fit(training, seed) -> fitted parameter dict or estimator; predict(model, inputs) -> prediction dict. "
                "These must be module-level functions, not class methods. A helper class is allowed behind them.")
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
        require(body and not all(_unimplemented(statement) for statement in body),
                f"{name} is an unfinished placeholder. Implement its calculations before training.")
    fit = functions["fit"]
    training_name = (fit.args.posonlyargs + fit.args.args)[0].arg
    require(any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == training_name
                for statement in fit.body for node in ast.walk(statement)),
            "fit does not read its training argument. Fit parameters from the supplied training['train'] rows "
            "and return the fitted model; fixed demo values or a seed alone are not training. "
            "Existing pretrained or fixed-rule baselines use evaluate_research_model.")
    require("run" not in functions, "The host supplies run(inputs); implement fit and predict")
    require(not any(isinstance(n, ast.Name) and n.id.startswith("_vf_") for n in ast.walk(tree)),
            "Names starting _vf_ are reserved for the prediction interface")
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


async def fit_model(workspace, development, runner, *, name, source_file, evaluation_file,
                    hypothesis, strategy="supervised", parent_experiment_id="",
                    variant="auto", prediction_key="value", seed=0, on_progress=None):
    require(isinstance(source_file, str) and Path(source_file).suffix.lower() == ".py",
            "source_file must be the saved .py implementation with fit(training, seed) and predict(model, inputs). "
            "Pass the dataset JSON as evaluation_file; data files cannot be executed as model source.")
    require(source_file != evaluation_file, "source_file and evaluation_file must refer to different files")
    require(strategy in STRATEGIES, "Use a declared training strategy")
    require(type(seed) is int and 0 <= seed < 2**32, "Seed must be an unsigned 32-bit integer")
    if variant == "auto":
        variant = "candidate" if parent_experiment_id else "baseline"
    require(variant in {"baseline", "candidate", "ablation"}, "Invalid experiment variant")
    parent = development.record(parent_experiment_id) if parent_experiment_id else None
    require(parent is not None or variant == "baseline", "Start with variant=baseline")
    require(len(list((development.root/"training").glob("*/receipt.json"))) < 64,
            "Training attempt budget reached")
    if evaluation_file:
        raw = development.load_evaluation(workspace, evaluation_file)
        data = validate_data(json.loads(raw), strategy)
    else:
        require(parent is not None, "The first baseline requires an evaluation_file")
        _, raw, data = development.evaluation_snapshot(parent_experiment_id)
        data = validate_data(data, strategy)
    data_hash = evaluation_digest(data)
    if parent:
        require(parent["evaluation_sha256"] == data_hash, "Keep parent data and splits unchanged")
    code = read_bytes(workspace, source_file, max_bytes=200000).decode("utf-8")
    code_hash = validate_algorithm(code)
    require(isinstance(data.get("unlabeled", []), list), "Unlabeled observations must be a list")
    if data.get("unlabeled"):
        validate_data(data, "semi_supervised")
    training = {"task": data["task"], "strategy": strategy, "train": data["splits"]["train"],
                "unlabeled": data.get("unlabeled", [])}
    identity = uuid.uuid4().hex
    folder = development.root/"training"/identity
    work = folder/"workspace";work.mkdir(parents=True)
    inputs = {"runtime.py": (code+"\n"+RUNTIME_SUFFIX).encode(),
              "train.py": training_script(seed).encode(),
              "training.json": json.dumps(training, sort_keys=True, allow_nan=False).encode()}
    for path, content in inputs.items():
        write_bytes(work, path, content)
    record = {"id": identity, "status": "running", "created": time.time(), "seed": seed,
              "algorithm_sha256": hashlib.sha256(code.encode()).hexdigest(),
              "algorithm_ast_sha256": code_hash, "evaluation_sha256": data_hash,
              "training_sha256": digest(training), "training_rows": len(training["train"]),
              "validation_and_test_mounted": False, "source_file": source_file,
              "evaluation_visibility": "host_only" if development.data_store else "agent_workspace",
              "parent_experiment_id": parent_experiment_id or None}
    write_json(folder/"receipt.json", record)
    try:
        kwargs = {"on_progress": on_progress} if on_progress is not None else {}
        execution = await runner(work, folder/"execution", "train.py", 120, **kwargs)
        record["execution"] = execution
        require(execution["exit_code"] == 0, "Training failed: "+python_failure(
                execution.get("stderr_tail") or execution.get("stderr") or "inspect the training log", work, 800,
                source_file=source_file, source_text=code))
        require(all(read_bytes(work, path) == content for path, content in inputs.items()),
                "Training modified its source or input data")
        state_metadata, state_files, state_hash = read_fitted_state(work)
        if parent:
            for prior in (development.root/"training").glob("*/receipt.json"):
                previous = json.loads(prior.read_text())
                if previous.get("experiment_id") == parent_experiment_id:
                    require((previous.get("algorithm_ast_sha256"),previous.get("state_sha256"))
                            != (code_hash,state_hash),
                            "A revision must change the algorithm or fitted state, not only names or comments")

        prefix = "models/"+identity+"/"
        exported = {**inputs, **state_files}
        fitted_paths = [prefix+path for path in state_files]
        for path, content in exported.items():
            immutable_copy(workspace, prefix+path, content)
        development.write_evaluation(workspace, prefix+"evaluation.json",
                                     json.dumps(data, sort_keys=True, allow_nan=False).encode())
        experiment = await development.evaluate(
            workspace, runner, name=name, source=prefix+"runtime.py", files=fitted_paths,
            evaluation_file=prefix+"evaluation.json", prediction_key=prediction_key, hypothesis=hypothesis,
            strategy=strategy, parent_experiment_id=parent_experiment_id, variant=variant)
        record.update(status="completed", state_sha256=state_hash, experiment_id=experiment["id"],
                      runtime_files=[prefix+"runtime.py", *fitted_paths], state_format=state_metadata["format"],
                      dependency_versions=state_metadata["dependencies"], python_version=state_metadata["python"],
                      evidence_files=[prefix+"train.py",prefix+"training.json",prefix+"evaluation.json"],
                      reproduction_script=prefix+"train.py", fitted_files=fitted_paths,
                      evaluation_file=prefix+"evaluation.json", source=prefix+"runtime.py")
        record["receipt_sha256"] = digest(record)
        write_json(folder/"receipt.json", record)
        write_bytes(workspace, "training/"+identity+".json", json.dumps(record, indent=2).encode())
        next_step = "Inspect host validation, revise the algorithm with parent_experiment_id, "
        next_step += "then submit the selected generation for independent Review. No model is published by training."
        diagnostics = experiment["validation"]["diagnostics"]
        if diagnostics["constant_predictions"]:
            behavior = "Validation predictions are constant"
            if diagnostics["matches_training_reference"]:
                behavior += " and match the training reference"
            next_step = (behavior + ". Check feature variation and learned parameters using training data; "
                         "measure a justified revision on the same split and explain any retained constant baseline. " + next_step)
        return {"training": record, "experiment": experiment, "next_step": next_step}
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        write_json(folder/"receipt.json", record)
        raise
