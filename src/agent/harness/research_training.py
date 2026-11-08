"""Fit agent-authored algorithms in a training-only sandbox and measure revisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

from .contracts import digest, file_hash
from .host import write_json
from .research_development import STRATEGIES, evaluation_digest, immutable_copy, require, validate_data
from .tool_diagnostics import python_failure, bounded_message
from .research_algorithm_checks import algorithm_diagnostics, checked_algorithm_digest, validate_algorithm as validate_algorithm
from utils.scoped_files import read_bytes, write_bytes

from .research_model_state import RUNTIME_SUFFIX, training_script, read_fitted_state
from .research_attempt_resolution import VERSION as ATTEMPT_PROTOCOL, TrainingAttemptError
from .research_execution_feedback import _key as feedback_key


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
    if parent is None and variant != "baseline":
        previous = development.records()[-3:]
        if previous:
            identities = ", ".join(row["id"] for row in previous)
            raise ValueError(
                "A candidate or ablation needs parent_experiment_id from a measured experiment on the same split. "
                f"Recent experiment IDs: {identities}. "
                "Select the parent you are improving; do not restart it as an unlinked baseline.")
        raise ValueError("Measure the first model with variant=baseline, then use its returned experiment.id "
                         "as parent_experiment_id for a candidate or ablation.")
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
    require(isinstance(data.get("unlabeled", []), list), "Unlabeled observations must be a list")
    if data.get("unlabeled"):
        validate_data(data, "semi_supervised")
    training = {"task": data["task"], "strategy": strategy, "train": data["splits"]["train"],
                "unlabeled": data.get("unlabeled", [])}
    identity = uuid.uuid4().hex
    folder = development.root/"training"/identity
    work = folder/"workspace";work.mkdir(parents=True)
    write_bytes(folder, "algorithm.py", code.encode("utf-8"))
    inputs = {"runtime.py": (code+"\n"+RUNTIME_SUFFIX).encode(),
              "train.py": training_script(seed).encode(),
              "training.json": json.dumps(training, sort_keys=True, allow_nan=False).encode()}
    for path, content in inputs.items():
        write_bytes(work, path, content)
    record = {"id": identity, "attempt_protocol": ATTEMPT_PROTOCOL,
              "feedback_key": feedback_key("train_research_model", {
                  "source_file": source_file, "parent_experiment_id": parent_experiment_id, "variant": variant}),
              "variant": variant, "prediction_key": prediction_key, "status": "running", "created": time.time(), "seed": seed,
              "algorithm_sha256": hashlib.sha256(code.encode()).hexdigest(),
              "source_snapshot": "algorithm.py", "stage": "preflight", "evaluation_sha256": data_hash,
              "training_sha256": digest(training), "training_rows": len(training["train"]),
              "validation_and_test_mounted": False, "source_file": source_file,
              "evaluation_visibility": "host_only" if development.data_store else "agent_workspace",
              "parent_experiment_id": parent_experiment_id or None}
    write_json(folder/"receipt.json", record)
    try:
        record["source_check"] = algorithm_diagnostics(code)
        code_hash = checked_algorithm_digest(record["source_check"])
        record.update(algorithm_ast_sha256=code_hash, stage="fitting")
        write_json(folder/"receipt.json", record)
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
        record["stage"] = "evaluation"
        write_json(folder/"receipt.json", record)
        experiment = await development.evaluate(
            workspace, runner, name=name, source=prefix+"runtime.py", files=fitted_paths,
            evaluation_file=prefix+"evaluation.json", prediction_key=prediction_key, hypothesis=hypothesis,
            strategy=strategy, parent_experiment_id=parent_experiment_id, variant=variant)
        record.update(status="completed", stage="completed", finished=time.time(), state_sha256=state_hash, experiment_id=experiment["id"],
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
        record.update(status="failed", finished=time.time(), error_type=type(exc).__name__,
                      error=bounded_message(str(exc), work, 1600))
        record["receipt_sha256"] = digest(record)
        write_json(folder/"receipt.json", record)
        if isinstance(exc, Exception):
            raise TrainingAttemptError(identity, bounded_message(str(exc), work, 1000)) from exc
        raise
