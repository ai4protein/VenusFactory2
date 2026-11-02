"""Complete CLEAN training as a fixed, source-bound reproduction capability.

Author routines remain in separately licensed CLEAN research assets. Reusing
this capability means fresh execution, not copying prior scientific outputs.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from agent.rsi import clean_training_evaluation as evaluation
from agent.rsi import clean_training_inputs as inputs

PLUGIN = "clean.training.v1"
SCOPE = "clean_full_split10_sequence_embeddings_author_training_and_evaluation"
ORIGIN = "new_sequence_embeddings_author_training_and_evaluation"
JOB_TIMEOUT = 259200
VERIFY_TIMEOUT = 7200
STAGES = ("inputs", "triplet", "supconH", "evaluation")
TIMEOUTS = {"inputs": 172800, "triplet": 21600, "supconH": 43200, "evaluation": 3600}
SOURCES = (
    "src/agent/rsi/cases/clean_training_plugin.py",
    "src/agent/rsi/cases/clean_training_inputs.py",
    "src/agent/rsi/cases/clean_author_training.py",
    "src/agent/rsi/cases/clean_training_evaluation.py",
    "src/agent/rsi/cases/clean_annotation.py",
    "scripts/rsi/cases/run_clean_training_plugin.py",
    "scripts/rsi/cases/run_clean_training_inputs.py",
    "scripts/rsi/cases/run_clean_author_training.py",
    "scripts/rsi/cases/run_clean_training_evaluation.py",
)
ALLOCATION = {
    "input_records": inputs.TOTAL,
    "input_counts": inputs.COUNTS,
    "schedules": evaluation.SCHEDULES,
    "evaluation": evaluation.ALLOCATION,
}
LIMITS = {**evaluation.LIMITS, "wet_lab_validation": False}


def read(path):
    return json.loads(Path(path).read_text())


def output_path(root, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    base = root / "research_runs/rsi"
    output.relative_to(base)
    if output == base or output.resolve() != output:
        raise ValueError("A dedicated, non-symlinked research output is required")
    return output


def runtime(root):
    return str(Path(root) / "research_runs/rsi/envs/clean/bin/python")


def environment(root, output):
    # Direct invocations get the same credential-free environment as broker jobs.
    return {
        "PATH": str(Path(runtime(root)).parent) + ":/usr/bin:/bin",
        "HOME": str(Path(output) / "job_home"),
        "PYTHONPATH": str(Path(root) / "src"),
        "PYTHONHASHSEED": "1234",
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
    }


def commands(root, output):
    root, output = Path(root).resolve(), output_path(root, output)
    scripts = root / "scripts/rsi"
    common = ["--workspace-root", str(root)]
    values = {
        "inputs": [
            runtime(root),
            "-u",
            str(scripts / "run_clean_training_inputs.py"),
            *common,
            "--output",
            str(output / "inputs"),
            "--wall-seconds",
            str(TIMEOUTS["inputs"]),
        ]
    }
    for name in ("triplet", "supconH"):
        values[name] = [
            runtime(root),
            "-u",
            str(scripts / "run_clean_author_training.py"),
            *common,
            "--inputs",
            str(output / "inputs"),
            "--input-protocol-sha256",
            evaluation.INPUT_PROTOCOL,
            "--output",
            str(output / "training" / name),
            "--algorithm",
            name,
        ]
    values["evaluation"] = [
        runtime(root),
        "-u",
        str(scripts / "run_clean_training_evaluation.py"),
        *common,
        "--inputs",
        str(output / "inputs"),
        "--training",
        str(output / "training"),
        "--output",
        str(output / "evaluation"),
    ]
    return values


def protocol(root, output):
    return {
        "schema": 1,
        "plugin": PLUGIN,
        "parameters": {},
        "allocation": ALLOCATION,
        "implementation": {p: inputs.sha(inputs.ordinary(root, p)) for p in SOURCES},
        "environment": inputs.ENVIRONMENT,
        "input_protocol_sha256": evaluation.INPUT_PROTOCOL,
        "commands": commands(root, output),
        "stage_timeouts_seconds": TIMEOUTS,
        "job_timeout_seconds": JOB_TIMEOUT,
        "verification_timeout_seconds": VERIFY_TIMEOUT,
        "seed": 1234,
        "input_policy": "Every invocation computes all 22525 representations; no resume or existing-output option",
        "limits": LIMITS,
    }


def artifact_names():
    return {
        "protocol.json",
        "audit.json",
        *("implementation/" + p for p in SOURCES),
        *("inputs/" + n for n in ["report.json", "plan.json", "protocol.json", "attempts.jsonl"]),
        *("training/" + a + "/" + n for a in evaluation.SCHEDULES for n in ["report.json", "epochs.jsonl"]),
        *("evaluation/" + n for n in ["report.json", "protocol.json", "independent_audit.json"]),
        *(name + ".log" for name in STAGES),
    }


def canonical_training(report):
    if (
        report.get("status") != "completed"
        or report.get("stage") != "audited"
        or report.get("plugin") != PLUGIN
        or report.get("parameters") != {}
        or report.get("execution_origin") != ORIGIN
        or report.get("allocation") != ALLOCATION
        or type(report.get("external_model_calls")) is not int
        or report["external_model_calls"] != 0
        or any(report.get(k) is not v for k, v in LIMITS.items())
    ):
        raise ValueError("Incomplete or overclaimed CLEAN training report")
    audit = report.get("audit_evidence", {})
    keys = {a + "." + c for a in evaluation.SCHEDULES for c in evaluation.COHORTS}
    if (
        audit.get("status") != "passed"
        or set(audit.get("training", {})) != set(evaluation.SCHEDULES)
        or set(audit.get("allocations", {})) != keys
        or audit.get("limits") != evaluation.LIMITS
    ):
        raise ValueError("Incomplete CLEAN independent audit")
    for a, schedule in evaluation.SCHEDULES.items():
        trained = audit["training"][a]
        if trained.get("status") != "passed" or trained.get("epochs") != schedule["epochs"]:
            raise ValueError("Incomplete author training schedule")
    metrics = {}
    for key in sorted(keys):
        result = audit["allocations"][key]
        cohort = key.split(".", 1)[1]
        if result.get("status") != "passed" or result.get("rows") != evaluation.COHORTS[cohort]:
            raise ValueError("Incomplete prediction allocation")
        if result.get("label_mismatches") != [] or any(
            result.get(k) is not True for k in ["rounded_distances_match", "metrics_match", "metadata_match"]
        ):
            raise ValueError("Prediction replay does not agree")
        values = result.get("full_union_metrics", {})
        metrics[key] = {}
        for name in ("micro_f1", "weighted_f1", "exact_match"):
            value = values.get(name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid audited label metric")
            metrics[key][name] = value
    return {
        "computational_scope": SCOPE,
        "execution_origin": ORIGIN,
        "training_sequences": 7757,
        "training_ec_classes": 2702,
        "augmentation_records": 14190,
        "new_embedding_records": 22525,
        "training_epochs": {k: v["epochs"] for k, v in evaluation.SCHEDULES.items()},
        "prediction_records": 1156,
        "author_split10_training_reproduced": True,
        "metrics": metrics,
        "fresh_execution_required_for_reuse": True,
        **LIMITS,
    }


def fresh_journals(output, stages):
    prepared = read(inputs.ordinary(output, "inputs/report.json"))
    if any(
        type(prepared.get(k)) is not int or prepared[k] != v
        for k, v in {
            "resumes": 0,
            "attempts_started": inputs.TOTAL,
            "completed_records": inputs.TOTAL,
        }.items()
    ):
        raise ValueError("Fresh complete representation computation required")
    plan = read(inputs.ordinary(output, "inputs/plan.json"))
    inputs.validate_plan(plan)
    events = []
    for row in plan["records"]:
        events.extend([("started", row["key"]), ("completed", row["key"])])
    start, finish = stages[0]["started_unix"], stages[0]["finished_unix"]
    previous, count = start - 5, 0
    with inputs.ordinary(output, "inputs/attempts.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            value = json.loads(line)
            at = value.get("at")
            if (
                count > len(events)
                or (value.get("event"), value.get("key")) != events[count - 1]
                or type(at) not in (int, float)
                or not math.isfinite(at)
                or not previous <= at <= finish + 5
            ):
                raise ValueError("Fresh input journal identity, ordering or time differs")
            previous = at
    if count != len(events):
        raise ValueError("Incomplete fresh input journal")
    for stage in stages[1:3]:
        for line in inputs.ordinary(output, "training/" + stage["name"] + "/epochs.jsonl").read_text().splitlines():
            at = json.loads(line).get("at")
            if (
                type(at) not in (int, float)
                or not math.isfinite(at)
                or not stage["started_unix"] - 5 <= at <= stage["finished_unix"] + 5
            ):
                raise ValueError("Training trace predates this execution")


def validate_artifacts(report, output, root):
    """Platform-neutral bindings; numerical replay runs in the pinned environment."""
    claims = canonical_training(report)
    output = output_path(root, output)
    expected = protocol(root, output)
    if read(inputs.ordinary(output, "protocol.json")) != expected or report.get("protocol_sha256") != inputs.digest(
        expected
    ):
        raise ValueError("Training protocol or current implementation changed")
    hashes = report.get("artifact_sha256", {})
    if set(hashes) != artifact_names():
        raise ValueError("Incomplete training artifact coverage")
    for name, wanted in hashes.items():
        if inputs.sha(inputs.ordinary(output, name)) != wanted:
            raise ValueError("Training artifact changed: " + name)
    for name, wanted in expected["implementation"].items():
        if hashes["implementation/" + name] != wanted:
            raise ValueError("Archived training implementation differs")
    stages = report.get("stages", [])
    if len(stages) != len(STAGES):
        raise ValueError("Incomplete pipeline stages")
    previous = 0
    for name, stage in zip(STAGES, stages):
        start, finish = stage.get("started_unix"), stage.get("finished_unix")
        if (
            stage.get("name") != name
            or stage.get("status") != "completed"
            or type(stage.get("exit_code")) is not int
            or stage["exit_code"] != 0
            or stage.get("argv") != expected["commands"][name]
            or stage.get("timeout_seconds") != TIMEOUTS[name]
            or type(start) not in (int, float)
            or type(finish) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(finish)
            or not previous <= start <= finish
        ):
            raise ValueError("Stage allocation or completion differs")
        previous = finish
    if read(inputs.ordinary(output, "audit.json")) != report["audit_evidence"]:
        raise ValueError("Numerical audit receipt changed")
    fresh_journals(output, stages)
    return claims


def verify_in_process(report, output, root):
    if {k: importlib.metadata.version(k) for k in inputs.ENVIRONMENT} != inputs.ENVIRONMENT:
        raise ValueError("Use the pinned CLEAN verification environment")
    claims = validate_artifacts(report, output, root)
    output, root = Path(output), Path(root)
    result = evaluation.verify_output(
        output / "evaluation",
        output / "inputs",
        output / "training",
        root / "research_runs/rsi/case_assets/CLEAN-training-v1",
    )
    if result != report["audit_evidence"]:
        raise ValueError("Independent numerical replay differs from the receipt")
    return {"status": "passed", "claims": claims, "audit_sha256": inputs.sha(output / "audit.json")}


def verification_request(report, output, root):
    claims = validate_artifacts(report, output, root)
    if read(inputs.ordinary(output, "report.json")) != report:
        raise ValueError("Verifier report argument differs from persisted report")
    argv = [
        runtime(root),
        "-u",
        str(Path(root) / "scripts/rsi/cases/run_clean_training_plugin.py"),
        "--output",
        str(output),
        "--verify",
    ]
    expected = {"status": "passed", "claims": claims, "audit_sha256": inputs.sha(Path(output) / "audit.json")}
    return argv, expected


def verify_artifacts(report, output, root):
    argv, expected = verification_request(report, output, root)
    result = subprocess.run(
        argv, cwd=root, env=environment(root, output), capture_output=True, text=True, timeout=VERIFY_TIMEOUT
    )
    if result.returncode or json.loads(result.stdout) != expected:
        raise ValueError("Isolated CLEAN numerical verification failed")
    return expected["claims"]


async def verify_artifacts_async(report, output, root):
    """Keep job polling responsive and terminate the verifier on cancellation."""
    argv, expected = verification_request(report, output, root)
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=root,
        env=environment(root, output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), VERIFY_TIMEOUT)
        if process.returncode or json.loads(stdout) != expected:
            raise ValueError("Isolated CLEAN numerical verification failed")
        return expected["claims"]
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()


def execute_stage(root, output, name, argv, report):
    stage = {
        "name": name,
        "argv": argv,
        "timeout_seconds": TIMEOUTS[name],
        "status": "running",
        "started_unix": time.time(),
    }
    report["stages"].append(stage)
    report["stage"] = name
    inputs.save(output / "report.json", report)
    process = None
    try:
        with (output / (name + ".log")).open("x") as log:
            # Inherit the broker's process group so cancellation also stops descendants.
            process = subprocess.Popen(
                argv, cwd=root, env=environment(root, output), stdout=log, stderr=subprocess.STDOUT
            )
            stage["pid"] = process.pid
            inputs.save(output / "report.json", report)
            stage["exit_code"] = process.wait(timeout=TIMEOUTS[name])
        if stage["exit_code"] != 0:
            raise ValueError("Training stage failed: " + name)
        stage["status"] = "completed"
    except BaseException:
        stage["status"] = "failed"
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        stage["finished_unix"] = time.time()
        inputs.save(output / "report.json", report)


def run(root, output):
    root, output = Path(root).resolve(), output_path(root, output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "job_home").mkdir()
    started = time.monotonic()
    report = {
        "status": "running",
        "stage": "prepare",
        "plugin": PLUGIN,
        "parameters": {},
        "execution_origin": ORIGIN,
        "allocation": ALLOCATION,
        "stages": [],
        "external_model_calls": 0,
        **LIMITS,
    }
    inputs.save(output / "report.json", report)
    try:
        frozen = protocol(root, output)
        inputs.save(output / "protocol.json", frozen)
        report["protocol_sha256"] = inputs.digest(frozen)
        for name in SOURCES:
            target = output / "implementation" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        for name in STAGES:
            if protocol(root, output) != frozen:
                raise ValueError("Implementation changed during execution")
            execute_stage(root, output, name, frozen["commands"][name], report)
        audit = read(inputs.ordinary(output, "evaluation/independent_audit.json"))
        inputs.save(output / "audit.json", audit)
        report["audit_evidence"] = audit
        report["artifact_sha256"] = {name: inputs.sha(inputs.ordinary(output, name)) for name in artifact_names()}
        candidate = {**report, "status": "completed", "stage": "audited"}
        report["stage"] = "independent_verification"
        inputs.save(output / "report.json", report)
        verify_in_process(candidate, output, root)
        report = candidate
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:400])
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        inputs.save(output / "report.json", report)
    return report


def public_progress(output):
    """Unverified progress is informative only; it never grants completion."""
    try:
        output = Path(output)
        report = read(inputs.ordinary(output, "report.json"))
        stage = report.get("stage")
        if stage not in {*STAGES, "prepare", "independent_verification", "audited"}:
            return {"available": False}
        result = {"available": True, "stage": stage, "unverified_progress": True}
        if stage == "inputs":
            child = read(inputs.ordinary(output, "inputs/report.json"))
            value = child.get("completed_records")
            if type(value) is int and 0 <= value <= inputs.TOTAL:
                result.update(completed_input_records=value, total_input_records=inputs.TOTAL)
        elif stage in evaluation.SCHEDULES:
            child = read(inputs.ordinary(output, "training/" + stage + "/report.json"))
            value, limit = child.get("completed_epochs"), evaluation.SCHEDULES[stage]["epochs"]
            if type(value) is int and 0 <= value <= limit:
                result.update(completed_epochs=value, total_epochs=limit)
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return {"available": False}
