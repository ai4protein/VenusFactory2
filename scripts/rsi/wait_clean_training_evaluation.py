"""Wait on the frozen CLEAN author-training dependency, then evaluate exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": stat[19], "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def live(expected):
    try:
        stat = Path(f"/proc/{expected['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[0] != "Z" and identity(expected["pid"]) == expected
    except (OSError, ProcessLookupError):
        return False


def classify_training(status, expected_identity, predecessor_alive):
    """An observation failure is retryable; an explicit terminal failure is not."""
    if status is None:
        return "observation_unavailable" if predecessor_alive else "predecessor_unavailable"
    if not isinstance(status, dict) or status.get("process") != expected_identity:
        raise ValueError("Predecessor identity changed")
    state = status.get("status")
    if state == "failed":
        raise RuntimeError("Author training failed; no automatic restart")
    if state == "completed":
        if status.get("training_completed") is not True or status.get("evaluation_completed") is not False:
            raise ValueError("Predecessor completion claims differ")
        stages = status.get("stages")
        if not isinstance(stages, list) or len(stages) != 2:
            raise ValueError("Training schedule coverage differs")
        for stage, algorithm, epochs in zip(stages, ("triplet", "supconH"), (2000, 1500)):
            if (
                stage.get("algorithm") != algorithm
                or stage.get("status") != "completed"
                or type(stage.get("epochs")) is not int
                or stage["epochs"] != epochs
                or type(stage.get("exit_code")) is not int
                or stage["exit_code"] != 0
            ):
                raise ValueError("Training stage did not finish successfully")
        return "ready"
    if state not in ("waiting_for_inputs", "running"):
        raise ValueError("Unrecognized predecessor state")
    return "waiting_for_training" if predecessor_alive else "predecessor_unavailable"


def save(path, value):
    tmp = path.with_suffix(".writing")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def check_sources(setup, allocation):
    for name, expected in allocation["dependency_files"].items():
        path = Path(name)
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("Frozen dependency changed: " + name)
    source_map = setup / "sources.json"
    if sha(source_map) != allocation["sources_sha256"]:
        raise ValueError("Evaluation source manifest changed")
    for name, expected in json.loads(source_map.read_text()).items():
        path = Path(allocation["repository"]) / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("Frozen evaluation source changed: " + name)


def main(setup):
    setup = setup.resolve()
    allocation = json.loads((setup / "allocation.json").read_text())
    state = {
        "status": "waiting_for_training",
        "process": identity(os.getpid()),
        "started_unix": time.time(),
        "external_model_calls": 0,
        "training_completed": False,
        "evaluation_completed": False,
    }
    status_path = setup / "status.json"
    save(status_path, state)
    started = time.monotonic()
    try:
        check_sources(setup, allocation)
        unavailable = 0
        observation_since = None
        while True:
            observation = None
            try:
                observation = json.loads((Path(allocation["training_setup"]) / "status.json").read_text())
            except (OSError, ValueError) as exc:
                state["last_observation_error"] = {"type": type(exc).__name__, "at": time.time()}
            phase = classify_training(
                observation, allocation["training_identity"], live(allocation["training_identity"])
            )
            state.update(status=phase, observed_unix=time.time())
            save(status_path, state)
            if phase == "ready":
                break
            unavailable = unavailable + 1 if phase == "predecessor_unavailable" else 0
            observation_since = (observation_since or time.monotonic()) if phase == "observation_unavailable" else None
            if unavailable >= 3:
                raise RuntimeError("Training predecessor absent for three observations without valid completion")
            if observation_since is not None and time.monotonic() - observation_since > 600:
                raise RuntimeError("Predecessor status unreadable for ten minutes")
            if time.monotonic() - started > allocation["wait_seconds"]:
                raise TimeoutError("Training wait budget reached")
            time.sleep(15)
        check_sources(setup, allocation)
        state.update(status="running", training_completed=True)
        save(status_path, state)
        repository = Path(allocation["repository"])
        command = [
            "/usr/bin/nice",
            "-n",
            "10",
            allocation["python"],
            "-u",
            str(repository / "scripts/rsi/run_clean_training_evaluation.py"),
            "--workspace-root",
            allocation["workspace"],
            "--inputs",
            allocation["input_output"],
            "--training",
            allocation["training_output"],
            "--output",
            allocation["output"],
        ]
        env = {
            "HOME": str(Path.home()),
            "PATH": str(Path(allocation["python"]).parent) + ":/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONHASHSEED": "1234",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with (setup / "evaluation.log").open("wb") as log:
            child = subprocess.Popen(
                command, cwd=setup, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            state.update(worker=identity(child.pid), evaluation_started_unix=time.time())
            save(status_path, state)
            try:
                code = child.wait(timeout=allocation["evaluation_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = 124
        state["exit_code"] = code
        if code != 0:
            raise RuntimeError("Evaluation failed; artifacts retained, no automatic retry")
        result = json.loads((Path(allocation["output"]) / "report.json").read_text())
        if result.get("status") != "completed" or result.get("evaluation_completed") is not True:
            raise ValueError("Evaluator exited without audited completion")
        check_sources(setup, allocation)
        state.update(
            status="completed",
            evaluation_completed=True,
            finished_unix=time.time(),
            report_sha256=sha(Path(allocation["output"]) / "report.json"),
        )
        save(status_path, state)
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc), finished_unix=time.time())
        save(status_path, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    main(parser.parse_args().setup)
