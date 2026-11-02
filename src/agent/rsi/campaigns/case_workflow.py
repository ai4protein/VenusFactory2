"""Resumable, provenance-checked stages for complete scientific case workflows.

Commands come from trusted case adapters, never free-form model text. A stage
is complete only after its domain artifact validator succeeds. Resume verifies
recipe, input and output hashes; a failed/partial stage needs a new run folder.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json, safe_child, sha256_file


@dataclass(frozen=True)
class CaseStage:
    id: str
    argv: tuple[str, ...]
    cwd: Path
    inputs: tuple[Path, ...]
    output: Path
    timeout_s: int
    validate: Callable[[Path], dict]


class CaseWorkflow:
    def __init__(self, root: Path, identity: dict):
        import json

        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "workflow.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError("workflow identity changed")
        else:
            atomic_json(path, identity)
        self.identity = identity

    def run_stage(self, stage: CaseStage):
        import json

        record_path = safe_child(self.root, stage.id + ".json")
        stage.output.resolve().relative_to(self.root)
        binding = {
            "id": stage.id,
            "argv": list(stage.argv),
            "cwd": str(stage.cwd),
            "input_hashes": {str(x): sha256_file(x) for x in stage.inputs},
            "output": str(stage.output),
            "timeout_s": stage.timeout_s,
        }
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous["binding"] != binding:
                raise ValueError("stage inputs or parameters changed")
            if previous["status"] != "completed":
                raise ValueError("failed/partial stage requires a new run directory")
            if sha256_file(stage.output) != previous["output_sha256"]:
                raise ValueError("completed stage output changed")
            stage.validate(stage.output)
            return previous
        if stage.output.exists():
            raise ValueError("unrecorded output exists; refusing to append")
        stage.output.parent.mkdir(parents=True, exist_ok=True)
        record = {"binding": binding, "status": "running", "recipe_sha256": digest(binding)}
        atomic_json(record_path, record)
        started = time.monotonic()
        # Minimal environment: no provider credentials or unrelated user env.
        env = {
            "PATH": str(Path(stage.argv[0]).parent) + ":/usr/bin:/bin",
            "HOME": str(self.root / "home"),
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONHASHSEED": "0",
        }
        Path(env["HOME"]).mkdir(exist_ok=True)
        with (self.root / (stage.id + ".log")).open("wb") as log:
            proc = None
            try:
                proc = subprocess.Popen(
                    stage.argv, cwd=stage.cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                proc.wait(timeout=stage.timeout_s)
                if proc.returncode:
                    raise RuntimeError(f"case stage exited with code {proc.returncode}")
                if {str(x): sha256_file(x) for x in stage.inputs} != binding["input_hashes"]:
                    raise ValueError("case input changed during execution")
                record["artifact_audit"] = stage.validate(stage.output)
                record.update(status="completed", output_sha256=sha256_file(stage.output))
            except Exception as exc:
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                record["wall_seconds"] = round(time.monotonic() - started, 3)
                atomic_json(record_path, record)
        return record


def audit_quiver(path: Path, expected_count: int) -> dict:
    import math

    records = {}
    current = None
    with path.open() as data:
        for line in data:
            if line.startswith("QV_TAG "):
                tag = line.split()[1]
                if tag in records:
                    raise ValueError("duplicate design id")
                current = tag
                records[tag] = {"atoms": 0, "chains": set()}
            elif line.startswith("ATOM "):
                if current is None:
                    raise ValueError("coordinates without design id")
                coords = [float(line[start : start + 8]) for start in (30, 38, 46)]
                if not all(math.isfinite(x) for x in coords):
                    raise ValueError("nonfinite coordinates")
                records[current]["atoms"] += 1
                records[current]["chains"].add(line[21])
    if len(records) != expected_count or any(r["atoms"] == 0 or len(r["chains"]) < 2 for r in records.values()):
        raise ValueError("incomplete design count or binder-target structure")
    return {
        "designs": len(records),
        "ids": list(records),
        "atom_counts": {k: v["atoms"] for k, v in records.items()},
        "chains": {k: sorted(v["chains"]) for k, v in records.items()},
    }
