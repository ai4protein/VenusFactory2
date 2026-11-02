"""Reconstruct CataPro training tables and execute unchanged author training code."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import csv
import fcntl
import json
import os
import random
import runpy
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi.cases.catapro_case import DATASETS, checked_manifest, dataset_rows, file_sha
from agent.rsi.cases.catapro_training import (
    PARAMETERS,
    STAGES,
    audit_stage,
    gpu_lease,
    make_table,
    stop_training_process,
    training_arguments,
    validate_fold_batches,
    verify_files,
)

EXPECTED_ROWS = {"kcat": 27658, "km": 42018, "activity": 25831}
SOURCE_COMMIT = "cc89b2c81768665cf6fd76dfda607ce88691f601"
IMPLEMENTATION = [
    "src/agent/rsi/__init__.py",
    "src/agent/rsi/_compat.py",
    "src/agent/rsi/cases/__init__.py",
    "src/agent/rsi/cases/catapro_case.py",
    "src/agent/rsi/cases/catapro_training.py",
    "scripts/rsi/cases/run_catapro_training.py",
]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def process_identity(pid):
    values = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": values[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def prepare(args):
    source, feature_root, root = args.source.resolve(), args.feature_root.resolve(), args.output.resolve()
    admission = checked_manifest(source, "admission.json")
    if admission["commit"] != SOURCE_COMMIT:
        raise ValueError("This reconstruction requires the admitted author revision")
    review = read(args.pdf_review)
    if review["visual_review_pending"] or review["parameters"] != PARAMETERS or review["table6_page"] != 21:
        raise ValueError("Reviewed supplementary Table 6 parameters are required")
    if file_sha(Path(review["source_pdf"])) != review["source_sha256"]:
        raise ValueError("Reviewed supplement changed")
    if type(args.epochs) is not int or not 1 <= args.epochs <= 150:
        raise ValueError("Epoch cap must be between one and 150")
    root.mkdir(parents=True, exist_ok=False)
    (root / "tables").mkdir()
    (root / "stages").mkdir()
    (root / "home").mkdir()
    sources = {}
    for family in ["kcat", "Km", "kcat_over_Km"]:
        for name in ["training.py", "model.py", "utils.py"]:
            relative = f"training/{family}/{name}"
            dest = root / "author_source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((source / relative).read_bytes())
            sources[str(dest.relative_to(root))] = file_sha(dest)
    implementation = {}
    for name in IMPLEMENTATION:
        dest = root / "repository" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
        implementation[str(dest.relative_to(root))] = file_sha(dest)
    tables, inputs = {}, {}
    for cohort in DATASETS:
        rows = dataset_rows(source, cohort)
        prior = read(feature_root / cohort / "report.json")
        contract = read(feature_root / cohort / "protocol.json")
        features_path = feature_root / cohort / "features.npy"
        csv_path = source / "datasets" / DATASETS[cohort]
        if len(rows) != EXPECTED_ROWS[cohort] or prior["status"] != "completed" or prior["rows_encoded"] != len(rows):
            raise ValueError("Full author cohort features are required")
        if file_sha(features_path) != prior["features_sha256"] or file_sha(csv_path) != contract["input_sha256"]:
            raise ValueError("Source data or reconstructed features changed")
        if (
            contract["source_admission_sha256"] != file_sha(source / "admission.json")
            or contract["source_commit"] != SOURCE_COMMIT
        ):
            raise ValueError("Feature source binding differs")
        features = np.load(features_path, allow_pickle=False, mmap_mode="r")
        kinetics = None
        if cohort == "activity":
            with csv_path.open(newline="") as stream:
                raw = list(csv.DictReader(stream))
            if [r[""] for r in raw] != [r["id"] for r in rows]:
                raise ValueError("Raw kinetic rows differ")
            kinetics = np.asarray([[float(r["kcat(s^-1)"]), 1000 * float(r["Km(M)"])] for r in raw])
        for stage, (dataset, target, _) in STAGES.items():
            if dataset != cohort:
                continue
            table = make_table(rows, features, target, kinetics)
            path = root / "tables" / (stage + ".pkl")
            table.to_pickle(path)
            tables[stage] = {
                "path": str(path.relative_to(root)),
                "sha256": file_sha(path),
                "rows": len(table),
                "columns": len(table.columns),
                "cohort": cohort,
                "target": target,
                "folds": validate_fold_batches(table, PARAMETERS[target]["batch_size"]),
            }
            del table
        inputs[cohort] = {
            "raw_csv_sha256": file_sha(csv_path),
            "features_sha256": file_sha(features_path),
            "feature_report_sha256": file_sha(feature_root / cohort / "report.json"),
            "feature_protocol_sha256": file_sha(feature_root / cohort / "protocol.json"),
            "rows": len(rows),
        }
    protocol = {
        "schema": 1,
        "source_commit": SOURCE_COMMIT,
        "source_repository": admission["repository"],
        "paper": admission["paper"],
        "author_source": sources,
        "implementation": implementation,
        "tables": tables,
        "inputs": inputs,
        "supplement_review": review,
        "supplement_review_sha256": file_sha(args.pdf_review),
        "parameters": PARAMETERS,
        "stages": list(STAGES),
        "epochs": args.epochs,
        "stage_seeds": {name: args.seed + i for i, name in enumerate(STAGES)},
        "stage_timeout_seconds": args.stage_timeout,
        "gpu_lock": str(args.gpu_lock.resolve()),
        "scope": "full released ten-fold training protocol"
        if args.epochs == 150
        else "full-row reduced-epoch execution validation; not a paper training reproduction",
        "training_fits": 50,
        "external_model_calls": 0,
        "execution_origin": "controller_initiated",
        "features": "Previously reconstructed from all raw sequences/SMILES with pinned encoders; no task-fitted features",
        "activity_base_training": "Separate kcat and Km pretraining on the activity cohort and its own original folds",
        "label_units": "Base log10 kcat(s^-1) and log10 Km(mM); activity raw kcat and Km(mM), transformed inside original author code",
        "alpha": {
            "value": 0.4,
            "basis": "Released inference/act_model.py default; training script default is 0.5",
            "source_sha256": file_sha(source / "inference/act_model.py"),
        },
        "preserved_author_semantics": [
            "Adam weight decay 0.01 for base models, 0.001 for activity",
            "Batchwise RMSE with epsilon 1e-6; early stop on mean validation-batch loss",
            "150 epoch default, patience 20, min_delta 0.001; checkpoint selection uses validation fold",
            "Activity base parameters detached, but BatchNorm buffers still update in train mode",
            "Eight training DataLoader workers; no dropped rows or altered batches",
        ],
        "reconstruction_decisions": [
            "Training feature pickle was not released; reconstruct positional tables from verified features and raw CSVs",
            "Author random seeds were not specified; predeclare one seed per ten-fold stage",
            "Use published Table 6 choices also for the two activity pretraining branches",
            "Disable TF32, set one torch CPU thread, seed Python/NumPy/Torch; run unchanged training.py via runpy",
        ],
        "limits": [
            "Validation folds also select checkpoints; not an unbiased new holdout",
            "No author grid-search reconstruction or wet-lab replication",
            "No claim of novel model or RSI improvement",
            "Training-run exact stochastic equivalence to unpublished author settings is not established",
        ],
        "audit_tolerances": {
            "rounded_prediction_atol": 2e-4,
            "rounded_label_atol": 6e-5,
            "aggregate_metric_atol": 2e-5,
        },
    }
    save(root / "protocol.json", protocol)
    save(
        root / "report.json", {"status": "prepared", "protocol_sha256": file_sha(root / "protocol.json"), "stages": []}
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "rows": EXPECTED_ROWS,
                "fits": 50,
                "epochs": args.epochs,
                "protocol_sha256": file_sha(root / "protocol.json"),
            }
        ),
        flush=True,
    )


def worker(root, stage):
    import torch

    protocol = read(root / "protocol.json")
    verify_files(root, protocol["author_source"] | protocol["implementation"])
    record = protocol["tables"][stage]
    verify_files(root, {record["path"]: record["sha256"]})
    fd = int(os.environ["VENUS_RSI_TRAINING_LEASE_FD"])
    held, expected = os.fstat(fd), Path(protocol["gpu_lock"]).stat()
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Wrong inherited GPU lease")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    seed = protocol["stage_seeds"][stage]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    family = STAGES[stage][2]
    author_dir = root / "author_source/training" / family
    sys.path.insert(0, str(author_dir))
    sys.argv = [str(author_dir / "training.py"), *training_arguments(root, stage, protocol["epochs"])]
    os.chdir(root / "stages" / stage)
    save(
        Path("runtime.json"),
        {
            "seed": seed,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": torch.cuda.get_device_name(0),
            "arguments": sys.argv[1:],
            "process": process_identity(os.getpid()),
        },
    )
    runpy.run_path(str(author_dir / "training.py"), run_name="__main__")


def run(root):
    protocol = read(root / "protocol.json")
    record = read(root / "report.json")
    if record["status"] != "prepared" or record["protocol_sha256"] != file_sha(root / "protocol.json"):
        raise ValueError("No automatic restart or changed protocol")
    verify_files(root, protocol["author_source"] | protocol["implementation"])
    process = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with (root / ".campaign.lock").open("a") as campaign_lock:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record.update(status="waiting_for_gpu", started_unix=time.time(), controller=process_identity(os.getpid()))
        save(root / "report.json", record)
        with gpu_lease(Path(protocol["gpu_lock"])) as gpu:
            try:
                record["status"] = "running"
                save(root / "report.json", record)
                for stage in protocol["stages"]:
                    if stage == "activity" and any(
                        r["status"] != "completed"
                        for r in record["stages"]
                        if r["stage"] in {"activity_kcat", "activity_km"}
                    ):
                        record["stages"].append({"stage": stage, "status": "dependency_failed"})
                        continue
                    folder = root / "stages" / stage
                    folder.mkdir(exist_ok=False)
                    row = {"stage": stage, "status": "running", "started_unix": time.time()}
                    record["stages"].append(row)
                    env = {
                        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                        "HOME": str(root / "home"),
                        "PYTHONPATH": str(ROOT / "src"),
                        "LANG": "C.UTF-8",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "VENUS_RSI_TRAINING_LEASE_FD": str(gpu.fileno()),
                    }
                    with (folder / "training.log").open("xb") as log:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                str(Path(__file__).resolve()),
                                "worker",
                                "--output",
                                str(root),
                                "--stage",
                                stage,
                            ],
                            cwd=ROOT,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            pass_fds=(gpu.fileno(),),
                        )
                        row["process"] = process_identity(process.pid)
                        save(root / "report.json", record)
                        try:
                            code = process.wait(timeout=protocol["stage_timeout_seconds"])
                        except subprocess.TimeoutExpired:
                            stop_training_process(process)
                            code = -999
                        process = None
                    row.update(
                        returncode=code, finished_unix=time.time(), status="training_failed" if code else "auditing"
                    )
                    save(root / "report.json", record)
                    if not code:
                        try:
                            audit = audit_stage(root, stage)
                            row.update(
                                status="completed",
                                audit_sha256=file_sha(folder / "audit.json"),
                                metrics=audit["metrics"],
                            )
                        except Exception as exc:
                            row.update(status="audit_failed", error_type=type(exc).__name__)
                            (folder / "audit_error.log").write_text(traceback.format_exc())
                    save(root / "report.json", record)
                verify_files(root, protocol["author_source"] | protocol["implementation"])
                record.update(
                    status="completed" if all(r["status"] == "completed" for r in record["stages"]) else "failed",
                    finished_unix=time.time(),
                )
                save(root / "report.json", record)
            except BaseException as exc:
                if process:
                    stop_training_process(process)
                record.update(status="failed", error_type=type(exc).__name__, finished_unix=time.time())
                save(root / "report.json", record)
                raise
    print(json.dumps({"status": record["status"], "stages": record["stages"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "run", "worker"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--feature-root", type=Path)
    p.add_argument("--pdf-review", type=Path)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--stage-timeout", type=int, default=14400)
    p.add_argument("--gpu-lock", type=Path, default=ROOT / "research_runs/rsi/.case_gpu.lock")
    p.add_argument("--stage", choices=list(STAGES))
    args = p.parse_args()
    if args.action == "prepare":
        if not all([args.source, args.feature_root, args.pdf_review]) or not 60 <= args.stage_timeout <= 86400:
            p.error("Prepare requires source, feature-root, pdf-review and a bounded stage timeout")
        prepare(args)
    elif args.action == "worker":
        if not args.stage:
            p.error("Worker requires stage")
        worker(args.output.resolve(), args.stage)
    else:
        run(args.output.resolve())
