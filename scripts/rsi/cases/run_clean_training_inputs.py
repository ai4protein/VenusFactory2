"""Prepare every CLEAN split10 training input, orphan augmentation and evaluation input.

This software uses or is derived from CLEAN, University of Illinois Urbana-Champaign,
Department of Chemical and Biomolecular Engineering. Author code is separately licensed.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import fcntl
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import clean_training_inputs as case  # noqa: E402


def run(output, workspace, resume=False, wall_seconds=172800):
    output, workspace = output.resolve(), workspace.resolve()
    output.relative_to(workspace / "research_runs/rsi")
    if output == workspace / "research_runs/rsi" or not 1 <= wall_seconds <= 604800:
        raise ValueError("Invalid training preparation allocation")
    if os.environ.get("PYTHONHASHSEED") != "1234":
        raise ValueError("Launch Python with PYTHONHASHSEED=1234")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("No prepared run to resume")
    with (output / ".worker.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(output, workspace, resume, wall_seconds)


def execute(output, workspace, resume, wall_seconds):
    started = time.monotonic()
    assets = workspace / "research_runs/rsi/case_assets/CLEAN-training-v1"
    manifest = case.verify_assets(assets)
    versions = {k: importlib.metadata.version(k) for k in case.ENVIRONMENT}
    if versions != case.ENVIRONMENT:
        raise ValueError("Pinned CLEAN environment required")
    for item in manifest["encoder"].values():
        if case.sha(item["path"]) != item["sha256"]:
            raise ValueError("Pinned ESM encoder changed")
    import esm
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1234)
    source_paths = [Path(__file__), Path(case.__file__)]
    source_hashes = {p.name: case.sha(p) for p in source_paths}
    if resume:
        protocol = json.loads(case.ordinary(output, "protocol.json").read_text())
        plan = json.loads(case.ordinary(output, "plan.json").read_text())
        if (
            protocol["implementation"] != source_hashes
            or protocol["environment"] != versions
            or protocol["plan_sha256"] != case.sha(output / "plan.json")
            or plan["encoder"] != manifest["encoder"]
        ):
            raise ValueError("Resume must retain the exact inputs and implementation")
        report = json.loads(case.ordinary(output, "report.json").read_text())
        if report["status"] == "completed":
            return case.verify_completed(output)
        report["resumes"] = report.get("resumes", 0) + 1
        (output / "history").mkdir(exist_ok=True)
        history = output / "history" / ("before_resume_" + str(report["resumes"]) + ".json")
        if history.exists():
            raise ValueError("Resume history identity already exists")
        shutil.copy2(output / "report.json", history)
        report["previous_seconds"] = report.get("previous_seconds", 0) + report.get("wall_seconds", 0)
        journal = output / "attempts.jsonl"
        report["attempts_started"] = (
            sum(json.loads(line)["event"] == "started" for line in journal.read_text().splitlines())
            if journal.exists()
            else 0
        )
    else:
        for p in source_paths:
            shutil.copy2(p, output / p.name)
        (output / "receipts").mkdir()
        plan = case.prepare_plan(output / "work", assets, workspace / "research_runs/rsi/case_assets/CLEAN/source")
        case.save(output / "plan.json", plan)
        protocol = {
            "schema": 1,
            "plan_sha256": case.sha(output / "plan.json"),
            "implementation": source_hashes,
            "environment": versions,
            "seed": 1234,
            "device": "cpu",
            "threads": 4,
            "scope": "full released split10 training-input preparation plus three evaluation cohorts",
            "training": manifest["planned_algorithms"],
            "omissions": [
                "training updates (separate stage)",
                "five-fold publication cross-validation",
                "autonomous model generation",
                "wet-lab validation",
                "RSI efficacy",
            ],
            "audit_boundary": "All token spans, tensor hashes and receipt bindings; no independent second transformer run",
        }
        case.save(output / "protocol.json", protocol)
        report = {
            "status": "preparing",
            "protocol_sha256": case.digest(protocol),
            "counts": case.COUNTS,
            "completed_records": 0,
            "training_completed": False,
            "autonomous_model_developed": False,
            "full_paper_reproduced": False,
            "external_model_calls": 0,
            "resumes": 0,
            "attempts_started": 0,
            "previous_seconds": 0,
        }
    case.validate_plan(plan)
    protocol_hash = case.digest(protocol)
    report.update(status="running", completed_records=0, stage="embeddings")
    case.save(output / "report.json", report)
    try:
        with torch.serialization.safe_globals([argparse.Namespace]):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(manifest["encoder"]["model"]["path"])
        model.eval()
        converter = alphabet.get_batch_converter()
        with (output / "attempts.jsonl").open("a") as attempts:
            for row in plan["records"]:
                key = row["key"]
                target = output / "work/data/esm_data" / (key + ".pt")
                receipt = output / "receipts" / (key + ".json")
                if receipt.exists() or target.exists():
                    case.verify_receipt(
                        output / "work", output / "receipts", row, manifest["encoder"]["model"]["sha256"], protocol_hash
                    )
                else:
                    if time.monotonic() - started > wall_seconds:
                        raise TimeoutError("Preparation wall budget reached; retained receipts can be resumed")
                    attempts.write(json.dumps({"event": "started", "key": key, "at": time.time()}) + "\n")
                    attempts.flush()
                    os.fsync(attempts.fileno())
                    report["attempts_started"] += 1
                    _, _, tokens = converter([(key, row["sequence"])])
                    if tokens.shape != (1, row["tokens"] + 2):
                        raise ValueError("ESM tokenization span changed")
                    with torch.inference_mode():
                        values = model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
                    vector = values[0, 1 : row["tokens"] + 1].mean(dim=0).detach().cpu().contiguous()
                    if vector.shape != (1280,) or vector.dtype != torch.float32 or not torch.isfinite(vector).all():
                        raise ValueError("Invalid new ESM representation")
                    temporary = target.with_suffix(".pt.writing")
                    if temporary.exists():
                        raise ValueError("Uncommitted embedding artifact requires reconciliation")
                    torch.save(vector, temporary)
                    os.replace(temporary, target)
                    case.save(
                        receipt,
                        case.receipt_payload(row, target, manifest["encoder"]["model"]["sha256"], protocol_hash),
                    )
                    attempts.write(json.dumps({"event": "completed", "key": key, "at": time.time()}) + "\n")
                    attempts.flush()
                report["completed_records"] += 1
                if report["completed_records"] % 20 == 0:
                    report["wall_seconds"] = time.monotonic() - started
                    case.save(output / "report.json", report)
                    print(
                        json.dumps(
                            {"completed": report["completed_records"], "total": case.TOTAL, "cohort": row["cohort"]}
                        ),
                        flush=True,
                    )
        report.update(status="completed", stage="verified_inputs", wall_seconds=time.monotonic() - started)
        case.save(output / "report.json", report)
        report["independent_receipt_audit"] = case.verify_completed(output)
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc), wall_seconds=time.monotonic() - started
        )
        raise
    finally:
        case.save(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-seconds", type=int, default=172800)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.workspace_root, args.resume, args.wall_seconds)), flush=True)
