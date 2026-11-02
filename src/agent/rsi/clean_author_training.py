"""Execute fixed author CLEAN training schedules with auditable tensor caching.

This software uses or is derived from CLEAN, developed at the University of
Illinois Urbana-Champaign, Department of Chemical and Biomolecular Engineering.
The separately licensed author scripts are executed without modification.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

from . import clean_training_inputs as inputs

SCHEDULES = {
    "triplet": {"script": "train-triplet.py", "epochs": 2000, "out_dim": 128, "adaptive_rate": 100},
    "supconH": {"script": "train-supconH.py", "epochs": 1500, "out_dim": 256, "adaptive_rate": 60},
}


@contextlib.contextmanager
def cached_embeddings(cache):
    """Return fresh tensor copies; preserve ordinary torch.load for every other path."""
    import torch

    original = torch.load
    calls = {"cache_hits": 0}

    def load(path, *args, **kwargs):
        if not args and not kwargs and isinstance(path, (str, os.PathLike)):
            key = str(Path(path).resolve())
            if key in cache:
                calls["cache_hits"] += 1
                return cache[key].clone()
        return original(path, *args, **kwargs)

    torch.load = load
    try:
        yield calls
    finally:
        torch.load = original


def load_author(assets, algorithm):
    if algorithm not in SCHEDULES:
        raise ValueError("Unknown fixed training schedule")
    inputs.verify_assets(assets)
    sys.path.insert(0, str(assets / "source/app/src"))
    path = assets / "source/app" / SCHEDULES[algorithm]["script"]
    spec = importlib.util.spec_from_file_location("clean_author_" + algorithm, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def verify_trace(records, schedule):
    if any(type(r.get("epoch")) is not int for r in records) or [r.get("epoch") for r in records] != list(
        range(1, schedule["epochs"] + 1)
    ):
        raise ValueError("Missing, repeated or unordered training epochs")
    if any(type(r.get("loss")) not in (int, float) or not math.isfinite(r["loss"]) or r["loss"] < 0 for r in records):
        raise ValueError("Invalid training loss")


def train_one(output, assets, input_output, expected_input_protocol, algorithm):
    import torch

    schedule = SCHEDULES.get(algorithm)
    if schedule is None:
        raise ValueError("Unknown training schedule")
    protocol = json.loads(inputs.ordinary(input_output, "protocol.json").read_text())
    if inputs.digest(protocol) != expected_input_protocol:
        raise ValueError("Training input provenance changed")
    inputs.verify_completed(input_output)
    plan = json.loads((input_output / "plan.json").read_text())
    author, script = load_author(assets, algorithm)
    script_hash = inputs.sha(script)
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    (work / "data").mkdir(parents=True)
    shutil.copy2(input_output / "work/data/split10.csv", work / "data/split10.csv")
    (work / "data/esm_data").symlink_to(input_output / "work/data/esm_data", target_is_directory=True)
    for path in [Path(__file__), script]:
        shutil.copy2(path, output / path.name)
    cache = {}
    for row in plan["records"]:
        if row["cohort"] not in {"train", "augmentation"}:
            continue
        tensor = inputs.verify_receipt(
            input_output / "work",
            input_output / "receipts",
            row,
            plan["encoder"]["model"]["sha256"],
            expected_input_protocol,
        )
        cache[str((input_output / "work/data/esm_data" / (row["key"] + ".pt")).resolve())] = tensor
    original_argv, original_cwd = sys.argv, Path.cwd()
    initial = output / "initial.pth"
    original_net, original_train = author.LayerNormNet, author.train
    started = time.monotonic()
    report = {
        "status": "running",
        "stage": "initial_distances",
        "algorithm": algorithm,
        "schedule": schedule,
        "source_commit": inputs.COMMIT,
        "author_script_sha256": script_hash,
        "input_protocol_sha256": expected_input_protocol,
        "completed_epochs": 0,
        "new_model_developed": False,
        "autonomous_model_developed": False,
        "full_paper_reproduced": False,
        "external_model_calls": 0,
        "evaluation_completed": False,
        "training_completed": False,
        "cache_policy": "Verified float32 tensors; each author load gets a fresh clone. RNG, sampling, optimizer, batches, loss and schedules are unchanged.",
    }
    inputs.save(output / "report.json", report)

    def network(*args, **kwargs):
        model = original_net(*args, **kwargs)
        torch.save(model.state_dict(), initial)
        return model

    def epoch(*args, **kwargs):
        value = original_train(*args, **kwargs)
        index = args[2]
        if not math.isfinite(value) or value < 0:
            raise ValueError("Author training produced invalid loss")
        with (output / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": index, "loss": value, "at": time.time()}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        report.update(
            stage="training", completed_epochs=index, last_loss=value, wall_seconds=time.monotonic() - started
        )
        inputs.save(output / "report.json", report)
        return value

    try:
        author.LayerNormNet, author.train = network, epoch
        os.chdir(work)
        sys.argv = [str(script), "--training_data", "split10", "--model_name", "split10_" + algorithm]
        args = author.parse()
        if (
            args.epoch != schedule["epochs"]
            or args.out_dim != schedule["out_dim"]
            or args.adaptive_rate != schedule["adaptive_rate"]
            or args.learning_rate != 5e-4
            or args.hidden_dim != 512
        ):
            raise ValueError("Author schedule no longer matches admitted defaults")
        if algorithm == "supconH" and (args.n_pos, args.n_neg, args.temp) != (9, 30, 0.1):
            raise ValueError("SupCon-Hard sampling changed")
        with cached_embeddings(cache) as cache_report:
            from CLEAN.utils import compute_esm_distance

            compute_esm_distance("split10")
            author.main()
        report["cache_hits"] = cache_report["cache_hits"]
        records = [json.loads(line) for line in (output / "epochs.jsonl").read_text().splitlines()]
        verify_trace(records, schedule)
        final = work / "data/model" / ("split10_" + algorithm + ".pth")
        shutil.copy2(final, output / "final.pth")
        before = torch.load(initial, map_location="cpu", weights_only=True)
        after = torch.load(output / "final.pth", map_location="cpu", weights_only=True)
        if set(before) != set(after) or any(not torch.isfinite(v).all() for v in after.values()):
            raise ValueError("Training state layout or finiteness differs")
        changed = sum(int(torch.count_nonzero(before[k] != after[k]).item()) for k in before)
        if not changed or after["fc3.weight"].shape != (schedule["out_dim"], 512):
            raise ValueError("No valid trained checkpoint")
        if inputs.sha(script) != script_hash:
            raise ValueError("Author script changed during execution")
        inputs.verify_assets(assets)
        report.update(
            status="completed",
            stage="trained_checkpoint",
            training_completed=True,
            parameter_values_changed=changed,
            artifact_sha256={
                n: inputs.sha(output / n)
                for n in ["initial.pth", "final.pth", "epochs.jsonl", script.name, Path(__file__).name]
            },
        )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        author.LayerNormNet, author.train = original_net, original_train
        sys.argv = original_argv
        os.chdir(original_cwd)
        report["wall_seconds"] = time.monotonic() - started
        inputs.save(output / "report.json", report)
    return report
