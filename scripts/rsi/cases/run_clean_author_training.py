"""Run one complete author CLEAN split10 training schedule after audited inputs."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import clean_author_training as training  # noqa: E402
from agent.rsi import clean_training_inputs as inputs  # noqa: E402


def main(args):
    import torch

    workspace = args.workspace_root.resolve()
    output, source = args.output.resolve(), args.inputs.resolve()
    for path in [output, source]:
        path.relative_to(workspace / "research_runs/rsi")
        if path == workspace / "research_runs/rsi":
            raise ValueError("A dedicated research directory is required")
    if os.environ.get("PYTHONHASHSEED") != "1234":
        raise ValueError("Launch with PYTHONHASHSEED=1234")
    if {k: importlib.metadata.version(k) for k in inputs.ENVIRONMENT} != inputs.ENVIRONMENT:
        raise ValueError("Pinned CLEAN environment required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    value = training.train_one(
        output,
        workspace / "research_runs/rsi/case_assets/CLEAN-training-v1",
        source,
        args.input_protocol_sha256,
        args.algorithm,
    )
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=list(training.SCHEDULES), required=True)
    main(parser.parse_args())
