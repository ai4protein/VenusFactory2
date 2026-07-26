#!/usr/bin/env python3
"""Download VenusFactory2 checkpoints from Hugging Face into ./ckpt.

Examples:
  python scripts/download_ckpts.py --preset demo
  python scripts/download_ckpts.py --preset predict-core
  python scripts/download_ckpts.py --preset proteinmpnn
  python scripts/download_ckpts.py --preset all
  python scripts/download_ckpts.py --include 'DeepSol/ankh-large/**' --include 'demo/**'
  python scripts/download_ckpts.py --list-presets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ckpt_hub import (  # noqa: E402
    ckpt_repo_id,
    ckpt_revision,
    ckpt_root,
    download_patterns,
    download_preset,
    estimate_download_bytes,
    list_presets,
    preset_patterns,
)
from hub_progress import format_bytes  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="Named preset to download (repeatable). See --list-presets.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra HF allow_pattern (repeatable), e.g. 'DeepSol/**'.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available presets and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local files exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress notices (sets VENUS_DOWNLOAD_QUIET=1).",
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help="Override VENUS_CKPT_REPO_ID for this run.",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Override VENUS_CKPT_REVISION for this run.",
    )
    parser.add_argument(
        "--local-dir",
        default="",
        help="Override VENUS_CKPT_DIR for this run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import os

    if args.quiet:
        os.environ["VENUS_DOWNLOAD_QUIET"] = "1"
    if args.repo_id:
        os.environ["VENUS_CKPT_REPO_ID"] = args.repo_id
    if args.revision:
        os.environ["VENUS_CKPT_REVISION"] = args.revision
    if args.local_dir:
        os.environ["VENUS_CKPT_DIR"] = args.local_dir

    if args.list_presets:
        for name in list_presets():
            patterns = list(preset_patterns(name))
            size = estimate_download_bytes(patterns)
            size_txt = f" (~{format_bytes(size)})" if size else ""
            print(f"{name}{size_txt}: {', '.join(patterns)}")
        return 0

    presets = list(args.preset)
    includes = list(args.include)
    if not presets and not includes:
        presets = ["predict-core"]

    planned: list[str] = []
    for preset in presets:
        planned.extend(preset_patterns(preset))
    planned.extend(includes)
    est = estimate_download_bytes(planned)

    print(f"repo      : {ckpt_repo_id()}@{ckpt_revision()}")
    print(f"local_dir : {ckpt_root()}")
    if est:
        print(f"approx_size: {format_bytes(est)} (from manifest)")
    print("note      : Hugging Face shows a live progress bar in the terminal.")
    print()

    for preset in presets:
        print(f"→ preset {preset}")
        download_preset(preset, force=args.force)

    if includes:
        print(f"→ include {includes}")
        download_patterns(includes, force=args.force)

    print()
    print("Done. Weights are ready under:", ckpt_root())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
