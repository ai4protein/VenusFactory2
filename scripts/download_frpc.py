#!/usr/bin/env python3
"""Download Gradio ``frpc`` binary (not shipped via git).

Sources (automatic fallback):
  1. Official Gradio CDN (install_config.json)
  2. Hugging Face: AI4Protein/VenusFactory2-ckpts / assets/frpc/

Examples:
  python scripts/download_frpc.py
  python scripts/download_frpc.py --force
  python scripts/download_frpc.py --to-gradio-cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from frpc_hub import (  # noqa: E402
    download_frpc,
    ensure_frpc,
    frpc_filename,
    gradio_cache_dir,
    install_to_gradio_cache,
    local_project_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Redownload even if local file exists.")
    parser.add_argument(
        "--to-gradio-cache",
        action="store_true",
        help="Also install into ~/.cache/huggingface/gradio/frpc (Gradio share).",
    )
    parser.add_argument(
        "--dest-dir",
        default="",
        help="Directory for the binary (default: repo root).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress notices (sets VENUS_DOWNLOAD_QUIET=1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        import os

        os.environ["VENUS_DOWNLOAD_QUIET"] = "1"

    print(f"target     : {frpc_filename()}")
    print("sources    : Gradio CDN → Hugging Face assets/frpc fallback")
    print("note       : Progress prints to stderr; HF also shows a bar when used.")
    print()

    dest_dir = Path(args.dest_dir) if args.dest_dir else None
    if args.to_gradio_cache and dest_dir is None:
        path = ensure_frpc(force=args.force)
    elif args.to_gradio_cache:
        path = download_frpc(dest_dir=dest_dir, force=args.force)
        cache_path = install_to_gradio_cache(force=args.force)
        print(f"gradio_cache: {cache_path}")
    else:
        path = download_frpc(dest_dir=dest_dir, force=args.force)

    print()
    print(f"filename   : {frpc_filename()}")
    print(f"local_path : {path}")
    print(f"size_bytes : {path.stat().st_size}")
    print(f"gradio_dir : {gradio_cache_dir()}")
    print(f"project_bin: {local_project_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
