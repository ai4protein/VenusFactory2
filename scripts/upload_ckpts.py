#!/usr/bin/env python3
"""Upload local ./ckpt weights to Hugging Face (AI4Protein/VenusFactory2-ckpts).

Also regenerates ckpt/manifest.json (sha256 + presets) before upload.

Examples:
  python scripts/upload_ckpts.py --dry-run
  python scripts/upload_ckpts.py --create-pr
  python scripts/upload_ckpts.py --include 'DeepSol/**' --create-pr
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ckpt_hub import (  # noqa: E402
    DEFAULT_REPO_ID,
    PRESETS,
    ckpt_root,
    iter_local_weight_files,
    normalize_rel_path,
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> dict:
    files = []
    for path in iter_local_weight_files(root):
        rel = normalize_rel_path(path)
        files.append(
            {
                "path": rel,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "version": 1,
        "repo_id": DEFAULT_REPO_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "presets": {name: list(patterns) for name, patterns in PRESETS.items()},
        "files": files,
    }


def _match_include(rel: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    from fnmatch import fnmatch

    return any(fnmatch(rel, pat.rstrip("/")) or fnmatch(rel, pat) for pat in patterns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main", help="Target branch name.")
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Open a PR instead of pushing directly to the branch (needed without write access).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only upload matching relative paths (fnmatch, repeatable).",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Do not regenerate/upload manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded, then exit.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update VenusFactory2 checkpoints",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = ckpt_root()
    if not root.is_dir():
        print(f"Local ckpt root not found: {root}", file=sys.stderr)
        return 1

    manifest = build_manifest(root)
    manifest_path = _REPO_ROOT / "ckpt" / "manifest.json"
    if not args.skip_manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {manifest_path} ({len(manifest['files'])} files)")

    upload_map: dict[str, Path] = {}
    for item in manifest["files"]:
        rel = item["path"]
        if not _match_include(rel, args.include):
            continue
        upload_map[rel] = root / rel

    # Include docs/manifest unless the caller filtered them out with --include.
    for extra in ("README.md", "manifest.json"):
        candidate = root / extra
        if not candidate.is_file():
            candidate = _REPO_ROOT / "ckpt" / extra
        if not candidate.is_file():
            continue
        if args.include and not _match_include(extra, args.include):
            continue
        upload_map[extra] = candidate

    if not upload_map:
        print("Nothing to upload.", file=sys.stderr)
        return 1

    total_bytes = sum(p.stat().st_size for p in upload_map.values())
    print(f"repo={args.repo_id} files={len(upload_map)} bytes={total_bytes}")
    for rel in sorted(upload_map):
        print(f"  {rel} ({upload_map[rel].stat().st_size} bytes)")

    if args.dry_run:
        print("Dry-run only; no upload.")
        return 0

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="venus-ckpt-upload-") as tmp:
        staging = Path(tmp)
        for rel, src in upload_map.items():
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        def _do_upload(*, create_pr: bool):
            return api.upload_folder(
                folder_path=str(staging),
                repo_id=args.repo_id,
                repo_type="model",
                revision=args.revision,
                commit_message=args.commit_message,
                create_pr=create_pr,
                allow_patterns=["**/*"],
            )

        create_pr = bool(args.create_pr)
        try:
            result = _do_upload(create_pr=create_pr)
        except HfHubHTTPError as exc:
            # Contributors without write access can usually still open a PR.
            if (not create_pr) and getattr(exc, "response", None) is not None and exc.response.status_code == 403:
                print(
                    "Direct push forbidden (need write access). Retrying with --create-pr...\n"
                    "After upload, an org admin must merge the PR(s) into main before "
                    "auto-download works for other users.",
                    file=sys.stderr,
                )
                result = _do_upload(create_pr=True)
            else:
                raise
        print("Upload result:", result)
        if create_pr:
            print(
                "Opened as a PR/draft — an org admin must merge it:\n"
                f"  https://huggingface.co/{args.repo_id}/discussions"
            )
        else:
            print(f"Pushed directly to {args.revision}: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
