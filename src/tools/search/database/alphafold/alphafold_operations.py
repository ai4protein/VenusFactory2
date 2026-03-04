"""
AlphaFold operations: single exit for query (return text) and download (save to file).

All public functions are either:
- query_*: return str (PDB/mmCIF or JSON text)
- download_*: write to file and return status message str.
"""

import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

try:
    from .alphafold_structure import (
        query_alphafold_structure as _query_structure,
        download_alphafold_structure as _download_structure_impl,
    )
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "alphafold" and str(_dir.parents[4]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[4]))
    from src.tools.search.database.alphafold.alphafold_structure import (
        query_alphafold_structure as _query_structure,
        download_alphafold_structure as _download_structure_impl,
    )

try:
    from .alphafold_metadata import (
        query_alphafold_metadata as _query_metadata,
        download_alphafold_metadata as _download_metadata,
    )
except ImportError:
    from src.tools.search.database.alphafold.alphafold_metadata import (
        query_alphafold_metadata as _query_metadata,
        download_alphafold_metadata as _download_metadata,
    )


# ---------- query_*: return text ----------


def query_alphafold_structure(
    uniprot_id: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> str:
    """Query AlphaFold structure by UniProt ID. Returns PDB or mmCIF text. No file save."""
    return _query_structure(
        uniprot_id, format=format, version=version, fragment=fragment
    )


def query_alphafold_metadata(uniprot_id: str) -> str:
    """Query AlphaFold prediction metadata by UniProt ID. Returns JSON text. No file save."""
    return _query_metadata(uniprot_id)


# ---------- download_*: save to file, return message ----------


def download_alphafold_structure(
    uniprot_id: str,
    out_path: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> str:
    """Download AlphaFold structure to a file or directory. Returns status message."""
    out_path = str(out_path).strip()
    if not out_path:
        return "Download failed: invalid out_path"
    if os.path.isdir(out_path) or out_path.endswith(os.sep):
        out_dir = out_path.rstrip(os.sep)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        want_file = None
    else:
        out_file = out_path
        out_dir = os.path.dirname(out_file)
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        want_file = out_file
    success, path = _download_structure_impl(
        uniprot_id, out_dir or ".", format=format, version=version, fragment=fragment
    )
    if not success or not path:
        return f"Download failed: {uniprot_id}"
    if want_file and os.path.exists(path) and os.path.abspath(path) != os.path.abspath(want_file):
        try:
            import shutil
            shutil.move(path, want_file)
            path = want_file
        except Exception as e:
            return f"Download failed: {e}"
    return f"Saved structure to {path}"


def download_alphafold_metadata(uniprot_id: str, out_path: str) -> str:
    """Download AlphaFold metadata to a file or directory. Returns status message."""
    out_path = str(out_path).strip()
    if not out_path:
        return "Download failed: invalid out_path"
    if os.path.isdir(out_path) or out_path.endswith(os.sep):
        out_dir = out_path.rstrip(os.sep)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        msg = _download_metadata(uniprot_id, out_dir)
        if "failed" in msg.lower():
            return msg
        return f"Saved metadata to {os.path.join(out_dir, uniprot_id + '.json')}"
    out_file = out_path
    out_dir = os.path.dirname(out_file)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    msg = _download_metadata(uniprot_id, out_dir or ".")
    if "failed" in msg.lower():
        return msg
    default_path = os.path.join(out_dir or ".", f"{uniprot_id}.json")
    if os.path.abspath(default_path) != os.path.abspath(out_file) and os.path.exists(default_path):
        try:
            import shutil
            shutil.move(default_path, out_file)
        except Exception as e:
            return f"Download failed: {e}"
    return f"Saved metadata to {out_file}"


def _download_alphafold_structure_impl(
    uniprot_id: str,
    out_dir: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> Tuple[bool, Optional[str]]:
    """Internal: download structure and return (success, path). Used by tools_mcp for OSS upload."""
    return _download_structure_impl(
        uniprot_id, out_dir, format=format, version=version, fragment=fragment
    )


__all__ = [
    "query_alphafold_structure",
    "query_alphafold_metadata",
    "download_alphafold_structure",
    "download_alphafold_metadata",
    "_download_alphafold_structure_impl",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="AlphaFold operations: query_* (return text) and download_* (save file)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_* and download_* samples; output under example/database/alphafold",
    )
    parser.add_argument("--uniprot_id", type=str, default="A0A1B0GTW7", help="UniProt ID for test")
    parser.add_argument("--format", type=str, default="pdb", choices=["pdb", "cif"])
    parser.add_argument("--version", type=str, default="v6", choices=["v1", "v2", "v4", "v6"])
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_base = os.path.join("example", "database", "alphafold")
    os.makedirs(out_base, exist_ok=True)
    structure_dir = os.path.join(out_base, "structure")
    metadata_dir = os.path.join(out_base, "metadata")
    os.makedirs(structure_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    uid = args.uniprot_id or "A0A1B0GTW7"
    fmt = "cif" if (args.format or "").lower() in ("cif", "mmcif") else "pdb"

    print("=== query_* (return text) ===")
    print("  query_alphafold_structure(...)")
    text = query_alphafold_structure(uid, format=fmt, version=args.version)
    if len(text) > 500:
        print(f"  (first 500 chars): {text[:500]}...")
    else:
        print(f"  result: {text}")
    query_structure_path = os.path.join(out_base, "query_structure_sample.txt")
    with open(query_structure_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  saved to {query_structure_path}")

    print("  query_alphafold_metadata(...)")
    meta_text = query_alphafold_metadata(uid)
    if len(meta_text) > 500:
        print(f"  (first 500 chars): {meta_text[:500]}...")
    else:
        print(f"  result: {meta_text}")
    query_metadata_path = os.path.join(out_base, "query_metadata_sample.json")
    with open(query_metadata_path, "w", encoding="utf-8") as f:
        f.write(meta_text)
    print(f"  saved to {query_metadata_path}")

    print("=== download_* (save to file) ===")
    print("  ", download_alphafold_structure(uid, structure_dir, format=fmt, version=args.version))
    print("  ", download_alphafold_metadata(uid, metadata_dir))

    print(f"Done. Output under {out_base}")
