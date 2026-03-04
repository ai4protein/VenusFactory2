"""
AlphaFold operations: single exit for query (return JSON with content) and download (save to file, return JSON).

All public functions return JSON string with:
- query_*: {"success": bool, "content": str}  (content is PDB/mmCIF or metadata JSON text)
- download_*: {"success": bool, "file_path": str or null}
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


def _query_result(success: bool, content: Optional[str] = None, error: Optional[str] = None) -> str:
    """Build JSON for query_* result: success, content (direct content for query)."""
    out = {"success": success, "content": content}
    if error is not None:
        out["error"] = error
    return json.dumps(out, ensure_ascii=False)


def _download_result(success: bool, file_path: Optional[str] = None, error: Optional[str] = None) -> str:
    """Build JSON for download_* result: success, file_path."""
    out = {"success": success, "file_path": file_path}
    if error is not None:
        out["error"] = error
    return json.dumps(out, ensure_ascii=False)


# ---------- query_*: return JSON with success + content ----------


def query_alphafold_structure_by_uniprot_id(
    uniprot_id: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> str:
    """Query AlphaFold structure by UniProt ID. Returns JSON: {success, content} (content is PDB/mmCIF text)."""
    raw = _query_structure(
        uniprot_id, format=format, version=version, fragment=fragment
    )
    if raw.strip().startswith(f"{uniprot_id} failed,") or raw.strip().startswith("failed"):
        return _query_result(False, content=None, error=raw)
    return _query_result(True, content=raw)


def query_alphafold_metadata_by_uniprot_id(uniprot_id: str) -> str:
    """Query AlphaFold prediction metadata by UniProt ID. Returns JSON: {success, content} (content is metadata JSON text)."""
    raw = _query_metadata(uniprot_id)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("success") is False:
            return _query_result(False, content=raw, error=data.get("error", "unknown"))
    except json.JSONDecodeError:
        pass
    return _query_result(True, content=raw)


# ---------- download_*: save to file, return JSON with success + file_path ----------


def download_alphafold_structure_by_uniprot_id(
    uniprot_id: str,
    out_dir: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = "v6",
    fragment: int = 1,
) -> str:
    """Download AlphaFold structure to a file or directory. Returns JSON: {success, file_path}."""
    out_dir = str(out_dir).strip()
    if not out_dir:
        return _download_result(False, file_path=None, error="invalid out_path")
    if os.path.isdir(out_dir) or out_dir.endswith(os.sep):
        out_dir = out_dir.rstrip(os.sep)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    success, path = _download_structure_impl(
        uniprot_id, out_dir, format=format, version=version, fragment=fragment
    )
    if not success or not path:
        return _download_result(False, file_path=None, error=str(uniprot_id))
    return _download_result(True, file_path=os.path.join(out_dir, f"{uniprot_id}.{format}"))


def download_alphafold_metadata_by_uniprot_id(uniprot_id: str, out_dir: str) -> str:
    """Download AlphaFold metadata to a file or directory. Returns JSON: {success, file_path}."""
    out_dir = str(out_dir).strip()
    if not out_dir:
        return _download_result(False, file_path=None, error="invalid out_dir")
    if os.path.isdir(out_dir) or out_dir.endswith(os.sep):
        out_dir = out_dir.rstrip(os.sep)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        msg = _download_metadata(uniprot_id, out_dir)
        if "failed" in msg.lower():
            return _download_result(False, file_path=None, error=msg)
        return _download_result(True, file_path=os.path.join(out_dir, f"{uniprot_id}.json"))
    return _download_result(False, file_path=None, error="invalid out_dir")


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
    "download_alphafold_structure_by_uniprot_id",
    "download_alphafold_metadata_by_uniprot_id",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="AlphaFold operations: query_* (return JSON with content) and download_* (return JSON with file_path)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_* and download_* samples; output under example/database/alphafold",
    )
    parser.add_argument("--uniprot_id", type=str, default="A0A1B0GTW7", help="UniProt ID for test")
    parser.add_argument("--format", type=str, default="pdb", choices=["pdb", "cif"], help="Structure format: 'pdb' (default) or 'cif'.")
    parser.add_argument("--version", type=str, default="v6", choices=["v1", "v2", "v4", "v6"], help="AlphaFold DB version: v1, v2, v4, or v6. Default v6.")
    parser.add_argument("--out_dir", type=str, default="example/database/alphafold", help="Output directory for AlphaFold structure and metadata. Default 'example/database/alphafold'.")
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    uid = args.uniprot_id or "A0A1B0GTW7"
    fmt = "cif" if (args.format or "").lower() in ("cif", "mmcif") else "pdb"

    print("=== query_* (return JSON: success + content) ===")
    print("  query_alphafold_structure_by_uniprot_id(...)")
    res = query_alphafold_structure_by_uniprot_id(uid, format=fmt, version=args.version)
    obj = json.loads(res)
    print(f"  structure: {res}")
    if obj.get("content") and len(obj["content"]) > 500:
        print(f"  (first 500 chars of content): {obj['content'][:500]}...")
    elif obj.get("content"):
        print(f"  content: {obj['content']}")
    if obj.get("error"):
        print(f"  error: {obj['error']}")
    query_structure_path = os.path.join(out_dir, f"query_structure_{uid}.txt")
    with open(query_structure_path, "w", encoding="utf-8") as f:
        f.write(res)
    print(f"  full JSON saved to {query_structure_path}")

    print("  query_alphafold_metadata_by_uniprot_id(...)")
    meta_res = query_alphafold_metadata_by_uniprot_id(uid)
    meta_obj = json.loads(meta_res)
    print(f"  metadata: {meta_res}")
    if meta_obj.get("content") and len(meta_obj["content"]) > 500:
        print(f"  (first 500 chars of content): {meta_obj['content'][:500]}...")
    elif meta_obj.get("content"):
        print(f"  content: {meta_obj['content']}")
    if meta_obj.get("error"):
        print(f"  error: {meta_obj['error']}")
    query_metadata_path = os.path.join(out_dir, f"query_metadata_{uid}.txt")
    with open(query_metadata_path, "w", encoding="utf-8") as f:
        f.write(meta_res)
    print(f"  full JSON saved to {query_metadata_path}")

    print("=== download_* (return JSON: success + file_path) ===")
    dl_struct = json.loads(download_alphafold_structure_by_uniprot_id(uid, out_dir, format=fmt, version=args.version))
    print(f"  structure: {dl_struct}")
    dl_meta = json.loads(download_alphafold_metadata_by_uniprot_id(uid, out_dir))
    print(f"  metadata: {dl_meta}")

    print(f"Done. Output under {out_dir}")
