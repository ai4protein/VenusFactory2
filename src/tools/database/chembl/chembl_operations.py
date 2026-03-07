"""
ChEMBL operations: single exit for query and download; both return rich JSON.

Success: status, file_info (download) or content (query), content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .chembl_molecule import get_molecule
    from .chembl_similarity import similarity_search
    from .chembl_substructure import substructure_search
    from .chembl_drug import get_drug, get_mechanisms, get_indications
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "chembl" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.chembl.chembl_molecule import get_molecule
    from src.tools.database.chembl.chembl_similarity import similarity_search
    from src.tools.database.chembl.chembl_substructure import substructure_search
    from src.tools.database.chembl.chembl_drug import get_drug, get_mechanisms, get_indications


_PREVIEW_LEN = 500
_SOURCE_CHEMBL = "ChEMBL"


def _json_serializer(obj: Any) -> Any:
    """Default serializer for ChEMBL API objects (e.g. dict with non-str values)."""
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
        try:
            return list(obj)
        except Exception:
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    """Build JSON for error: status error, error { type, message, suggestion }, file_info null."""
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _download_success_response(
    file_path: str,
    content_preview: Optional[str] = None,
    biological_metadata: Optional[Dict[str, Any]] = None,
    download_time_ms: int = 0,
    source: str = _SOURCE_CHEMBL,
) -> str:
    """Build JSON for download success: status, file_info, content_preview, biological_metadata, execution_context."""
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "json"
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": str(path.resolve()) if path.exists() else file_path,
            "file_name": path.name,
            "file_size": file_size,
            "format": fmt,
        },
        "content_preview": (content_preview or "")[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata or {},
        "execution_context": {"download_time_ms": download_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


def _query_success_response(
    content: str,
    content_preview: Optional[str] = None,
    biological_metadata: Optional[Dict[str, Any]] = None,
    query_time_ms: int = 0,
    source: str = _SOURCE_CHEMBL,
) -> str:
    """Build JSON for query success: status, content, content_preview, biological_metadata, execution_context."""
    preview = (content_preview or content or "")[:_PREVIEW_LEN]
    out: Dict[str, Any] = {
        "status": "success",
        "content": content,
        "content_preview": preview,
        "biological_metadata": biological_metadata or {},
        "execution_context": {"query_time_ms": query_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False, default=_json_serializer)


def _read_preview(path: str, max_chars: int = _PREVIEW_LEN) -> str:
    """Read first max_chars from file for content_preview."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


# ---------- query_chembl_*: return rich JSON (status, content, content_preview, biological_metadata, execution_context) ----------


def query_chembl_molecule_by_id(chembl_id: str) -> str:
    """Query ChEMBL molecule by ChEMBL ID. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        mol = get_molecule(chembl_id)
        if mol is None:
            return _error_response("NotFound", f"Molecule {chembl_id} not found", suggestion="Check ChEMBL ID format (e.g. CHEMBL25).")
        payload = {"chembl_id": chembl_id, "molecule": mol}
        content = json.dumps(payload, ensure_ascii=False, default=_json_serializer)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"chembl_id": chembl_id}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except ImportError as e:
        return _error_response("ImportError", f"ChEMBL dependency missing: {e}", suggestion="pip install chembl-webresource-client")
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check ChEMBL ID and network connection.")


def query_chembl_similarity_by_smiles(
    smiles: str,
    threshold: int = 85,
    max_results: Optional[int] = None,
) -> str:
    """Query ChEMBL for similar molecules by SMILES (Tanimoto 0-100). Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        results = similarity_search(smiles, threshold=threshold, max_results=max_results)
        payload = {"smiles": smiles, "threshold": threshold, "count": len(results) if results else 0, "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=_json_serializer)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"smiles": smiles, "threshold": threshold, "result_count": len(results) if results else 0}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except ImportError as e:
        return _error_response("ImportError", f"ChEMBL dependency missing: {e}", suggestion="pip install chembl-webresource-client")
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check SMILES string validity and threshold (0-100).")


def query_chembl_substructure_by_smiles(
    smiles: str,
    max_results: Optional[int] = None,
) -> str:
    """Query ChEMBL for molecules containing SMILES substructure. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        results = substructure_search(smiles, max_results=max_results)
        payload = {"smiles": smiles, "count": len(results) if results else 0, "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=_json_serializer)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"smiles": smiles, "result_count": len(results) if results else 0}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except ImportError as e:
        return _error_response("ImportError", f"ChEMBL dependency missing: {e}", suggestion="pip install chembl-webresource-client")
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check SMILES substructure string validity.")


def query_chembl_drug_by_id(
    chembl_id: str,
    max_results: Optional[int] = None,
) -> str:
    """Query ChEMBL drug info by ChEMBL ID (drug, mechanisms, indications). Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        drug = get_drug(chembl_id)
        mechanisms = get_mechanisms(chembl_id, max_results=max_results) if drug else []
        indications = get_indications(chembl_id, max_results=max_results) if drug else []
        payload = {"chembl_id": chembl_id, "drug": drug, "mechanisms": mechanisms, "indications": indications}
        content = json.dumps(payload, ensure_ascii=False, default=_json_serializer)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "chembl_id": chembl_id,
            "has_drug": drug is not None,
            "mechanism_count": len(mechanisms),
            "indication_count": len(indications),
        }
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except ImportError as e:
        return _error_response("ImportError", f"ChEMBL dependency missing: {e}", suggestion="pip install chembl-webresource-client")
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check ChEMBL ID and network connection.")


# ---------- download_chembl_*: save to file, return JSON with success + file_path ----------


def download_chembl_molecule_by_id(chembl_id: str, out_path: str) -> str:
    """Download ChEMBL molecule JSON by ChEMBL ID to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        mol = get_molecule(chembl_id)
        if mol is None:
            return _error_response("NotFound", f"Molecule {chembl_id} not found", suggestion="Check ChEMBL ID format (e.g. CHEMBL25).")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"chembl_id": chembl_id, "molecule": mol}, f, indent=2, default=_json_serializer)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"chembl_id": chembl_id}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except ImportError as e:
        return _error_response("ImportError", f"ChEMBL dependency missing: {e}", suggestion="pip install chembl-webresource-client")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check ChEMBL ID and output path.")


def download_chembl_similarity_by_smiles(
    smiles: str,
    out_path: str,
    threshold: int = 85,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL similarity search results to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        results = similarity_search(smiles, threshold=threshold, max_results=max_results)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"smiles": smiles, "threshold": threshold, "count": len(results) if results else 0, "results": results}, f, indent=2, default=_json_serializer)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"smiles": smiles, "threshold": threshold, "result_count": len(results) if results else 0}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except ImportError as e:
        return _error_response("ImportError", f"ChEMBL dependency missing: {e}", suggestion="pip install chembl-webresource-client")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check SMILES string and output path.")


def download_chembl_substructure_by_smiles(
    smiles: str,
    out_path: str,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL substructure search results to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        results = substructure_search(smiles, max_results=max_results)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"smiles": smiles, "count": len(results) if results else 0, "results": results}, f, indent=2, default=_json_serializer)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"smiles": smiles, "result_count": len(results) if results else 0}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except ImportError as e:
        return _error_response("ImportError", f"ChEMBL dependency missing: {e}", suggestion="pip install chembl-webresource-client")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check SMILES string and output path.")


def download_chembl_drug_by_id(
    chembl_id: str,
    out_path: str,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL drug info (drug, mechanisms, indications) to JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        drug = get_drug(chembl_id)
        mechanisms = get_mechanisms(chembl_id, max_results=max_results) if drug else []
        indications = get_indications(chembl_id, max_results=max_results) if drug else []
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"chembl_id": chembl_id, "drug": drug, "mechanisms": mechanisms, "indications": indications}, f, indent=2, default=_json_serializer)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "chembl_id": chembl_id,
            "has_drug": drug is not None,
            "mechanism_count": len(mechanisms),
            "indication_count": len(indications),
        }
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except ImportError as e:
        return _error_response("ImportError", f"ChEMBL dependency missing: {e}", suggestion="pip install chembl-webresource-client")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check ChEMBL ID and output path.")


__all__ = [
    "query_chembl_molecule_by_id",
    "query_chembl_similarity_by_smiles",
    "query_chembl_substructure_by_smiles",
    "query_chembl_drug_by_id",
    "download_chembl_molecule_by_id",
    "download_chembl_similarity_by_smiles",
    "download_chembl_substructure_by_smiles",
    "download_chembl_drug_by_id",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ChEMBL operations: query_chembl_* (return rich JSON) and download_chembl_* (return rich JSON with file_path)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_chembl_* and download_chembl_* samples; output under example/database/chembl",
    )
    parser.add_argument("--mol_id", type=str, default="CHEMBL25", help="ChEMBL molecule ID")
    parser.add_argument("--smiles", type=str, default="CC(=O)Oc1ccccc1C(=O)O", help="SMILES for similarity/substructure")
    parser.add_argument("--threshold", type=int, default=70, help="Similarity threshold 0-100")
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_dir = os.path.join("example", "database", "chembl")
    os.makedirs(out_dir, exist_ok=True)

    def _print_result(name: str, res: str) -> None:
        obj = json.loads(res)
        print(f"  {name}: status={obj.get('status')} ...")
        if obj.get("status") == "success":
            if obj.get("content") and len(obj["content"]) > 500:
                print(f"  (content_preview): {obj.get('content_preview', '')[:200]}...")
            elif obj.get("content"):
                print(f"  content: {obj['content'][:200]}...")
            if obj.get("file_info"):
                print(f"  file_info: {obj['file_info']}")
            if not obj.get("content") and obj.get("content_preview"):
                print(f"  content_preview: {obj['content_preview'][:300]}...")
            if obj.get("biological_metadata"):
                print(f"  biological_metadata: {obj['biological_metadata']}")
            if obj.get("execution_context"):
                print(f"  execution_context: {obj['execution_context']}")
        if obj.get("status") == "error" and obj.get("error"):
            print(f"  error: {obj['error']}")

    print("=== query_chembl_* (return rich JSON: status, content, content_preview, biological_metadata, execution_context) ===")
    print("  query_chembl_molecule_by_id(...)")
    _print_result("query_chembl_molecule_by_id", query_chembl_molecule_by_id(args.mol_id))

    print("  query_chembl_similarity_by_smiles(...)")
    _print_result("query_chembl_similarity_by_smiles", query_chembl_similarity_by_smiles(args.smiles, threshold=args.threshold))

    print("  query_chembl_substructure_by_smiles(...)")
    _print_result("query_chembl_substructure_by_smiles", query_chembl_substructure_by_smiles("c1ccccc1"))

    print("  query_chembl_drug_by_id(...)")
    _print_result("query_chembl_drug_by_id", query_chembl_drug_by_id(args.mol_id))

    print("=== download_chembl_* (return rich JSON: status, file_info, content_preview, biological_metadata, execution_context) ===")
    for name, res in [
        ("download_chembl_molecule_by_id", download_chembl_molecule_by_id(args.mol_id, os.path.join(out_dir, "chembl_molecule_sample.json"))),
        ("download_chembl_similarity_by_smiles", download_chembl_similarity_by_smiles(args.smiles, os.path.join(out_dir, "chembl_similarity_sample.json"), threshold=args.threshold)),
        ("download_chembl_substructure_by_smiles", download_chembl_substructure_by_smiles("c1ccccc1", os.path.join(out_dir, "chembl_substructure_sample.json"))),
        ("download_chembl_drug_by_id", download_chembl_drug_by_id(args.mol_id, os.path.join(out_dir, "chembl_drug_sample.json"))),
    ]:
        _print_result(name, res)

    print(f"Done. Output under {out_dir}")
