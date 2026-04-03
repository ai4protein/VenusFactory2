"""
STRING DB operations: query and download for all STRING endpoints.
Returns: rich JSON format containing status, file_info (download) or content (query), biological_metadata, etc.
Endpoints: version, map_ids, network, network_image, interaction_partners, enrichment, ppi_enrichment, homology.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.tools.path_sanitizer import to_client_file_path

try:
    from .string_map_ids import string_map_ids
    from .string_network import string_network
    from .string_network_image import string_network_image
    from .string_interaction_partners import string_interaction_partners
    from .string_enrichment import string_enrichment
    from .string_ppi_enrichment import string_ppi_enrichment
    from .string_homology import string_homology
    from .string_version import string_version
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "string" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.string.string_map_ids import string_map_ids
    from src.tools.database.string.string_network import string_network
    from src.tools.database.string.string_network_image import string_network_image
    from src.tools.database.string.string_interaction_partners import string_interaction_partners
    from src.tools.database.string.string_enrichment import string_enrichment
    from src.tools.database.string.string_ppi_enrichment import string_ppi_enrichment
    from src.tools.database.string.string_homology import string_homology
    from src.tools.database.string.string_version import string_version


_PREVIEW_LEN = 500
_SOURCE_STRING = "STRING"
_CALLER_IDENTITY = "claude_scientific_skills"


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
    source: str = _SOURCE_STRING,
) -> str:
    """Build JSON for download success: status, file_info, content_preview, biological_metadata, execution_context."""
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "txt"
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else file_path),
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
    source: str = _SOURCE_STRING,
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
    return json.dumps(out, ensure_ascii=False)


def _read_preview(path: str, max_chars: int = _PREVIEW_LEN) -> str:
    """Read first max_chars from file for content_preview."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


# ---------- version (meta/info) ----------
def query_string_version() -> str:
    """Query STRING DB version. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = string_version()
        if data.startswith("Error:"):
            return _error_response("QueryError", data, suggestion="Check network connectivity.")
        
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return _query_success_response(
            content=data,
            content_preview=data,
            biological_metadata={},
            query_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_string_version(out_dir: str, filename: str = "version.txt") -> str:
    """Download STRING version to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        data = string_version()
        if data.startswith("Error:"):
            return _error_response("DownloadError", data, suggestion="Check network connectivity.")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return _download_success_response(
            file_path=out_path,
            content_preview=_read_preview(out_path),
            biological_metadata={},
            download_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("DownloadError", str(e))


# ---------- map_ids ----------
def query_string_map_ids(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    limit: int = 1,
    echo_query: int = 1,
) -> str:
    """Map protein names/IDs to STRING IDs. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = string_map_ids(
            identifiers, 
            species=species, 
            limit=limit, 
            echo_query=echo_query, 
            caller_identity=_CALLER_IDENTITY
        )
        if data.startswith("Error:"):
            return _error_response("QueryError", data, suggestion="Check identifiers and species ID.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "limit": limit
        }
        return _query_success_response(
            content=data,
            content_preview=data,
            biological_metadata=meta,
            query_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_string_map_ids(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    limit: int = 1,
    echo_query: int = 1,
    filename: str = "map_ids.tsv",
) -> str:
    """Download map_ids result to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        data = string_map_ids(
            identifiers, 
            species=species, 
            limit=limit, 
            echo_query=echo_query, 
            caller_identity=_CALLER_IDENTITY
        )
        if data.startswith("Error:"):
            return _error_response("DownloadError", data, suggestion="Check identifiers and species ID.")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "limit": limit
        }
        return _download_success_response(
            file_path=out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("DownloadError", str(e))


# ---------- network ----------
def query_string_network(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    network_type: str = "functional",
    add_nodes: int = 0,
) -> str:
    """Get PPI network (TSV). Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = string_network(
            identifiers,
            species=species,
            required_score=required_score,
            network_type=network_type,
            add_nodes=add_nodes,
            caller_identity=_CALLER_IDENTITY,
        )
        if data.startswith("Error:"):
            return _error_response("QueryError", data, suggestion="Check identifiers, species ID, and requested parameters.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "required_score": required_score,
            "network_type": network_type,
            "add_nodes": add_nodes
        }
        return _query_success_response(
            content=data,
            content_preview=data,
            biological_metadata=meta,
            query_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_string_network(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    network_type: str = "functional",
    add_nodes: int = 0,
    filename: str = "network.tsv",
) -> str:
    """Download network TSV to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        data = string_network(
            identifiers,
            species=species,
            required_score=required_score,
            network_type=network_type,
            add_nodes=add_nodes,
            caller_identity=_CALLER_IDENTITY,
        )
        if data.startswith("Error:"):
            return _error_response("DownloadError", data, suggestion="Check identifiers, species ID, and requested parameters.")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "required_score": required_score,
            "network_type": network_type,
            "add_nodes": add_nodes
        }
        return _download_success_response(
            file_path=out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("DownloadError", str(e))


# ---------- network_image ----------
def query_string_network_image(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    network_flavor: str = "evidence",
    add_nodes: int = 0,
) -> str:
    """Get network as PNG. Since it's binary, this endpoint wraps it in a JSON with base64 content, or returns error."""
    import base64
    t0 = time.perf_counter()
    try:
        data = string_network_image(
            identifiers,
            species=species,
            required_score=required_score,
            network_flavor=network_flavor,
            add_nodes=add_nodes,
            caller_identity=_CALLER_IDENTITY,
        )
        if isinstance(data, bytes) and data.startswith(b"Error:"):
            return _error_response("QueryError", data.decode("utf-8", errors="replace"), suggestion="Check params")
            
        content_b64 = base64.b64encode(data).decode('ascii')
        
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "required_score": required_score,
            "network_flavor": network_flavor,
            "add_nodes": add_nodes,
            "image_format": "png"
        }
        
        return _query_success_response(
            content=content_b64,
            content_preview="[BASE64-ENCODED PNG DATA]",
            biological_metadata=meta,
            query_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_string_network_image(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    network_flavor: str = "evidence",
    add_nodes: int = 0,
    filename: str = "network.png",
) -> str:
    """Download network PNG to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        data = string_network_image(
            identifiers,
            species=species,
            required_score=required_score,
            network_flavor=network_flavor,
            add_nodes=add_nodes,
            caller_identity=_CALLER_IDENTITY,
        )
        
        if isinstance(data, bytes) and data.startswith(b"Error:"):
            return _error_response("DownloadError", data.decode("utf-8", errors="replace"), suggestion="Check connection or parameters.")
            
        with open(out_path, "wb") as f:
            f.write(data)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "required_score": required_score,
            "network_flavor": network_flavor,
            "add_nodes": add_nodes
        }
        return _download_success_response(
            file_path=out_path,
            content_preview=f"[PNG Image: {len(data)} bytes]",
            biological_metadata=meta,
            download_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("DownloadError", str(e))


# ---------- interaction_partners ----------
def query_string_interaction_partners(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    limit: int = 10,
) -> str:
    """Get interaction partners (TSV). Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = string_interaction_partners(
            identifiers,
            species=species,
            required_score=required_score,
            limit=limit,
            caller_identity=_CALLER_IDENTITY,
        )
        if data.startswith("Error:"):
            return _error_response("QueryError", data, suggestion="Check identifiers and species.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "required_score": required_score,
            "limit": limit
        }
        return _query_success_response(
            content=data,
            content_preview=data,
            biological_metadata=meta,
            query_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_string_interaction_partners(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    limit: int = 10,
    filename: str = "interaction_partners.tsv",
) -> str:
    """Download interaction partners TSV to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        data = string_interaction_partners(
            identifiers,
            species=species,
            required_score=required_score,
            limit=limit,
            caller_identity=_CALLER_IDENTITY,
        )
        if data.startswith("Error:"):
            return _error_response("DownloadError", data, suggestion="Check connection or parameters.")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "required_score": required_score,
            "limit": limit
        }
        return _download_success_response(
            file_path=out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("DownloadError", str(e))


# ---------- enrichment ----------
def query_string_enrichment(
    identifiers: Union[str, List[str]],
    species: int = 9606,
) -> str:
    """Functional enrichment (TSV). Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = string_enrichment(
            identifiers, 
            species=species, 
            caller_identity=_CALLER_IDENTITY
        )
        if data.startswith("Error:"):
            return _error_response("QueryError", data, suggestion="Check identifiers and species.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species
        }
        return _query_success_response(
            content=data,
            content_preview=data,
            biological_metadata=meta,
            query_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_string_enrichment(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    filename: str = "enrichment.tsv",
) -> str:
    """Download enrichment TSV to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        data = string_enrichment(
            identifiers, 
            species=species, 
            caller_identity=_CALLER_IDENTITY
        )
        if data.startswith("Error:"):
            return _error_response("DownloadError", data, suggestion="Check connection or parameters.")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species
        }
        return _download_success_response(
            file_path=out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("DownloadError", str(e))


# ---------- ppi_enrichment ----------
def query_string_ppi_enrichment(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
) -> str:
    """PPI enrichment (JSON array string). Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = string_ppi_enrichment(
            identifiers,
            species=species,
            required_score=required_score,
            caller_identity=_CALLER_IDENTITY,
        )
        if data.startswith("Error:"):
            return _error_response("QueryError", data, suggestion="Check identifiers and species.")
            
        # Optional: re-parse data if it's JSON to make it an object instead of string in our wrapper
        parsed_content = json.loads(data) if data.strip().startswith("[") or data.strip().startswith("{") else data
        
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "required_score": required_score
        }
        
        # We store the raw string in content for consistency with other methods, 
        # or the parsed object - let's stick to raw string to avoid double encoding issues
        # Or wait, _query_success_response expects string content, so let's pass data.
        return _query_success_response(
            content=data,
            content_preview=data,
            biological_metadata=meta,
            query_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_string_ppi_enrichment(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    required_score: int = 400,
    filename: str = "ppi_enrichment.json",
) -> str:
    """Download ppi_enrichment JSON to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        data = string_ppi_enrichment(
            identifiers,
            species=species,
            required_score=required_score,
            caller_identity=_CALLER_IDENTITY,
        )
        if data.startswith("Error:"):
            return _error_response("DownloadError", data, suggestion="Check connection or parameters.")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species,
            "required_score": required_score
        }
        return _download_success_response(
            file_path=out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("DownloadError", str(e))


# ---------- homology ----------
def query_string_homology(
    identifiers: Union[str, List[str]],
    species: int = 9606,
) -> str:
    """Homology scores (TSV). Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = string_homology(
            identifiers, 
            species=species, 
            caller_identity=_CALLER_IDENTITY
        )
        if data.startswith("Error:"):
            return _error_response("QueryError", data, suggestion="Check identifiers and species.")
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species
        }
        return _query_success_response(
            content=data,
            content_preview=data,
            biological_metadata=meta,
            query_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("QueryError", str(e))


def download_string_homology(
    identifiers: Union[str, List[str]],
    out_dir: str,
    species: int = 9606,
    filename: str = "homology.tsv",
) -> str:
    """Download homology TSV to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        data = string_homology(
            identifiers, 
            species=species, 
            caller_identity=_CALLER_IDENTITY
        )
        if data.startswith("Error:"):
            return _error_response("DownloadError", data, suggestion="Check connection or parameters.")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "identifiers": identifiers if isinstance(identifiers, list) else [identifiers],
            "species": species
        }
        return _download_success_response(
            file_path=out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms
        )
    except Exception as e:
        return _error_response("DownloadError", str(e))


__all__ = [
    "query_string_version",
    "download_string_version",
    "query_string_map_ids",
    "download_string_map_ids",
    "query_string_network",
    "download_string_network",
    "query_string_network_image",
    "download_string_network_image",
    "query_string_interaction_partners",
    "download_string_interaction_partners",
    "query_string_enrichment",
    "download_string_enrichment",
    "query_string_ppi_enrichment",
    "download_string_ppi_enrichment",
    "query_string_homology",
    "download_string_homology",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STRING DB: query/download all endpoints; use --test to run sample and save under example/database/string")
    parser.add_argument("--test", action="store_true", help="Run tests; output under example/database/string")
    args = parser.parse_args()

    if args.test:
        out_dir = os.path.join("example", "database", "string")
        os.makedirs(out_dir, exist_ok=True)
        meta_dir = os.path.join(out_dir, "metadata")
        test_id = "P43403"
        species = 9606

        def _print_res(name: str, res: str) -> None:
            obj = json.loads(res)
            print(f"  {name}: status={obj.get('status')} ...")
            if obj.get("status") == "success":
                if obj.get("file_info"):
                    print(f"  file_info: {obj.get('file_info')}")
                if "content_preview" in obj:
                    print(f"  content_preview: {str(obj.get('content_preview', ''))[:100]}...")
                print(f"  execution_context: {obj.get('execution_context')}")
            else:
                print(f"  error: {obj.get('error')}")

        print("Testing query_string_version / download_string_version(...)")
        _print_res("query_string_version", query_string_version())
        _print_res("download_string_version", download_string_version(meta_dir, "version.txt"))

        print("Testing query_string_map_ids / download_string_map_ids(...)")
        _print_res("query_string_map_ids", query_string_map_ids(test_id, species=species))
        _print_res("download_string_map_ids", download_string_map_ids(test_id, meta_dir, species=species, filename="map_ids.tsv"))

        print("Testing query_string_network / download_string_network(...)")
        _print_res("query_string_network", query_string_network(test_id, species=species))
        _print_res("download_string_network", download_string_network(test_id, out_dir, species=species, filename="network.tsv"))

        print("Testing query_string_network_image / download_string_network_image(...)")
        _print_res("query_string_network_image", query_string_network_image(test_id, species=species))
        _print_res("download_string_network_image", download_string_network_image(test_id, out_dir, species=species, filename="network.png"))

        print("Testing query_string_interaction_partners / download_string_interaction_partners(...)")
        _print_res("query_string_interaction_partners", query_string_interaction_partners(test_id, species=species, limit=5))
        _print_res("download_string_interaction_partners", download_string_interaction_partners(test_id, out_dir, species=species, limit=5, filename="interaction_partners.tsv"))

        print("Testing query_string_enrichment / download_string_enrichment(...)")
        _print_res("query_string_enrichment", query_string_enrichment(test_id, species=species))
        _print_res("download_string_enrichment", download_string_enrichment(test_id, out_dir, species=species, filename="enrichment.tsv"))

        print("Testing query_string_ppi_enrichment / download_string_ppi_enrichment(...)")
        _print_res("query_string_ppi_enrichment", query_string_ppi_enrichment(test_id, species=species))
        _print_res("download_string_ppi_enrichment", download_string_ppi_enrichment(test_id, out_dir, species=species, filename="ppi_enrichment.json"))

        print("Testing query_string_homology / download_string_homology(...)")
        _print_res("query_string_homology", query_string_homology(test_id, species=species))
        _print_res("download_string_homology", download_string_homology(test_id, out_dir, species=species, filename="homology.tsv"))

        print(f"Done. Output under {out_dir}")
        sys.exit(0)

    parser.print_help()

