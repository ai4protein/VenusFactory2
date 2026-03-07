"""
BRENDA operations: single exit for query and download; both return rich JSON.

Success: status, file_info (download) or content (query), content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .brenda_client import get_km_values, get_reactions
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "brenda" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.brenda.brenda_client import get_km_values, get_reactions

try:
    from .brenda_queries import (
        compare_across_organisms,
        export_kinetic_data,
        get_environmental_parameters,
        search_enzymes_by_substrate,
    )
except ImportError:
    from src.tools.database.brenda.brenda_queries import (  # noqa: E501
        compare_across_organisms,
        export_kinetic_data,
        get_environmental_parameters,
        search_enzymes_by_substrate,
    )

try:
    from .enzyme_pathway_builder import (
        find_pathway_for_product,
        generate_pathway_report,
    )
except ImportError:
    from src.tools.database.brenda.enzyme_pathway_builder import (  # noqa: E501
        find_pathway_for_product,
        generate_pathway_report,
    )


_PREVIEW_LEN = 500
_SOURCE_BRENDA = "BRENDA"


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
    source: str = _SOURCE_BRENDA,
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
        "content_preview": (content_preview or "")[: _PREVIEW_LEN],
        "biological_metadata": biological_metadata or {},
        "execution_context": {"download_time_ms": download_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


def _query_success_response(
    content: str,
    content_preview: Optional[str] = None,
    biological_metadata: Optional[Dict[str, Any]] = None,
    query_time_ms: int = 0,
    source: str = _SOURCE_BRENDA,
) -> str:
    """Build JSON for query success: status, content, content_preview, biological_metadata, execution_context."""
    preview = (content_preview or content or "")[: _PREVIEW_LEN]
    out: Dict[str, Any] = {
        "status": "success",
        "content": content,
        "content_preview": preview,
        "biological_metadata": biological_metadata or {},
        "execution_context": {"query_time_ms": query_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


# ---------- query_*: return rich JSON (status, content, content_preview, biological_metadata, execution_context) ----------


def query_brenda_km_values_by_ec_number(
    ec_number: str,
    organism: str = "*",
    substrate: str = "*",
) -> str:
    """Query Km values by EC number. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        entries = get_km_values(ec_number, organism=organism, substrate=substrate)
        payload = {"ec_number": ec_number, "count": len(entries), "entries": entries or []}
        content = json.dumps(payload, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number, "organism": organism, "substrate": substrate, "entry_count": len(entries)}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check EC number and BRENDA_EMAIL/BRENDA_PASSWORD.")


def query_brenda_reactions_by_ec_number(
    ec_number: str,
    organism: str = "*",
) -> str:
    """Query reaction equations by EC number. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        entries = get_reactions(ec_number, organism=organism)
        payload = {"ec_number": ec_number, "count": len(entries), "entries": entries or []}
        content = json.dumps(payload, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number, "organism": organism, "entry_count": len(entries)}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check EC number and BRENDA credentials.")


def query_brenda_enzymes_by_substrate(substrate: str, limit: int = 50) -> str:
    """Query enzymes that act on a substrate. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = search_enzymes_by_substrate(substrate, limit=limit)
        content = json.dumps({"substrate": substrate, "count": len(data), "enzymes": data}, default=str, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"substrate": substrate, "limit": limit, "enzyme_count": len(data)}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check substrate name and BRENDA credentials.")


def query_brenda_compare_organisms_by_ec_number(ec_number: str, organisms: List[str]) -> str:
    """Compare enzyme parameters across organisms by EC number. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = compare_across_organisms(ec_number, organisms)
        content = json.dumps({"ec_number": ec_number, "organisms": organisms, "comparison": data}, default=str, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number, "organisms": organisms, "organism_count": len(organisms)}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check EC number and organism names.")


def query_brenda_environmental_parameters_by_ec_number(ec_number: str) -> str:
    """Query environmental parameters (pH, temperature) by EC number. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = get_environmental_parameters(ec_number)
        content = json.dumps(data, default=str, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check EC number and BRENDA credentials.")


def query_brenda_pathway_by_product(
    product: str,
    max_steps: int = 3,
    starting_materials: Optional[List[str]] = None,
) -> str:
    """Query pathway by product. Returns rich JSON: status, content, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = find_pathway_for_product(product, max_steps=max_steps, starting_materials=starting_materials)
        content = json.dumps(data, default=str, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"product": product, "max_steps": max_steps, "steps_count": len(data.get("steps", [])) if isinstance(data, dict) else 0}
        return _query_success_response(content, content_preview=content, biological_metadata=meta, query_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("QueryError", str(e), suggestion="Check product name or try different max_steps.")


# ---------- download_*: save to file, return JSON with success + file_path ----------


def _read_preview(path: str, max_chars: int = _PREVIEW_LEN) -> str:
    """Read first max_chars from file for content_preview."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def download_brenda_km_values_by_ec_number(
    ec_number: str,
    out_path: str,
    organism: str = "*",
    substrate: str = "*",
) -> str:
    """Download Km values by EC number to a text/JSON file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        entries = get_km_values(ec_number, organism=organism, substrate=substrate)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.endswith(".json"):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"ec_number": ec_number, "count": len(entries), "entries": entries}, f, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(entries) if entries else "")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number, "organism": organism, "substrate": substrate, "entry_count": len(entries)}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check EC number and BRENDA_EMAIL/BRENDA_PASSWORD.")


def download_brenda_reactions_by_ec_number(ec_number: str, out_path: str, organism: str = "*") -> str:
    """Download reaction entries by EC number to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        entries = get_reactions(ec_number, organism=organism)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.endswith(".json"):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"ec_number": ec_number, "count": len(entries), "entries": entries}, f, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(entries) if entries else "")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number, "organism": organism, "entry_count": len(entries)}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check EC number and BRENDA credentials.")


def download_brenda_enzymes_by_substrate(substrate: str, out_path: str, limit: int = 50) -> str:
    """Download enzyme-by-substrate search results to JSON. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = search_enzymes_by_substrate(substrate, limit=limit)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"substrate": substrate, "count": len(data), "enzymes": data}, f, indent=2, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"substrate": substrate, "limit": limit, "enzyme_count": len(data)}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check substrate name and BRENDA credentials.")


def download_brenda_compare_organisms_by_ec_number(ec_number: str, organisms: List[str], out_path: str) -> str:
    """Download organism comparison by EC number to JSON. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = compare_across_organisms(ec_number, organisms)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"ec_number": ec_number, "organisms": organisms, "comparison": data}, f, indent=2, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number, "organisms": organisms, "organism_count": len(organisms)}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check EC number and organism names.")


def download_brenda_environmental_parameters_by_ec_number(ec_number: str, out_path: str) -> str:
    """Download environmental parameters by EC number to JSON. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        data = get_environmental_parameters(ec_number)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check EC number and BRENDA credentials.")


def download_brenda_kinetic_data_by_ec_number(ec_number: str, out_path: str, format: str = "json") -> str:
    """Download kinetic data by EC number to file (csv/json). Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        export_kinetic_data(ec_number, format=format, filename=out_path)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"ec_number": ec_number, "format": format}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check EC number and BRENDA credentials.")


def download_brenda_pathway_report(pathway: Dict[str, Any], out_path: str) -> str:
    """Generate and save pathway report to file. Returns rich JSON: status, file_info, content_preview, biological_metadata, execution_context."""
    t0 = time.perf_counter()
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        generate_pathway_report(pathway, filename=out_path)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {"target": pathway.get("target") or pathway.get("product"), "steps": len(pathway.get("steps", []))}
        return _download_success_response(out_path, content_preview=_read_preview(out_path), biological_metadata=meta, download_time_ms=elapsed_ms)
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Ensure pathway dict is valid (e.g. from query_brenda_pathway_by_product).")


__all__ = [
    "query_brenda_km_values_by_ec_number",
    "query_brenda_reactions_by_ec_number",
    "query_brenda_enzymes_by_substrate",
    "query_brenda_compare_organisms_by_ec_number",
    "query_brenda_environmental_parameters_by_ec_number",
    "query_brenda_pathway_by_product",
    "download_brenda_km_values_by_ec_number",
    "download_brenda_reactions_by_ec_number",
    "download_brenda_enzymes_by_substrate",
    "download_brenda_compare_organisms_by_ec_number",
    "download_brenda_environmental_parameters_by_ec_number",
    "download_brenda_kinetic_data_by_ec_number",
    "download_brenda_pathway_report",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BRENDA operations: query_* (return JSON with content) and download_* (return JSON with file_path)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run query_* and download_* samples; output under example/database/brenda",
    )
    parser.add_argument("--ec", type=str, default="1.1.1.1", help="EC number for test. Default 1.1.1.1.")
    parser.add_argument("--substrate", type=str, default="glucose", help="Substrate for enzyme search. Default glucose.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="example/database/brenda",
        help="Output directory. Default example/database/brenda.",
    )
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    ec = args.ec
    substrate = args.substrate
    organisms = ["Escherichia coli", "Saccharomyces cerevisiae", "Homo sapiens"]

    def _print_query(name: str, res: str) -> None:
        obj = json.loads(res)
        print(f"  {name}: status={obj.get('status')} ...")
        if obj.get("status") == "success":
            if obj.get("content") and len(obj["content"]) > 500:
                print(f"  (content_preview): {obj.get('content_preview', '')[:200]}...")
            elif obj.get("content"):
                print(f"  content: {obj['content'][:200]}...")
            if obj.get("execution_context"):
                print(f"  execution_context: {obj['execution_context']}")
        if obj.get("status") == "error" and obj.get("error"):
            print(f"  error: {obj['error']}")

    print("=== query_* (return rich JSON: status, content, content_preview, biological_metadata, execution_context) ===")
    res_km = query_brenda_km_values_by_ec_number(ec)
    _print_query("query_brenda_km_values_by_ec_number(...)", res_km)
    with open(os.path.join(out_dir, "query_km_sample.txt"), "w", encoding="utf-8") as f:
        f.write(res_km)
    print(f"  full JSON saved to {os.path.join(out_dir, 'query_km_sample.txt')}")

    res_reactions = query_brenda_reactions_by_ec_number(ec)
    _print_query("query_brenda_reactions_by_ec_number(...)", res_reactions)
    with open(os.path.join(out_dir, "query_reactions_sample.txt"), "w", encoding="utf-8") as f:
        f.write(res_reactions)

    res_compare = query_brenda_compare_organisms_by_ec_number(ec, organisms)
    _print_query("query_brenda_compare_organisms_by_ec_number(...)", res_compare)

    res_env = query_brenda_environmental_parameters_by_ec_number(ec)
    _print_query("query_brenda_environmental_parameters_by_ec_number(...)", res_env)

    res_pathway = query_brenda_pathway_by_product("lactate")
    _print_query("query_brenda_pathway_by_product(...)", res_pathway)

    print("=== download_* (return rich JSON: status, file_info, content_preview, biological_metadata, execution_context) ===")
    for name, res in [
        ("download_brenda_km_values_by_ec_number", download_brenda_km_values_by_ec_number(ec, os.path.join(out_dir, "brenda_km_sample.json"))),
        ("download_brenda_reactions_by_ec_number", download_brenda_reactions_by_ec_number(ec, os.path.join(out_dir, "brenda_reactions_sample.txt"))),
        (
            "download_brenda_compare_organisms_by_ec_number",
            download_brenda_compare_organisms_by_ec_number(ec, organisms, os.path.join(out_dir, "brenda_compare_organisms_sample.json")),
        ),
        (
            "download_brenda_environmental_parameters_by_ec_number",
            download_brenda_environmental_parameters_by_ec_number(ec, os.path.join(out_dir, "brenda_environmental_sample.json")),
        ),
        (
            "download_brenda_enzymes_by_substrate",
            download_brenda_enzymes_by_substrate(substrate, os.path.join(out_dir, "brenda_enzymes_by_substrate_sample.json")),
        ),
    ]:
        dl_obj = json.loads(res)
        print(f"  {name}: {dl_obj}")

    pathway_obj = json.loads(res_pathway)
    if pathway_obj.get("status") == "success" and pathway_obj.get("content"):
        try:
            pathway_data = json.loads(pathway_obj["content"])
            if isinstance(pathway_data, dict) and ("steps" in pathway_data or "target" in pathway_data):
                dl_pathway = download_brenda_pathway_report(
                    pathway_data, os.path.join(out_dir, "brenda_pathway_report_sample.txt")
                )
                print(f"  download_brenda_pathway_report: {json.loads(dl_pathway)}")
        except (json.JSONDecodeError, TypeError):
            pass

    print(f"Done. Output under {out_dir}")
