"""
BRENDA operations: single exit for query (return JSON with content) and download (save to file, return JSON).

All public functions return JSON string with:
- query_*: {"success": bool, "content": str}  (content is result data as JSON text)
- download_*: {"success": bool, "file_path": str or null}
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .brenda_client import get_km_values, get_reactions
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "brenda" and str(_dir.parents[4]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[4]))
    from src.tools.search.database.brenda.brenda_client import get_km_values, get_reactions

try:
    from .brenda_queries import (
        compare_across_organisms,
        export_kinetic_data,
        get_environmental_parameters,
        search_enzymes_by_substrate,
    )
except ImportError:
    from src.tools.search.database.brenda.brenda_queries import (  # noqa: E501
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
    from src.tools.search.database.brenda.enzyme_pathway_builder import (  # noqa: E501
        find_pathway_for_product,
        generate_pathway_report,
    )


def _query_result(success: bool, content: Optional[str] = None, error: Optional[str] = None) -> str:
    """Build JSON for query_* result: success, content (result data as JSON text)."""
    out: Dict[str, Any] = {"success": success, "content": content}
    if error is not None:
        out["error"] = error
    return json.dumps(out, ensure_ascii=False)


def _download_result(success: bool, file_path: Optional[str] = None, error: Optional[str] = None) -> str:
    """Build JSON for download_* result: success, file_path."""
    out: Dict[str, Any] = {"success": success, "file_path": file_path}
    if error is not None:
        out["error"] = error
    return json.dumps(out, ensure_ascii=False)


# ---------- query_*: return JSON with success + content ----------


def query_brenda_km_values_by_ec_number(
    ec_number: str,
    organism: str = "*",
    substrate: str = "*",
) -> str:
    """Query Km values by EC number. Returns JSON: {success, content} (content is result data)."""
    try:
        entries = get_km_values(ec_number, organism=organism, substrate=substrate)
        payload = {"ec_number": ec_number, "count": len(entries), "entries": entries or []}
        return _query_result(True, content=json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        return _query_result(False, content=None, error=str(e))


def query_brenda_reactions_by_ec_number(
    ec_number: str,
    organism: str = "*",
) -> str:
    """Query reaction equations by EC number. Returns JSON: {success, content} (content is result data)."""
    try:
        entries = get_reactions(ec_number, organism=organism)
        payload = {"ec_number": ec_number, "count": len(entries), "entries": entries or []}
        return _query_result(True, content=json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        return _query_result(False, content=None, error=str(e))


def query_brenda_enzymes_by_substrate(substrate: str, limit: int = 50) -> str:
    """Query enzymes that act on a substrate. Returns JSON: {success, content} (content is result data)."""
    try:
        data = search_enzymes_by_substrate(substrate, limit=limit)
        content = json.dumps({"substrate": substrate, "count": len(data), "enzymes": data}, default=str, ensure_ascii=False)
        return _query_result(True, content=content)
    except Exception as e:
        return _query_result(False, content=None, error=str(e))


def query_brenda_compare_organisms_by_ec_number(ec_number: str, organisms: List[str]) -> str:
    """Compare enzyme parameters across organisms by EC number. Returns JSON: {success, content} (content is result data)."""
    try:
        data = compare_across_organisms(ec_number, organisms)
        content = json.dumps({"ec_number": ec_number, "organisms": organisms, "comparison": data}, default=str, ensure_ascii=False)
        return _query_result(True, content=content)
    except Exception as e:
        return _query_result(False, content=None, error=str(e))


def query_brenda_environmental_parameters_by_ec_number(ec_number: str) -> str:
    """Query environmental parameters (pH, temperature) by EC number. Returns JSON: {success, content}."""
    try:
        data = get_environmental_parameters(ec_number)
        content = json.dumps(data, default=str, ensure_ascii=False)
        return _query_result(True, content=content)
    except Exception as e:
        return _query_result(False, content=None, error=str(e))


def query_brenda_pathway_by_product(
    product: str,
    max_steps: int = 3,
    starting_materials: Optional[List[str]] = None,
) -> str:
    """Query pathway by product. Returns JSON: {success, content} (content is result data)."""
    try:
        data = find_pathway_for_product(product, max_steps=max_steps, starting_materials=starting_materials)
        content = json.dumps(data, default=str, ensure_ascii=False)
        return _query_result(True, content=content)
    except Exception as e:
        return _query_result(False, content=None, error=str(e))


# ---------- download_*: save to file, return JSON with success + file_path ----------


def download_brenda_km_values_by_ec_number(
    ec_number: str,
    out_path: str,
    organism: str = "*",
    substrate: str = "*",
) -> str:
    """Download Km values by EC number to a text/JSON file. Returns JSON: {success, file_path}."""
    try:
        entries = get_km_values(ec_number, organism=organism, substrate=substrate)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.endswith(".json"):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"ec_number": ec_number, "count": len(entries), "entries": entries}, f, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(entries) if entries else "")
        return _download_result(True, file_path=out_path)
    except Exception as e:
        return _download_result(False, file_path=None, error=str(e))


def download_brenda_reactions_by_ec_number(ec_number: str, out_path: str, organism: str = "*") -> str:
    """Download reaction entries by EC number to file. Returns JSON: {success, file_path}."""
    try:
        entries = get_reactions(ec_number, organism=organism)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.endswith(".json"):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"ec_number": ec_number, "count": len(entries), "entries": entries}, f, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(entries) if entries else "")
        return _download_result(True, file_path=out_path)
    except Exception as e:
        return _download_result(False, file_path=None, error=str(e))


def download_brenda_enzymes_by_substrate(substrate: str, out_path: str, limit: int = 50) -> str:
    """Download enzyme-by-substrate search results to JSON. Returns JSON: {success, file_path}."""
    try:
        data = search_enzymes_by_substrate(substrate, limit=limit)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"substrate": substrate, "count": len(data), "enzymes": data}, f, indent=2, default=str)
        return _download_result(True, file_path=out_path)
    except Exception as e:
        return _download_result(False, file_path=None, error=str(e))


def download_brenda_compare_organisms_by_ec_number(ec_number: str, organisms: List[str], out_path: str) -> str:
    """Download organism comparison by EC number to JSON. Returns JSON: {success, file_path}."""
    try:
        data = compare_across_organisms(ec_number, organisms)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"ec_number": ec_number, "organisms": organisms, "comparison": data}, f, indent=2, default=str)
        return _download_result(True, file_path=out_path)
    except Exception as e:
        return _download_result(False, file_path=None, error=str(e))


def download_brenda_environmental_parameters_by_ec_number(ec_number: str, out_path: str) -> str:
    """Download environmental parameters by EC number to JSON. Returns JSON: {success, file_path}."""
    try:
        data = get_environmental_parameters(ec_number)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return _download_result(True, file_path=out_path)
    except Exception as e:
        return _download_result(False, file_path=None, error=str(e))


def download_brenda_kinetic_data_by_ec_number(ec_number: str, out_path: str, format: str = "json") -> str:
    """Download kinetic data by EC number to file (csv/json). Returns JSON: {success, file_path}."""
    try:
        export_kinetic_data(ec_number, format=format, filename=out_path)
        return _download_result(True, file_path=out_path)
    except Exception as e:
        return _download_result(False, file_path=None, error=str(e))


def download_brenda_pathway_report(pathway: Dict[str, Any], out_path: str) -> str:
    """Generate and save pathway report to file. Returns JSON: {success, file_path}."""
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        generate_pathway_report(pathway, filename=out_path)
        return _download_result(True, file_path=out_path)
    except Exception as e:
        return _download_result(False, file_path=None, error=str(e))


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
        "--out_base",
        type=str,
        default="example/database/brenda",
        help="Output directory. Default example/database/brenda.",
    )
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)

    out_base = args.out_base
    os.makedirs(out_base, exist_ok=True)
    ec = args.ec
    substrate = args.substrate
    organisms = ["Escherichia coli", "Saccharomyces cerevisiae", "Homo sapiens"]

    def _print_query(name: str, res: str) -> None:
        obj = json.loads(res)
        print(f"  {name}: {res[:200]}..." if len(res) > 200 else f"  {name}: {res}")
        if obj.get("content") and len(obj["content"]) > 500:
            print(f"  (first 500 chars of content): {obj['content'][:500]}...")
        elif obj.get("content"):
            print(f"  content: {obj['content']}")
        if obj.get("error"):
            print(f"  error: {obj['error']}")

    print("=== query_* (return JSON: success + content) ===")
    res_km = query_brenda_km_values_by_ec_number(ec)
    _print_query("query_brenda_km_values_by_ec_number(...)", res_km)
    with open(os.path.join(out_base, "query_km_sample.txt"), "w", encoding="utf-8") as f:
        f.write(res_km)
    print(f"  full JSON saved to {os.path.join(out_base, 'query_km_sample.txt')}")

    res_reactions = query_brenda_reactions_by_ec_number(ec)
    _print_query("query_brenda_reactions_by_ec_number(...)", res_reactions)
    with open(os.path.join(out_base, "query_reactions_sample.txt"), "w", encoding="utf-8") as f:
        f.write(res_reactions)

    res_compare = query_brenda_compare_organisms_by_ec_number(ec, organisms)
    _print_query("query_brenda_compare_organisms_by_ec_number(...)", res_compare)

    res_env = query_brenda_environmental_parameters_by_ec_number(ec)
    _print_query("query_brenda_environmental_parameters_by_ec_number(...)", res_env)

    res_pathway = query_brenda_pathway_by_product("lactate")
    _print_query("query_brenda_pathway_by_product(...)", res_pathway)

    print("=== download_* (return JSON: success + file_path) ===")
    for name, res in [
        ("download_brenda_km_values_by_ec_number", download_brenda_km_values_by_ec_number(ec, os.path.join(out_base, "brenda_km_sample.json"))),
        ("download_brenda_reactions_by_ec_number", download_brenda_reactions_by_ec_number(ec, os.path.join(out_base, "brenda_reactions_sample.txt"))),
        (
            "download_brenda_compare_organisms_by_ec_number",
            download_brenda_compare_organisms_by_ec_number(ec, organisms, os.path.join(out_base, "brenda_compare_organisms_sample.json")),
        ),
        (
            "download_brenda_environmental_parameters_by_ec_number",
            download_brenda_environmental_parameters_by_ec_number(ec, os.path.join(out_base, "brenda_environmental_sample.json")),
        ),
        (
            "download_brenda_enzymes_by_substrate",
            download_brenda_enzymes_by_substrate(substrate, os.path.join(out_base, "brenda_enzymes_by_substrate_sample.json")),
        ),
    ]:
        dl_obj = json.loads(res)
        print(f"  {name}: {dl_obj}")

    pathway_obj = json.loads(res_pathway)
    if pathway_obj.get("success") and pathway_obj.get("content"):
        try:
            pathway_data = json.loads(pathway_obj["content"])
            if isinstance(pathway_data, dict) and ("steps" in pathway_data or "target" in pathway_data):
                dl_pathway = download_brenda_pathway_report(
                    pathway_data, os.path.join(out_base, "brenda_pathway_report_sample.txt")
                )
                print(f"  download_brenda_pathway_report: {json.loads(dl_pathway)}")
        except (json.JSONDecodeError, TypeError):
            pass

    print(f"Done. Output under {out_base}")
