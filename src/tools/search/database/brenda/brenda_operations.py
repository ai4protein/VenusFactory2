"""
BRENDA operations: query (return text) and download (save to file) only.

All public functions are either:
- query_*: return str (JSON or plain text)
- download_*: write to file and return status message str.
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


# ---------- query_*: return text ----------


def query_km_values(
    ec_number: str,
    organism: str = "*",
    substrate: str = "*",
) -> str:
    """Query Km values for an EC number. Returns JSON with entries or error."""
    try:
        entries = get_km_values(ec_number, organism=organism, substrate=substrate)
        if not entries:
            return json.dumps({"ec_number": ec_number, "count": 0, "entries": []})
        return json.dumps({"ec_number": ec_number, "count": len(entries), "entries": entries})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "ec_number": ec_number})


def query_reactions(
    ec_number: str,
    organism: str = "*",
) -> str:
    """Query reaction equations for an EC number. Returns JSON with entries or error."""
    try:
        entries = get_reactions(ec_number, organism=organism)
        if not entries:
            return json.dumps({"ec_number": ec_number, "count": 0, "entries": []})
        return json.dumps({"ec_number": ec_number, "count": len(entries), "entries": entries})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "ec_number": ec_number})


def query_enzymes_by_substrate(substrate: str, limit: int = 50) -> str:
    """Query enzymes that act on a substrate. Returns JSON."""
    try:
        data = search_enzymes_by_substrate(substrate, limit=limit)
        return json.dumps({"substrate": substrate, "count": len(data), "enzymes": data}, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "substrate": substrate})


def query_compare_organisms(ec_number: str, organisms: List[str]) -> str:
    """Compare enzyme parameters across organisms. Returns JSON."""
    try:
        data = compare_across_organisms(ec_number, organisms)
        return json.dumps({"ec_number": ec_number, "organisms": organisms, "comparison": data}, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "ec_number": ec_number})


def query_environmental_parameters(ec_number: str) -> str:
    """Query environmental parameters (pH, temperature) for an enzyme. Returns JSON."""
    try:
        data = get_environmental_parameters(ec_number)
        return json.dumps(data, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "ec_number": ec_number})


def query_pathway_for_product(
    product: str,
    max_steps: int = 3,
    starting_materials: Optional[List[str]] = None,
) -> str:
    """Query pathway to produce a product. Returns JSON."""
    try:
        data = find_pathway_for_product(product, max_steps=max_steps, starting_materials=starting_materials)
        return json.dumps(data, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "product": product})


# ---------- download_*: save to file, return message ----------


def download_km_values(
    ec_number: str,
    out_path: str,
    organism: str = "*",
    substrate: str = "*",
) -> str:
    """Download Km values to a text/JSON file. Returns status message."""
    try:
        entries = get_km_values(ec_number, organism=organism, substrate=substrate)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.endswith(".json"):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"ec_number": ec_number, "count": len(entries), "entries": entries}, f, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(entries) if entries else "")
        return f"Saved {len(entries)} Km entries to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_reactions(ec_number: str, out_path: str, organism: str = "*") -> str:
    """Download reaction entries to file. Returns status message."""
    try:
        entries = get_reactions(ec_number, organism=organism)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if out_path.endswith(".json"):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"ec_number": ec_number, "count": len(entries), "entries": entries}, f, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(entries) if entries else "")
        return f"Saved {len(entries)} reaction entries to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_enzymes_by_substrate(substrate: str, out_path: str, limit: int = 50) -> str:
    """Download enzyme-by-substrate search results to JSON. Returns status message."""
    try:
        data = search_enzymes_by_substrate(substrate, limit=limit)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"substrate": substrate, "count": len(data), "enzymes": data}, f, indent=2, default=str)
        return f"Saved {len(data)} enzymes to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_compare_organisms(ec_number: str, organisms: List[str], out_path: str) -> str:
    """Download organism comparison to JSON. Returns status message."""
    try:
        data = compare_across_organisms(ec_number, organisms)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"ec_number": ec_number, "organisms": organisms, "comparison": data}, f, indent=2, default=str)
        return f"Saved comparison for {len(organisms)} organisms to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_environmental_parameters(ec_number: str, out_path: str) -> str:
    """Download environmental parameters to JSON. Returns status message."""
    try:
        data = get_environmental_parameters(ec_number)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return f"Saved environmental parameters to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_kinetic_data(ec_number: str, out_path: str, format: str = "json") -> str:
    """Download kinetic data export to file (csv/json). Returns status message."""
    try:
        export_kinetic_data(ec_number, format=format, filename=out_path)
        return f"Saved kinetic data to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_pathway_report(pathway: Dict[str, Any], out_path: str) -> str:
    """Generate and save pathway report to file. Returns status message."""
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        generate_pathway_report(pathway, filename=out_path)
        return f"Saved pathway report to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


__all__ = [
    "query_km_values",
    "query_reactions",
    "query_enzymes_by_substrate",
    "query_compare_organisms",
    "query_environmental_parameters",
    "query_pathway_for_product",
    "download_km_values",
    "download_reactions",
    "download_enzymes_by_substrate",
    "download_compare_organisms",
    "download_environmental_parameters",
    "download_kinetic_data",
    "download_pathway_report",
]


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="BRENDA operations: query_* (return text) and download_* (save file).")
    parser.add_argument("--test", action="store_true", help="Run query_* and download_* samples; output under example/database/brenda")
    args = parser.parse_args()
    if not args.test:
        print("Use --test to run operations tests.")
        exit(0)
    out_base = os.path.join("example", "database", "brenda")
    os.makedirs(out_base, exist_ok=True)
    ec, substrate, organisms = "1.1.1.1", "glucose", ["Escherichia coli", "Saccharomyces cerevisiae", "Homo sapiens"]
    print("=== query_* (return text) ===")
    print("  query_km_values(...)")
    print(query_km_values(ec)[:200] + "..." if len(query_km_values(ec)) > 200 else query_km_values(ec))
    print("  query_reactions(...)")
    print(query_reactions(ec)[:200] + "..." if len(query_reactions(ec)) > 200 else query_reactions(ec))
    print("  query_compare_organisms(...)")
    print(query_compare_organisms(ec, organisms)[:300] + "..." if len(query_compare_organisms(ec, organisms)) > 300 else query_compare_organisms(ec, organisms))
    print("  query_environmental_parameters(...)")
    print(query_environmental_parameters(ec)[:200])
    print("  query_pathway_for_product(...)")
    print(query_pathway_for_product("lactate")[:200])
    print("=== download_* (save to file) ===")
    print("  ", download_km_values(ec, os.path.join(out_base, "km_sample.json")))
    print("  ", download_reactions(ec, os.path.join(out_base, "reactions_sample.txt")))
    print("  ", download_compare_organisms(ec, organisms, os.path.join(out_base, "compare_organisms_sample.json")))
    print("  ", download_environmental_parameters(ec, os.path.join(out_base, "environmental_sample.json")))
    print("  ", download_enzymes_by_substrate(substrate, os.path.join(out_base, "enzymes_by_substrate_sample.json")))
    pathway_json = query_pathway_for_product("lactate")
    try:
        pathway = json.loads(pathway_json)
        if "steps" in pathway or "target" in pathway:
            print("  ", download_pathway_report(pathway, os.path.join(out_base, "pathway_report_sample.txt")))
    except Exception as e:
        print("  ", f"pathway report: {e}")
    print(f"Done. Output under {out_base}")
