"""
ChEMBL operations: single exit for query (return JSON text) and download (save to file).

All public functions are named with database source (chembl):
- query_chembl_*: return str (JSON text)
- download_chembl_*: write to file and return status message str.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from .chembl_molecule import get_molecule
    from .chembl_similarity import similarity_search
    from .chembl_substructure import substructure_search
    from .chembl_drug import get_drug, get_mechanisms, get_indications
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "chembl" and str(_dir.parents[4]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[4]))
    from src.tools.search.database.chembl.chembl_molecule import get_molecule
    from src.tools.search.database.chembl.chembl_similarity import similarity_search
    from src.tools.search.database.chembl.chembl_substructure import substructure_search
    from src.tools.search.database.chembl.chembl_drug import get_drug, get_mechanisms, get_indications


def _json_serializer(obj: Any) -> Any:
    """Default serializer for ChEMBL API objects (e.g. dict with non-str values)."""
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
        try:
            return list(obj)
        except Exception:
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------- query_chembl_*: return JSON text ----------


def query_chembl_molecule(chembl_id: str) -> str:
    """Query ChEMBL molecule by ChEMBL ID. Returns JSON text, no file download."""
    try:
        mol = get_molecule(chembl_id)
        if mol is None:
            return json.dumps({"success": False, "error": f"Molecule {chembl_id} not found"}, ensure_ascii=False)
        return json.dumps({"success": True, "molecule": mol}, ensure_ascii=False, default=_json_serializer)
    except ImportError as e:
        return json.dumps({"success": False, "error": f"ChEMBL dependency missing: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def query_chembl_similarity(
    smiles: str,
    threshold: int = 85,
    max_results: Optional[int] = None,
) -> str:
    """Query ChEMBL for similar molecules by SMILES (Tanimoto 0-100). Returns JSON text, no file download."""
    try:
        results = similarity_search(smiles, threshold=threshold, max_results=max_results)
        return json.dumps(
            {"success": True, "smiles": smiles, "threshold": threshold, "results": results},
            ensure_ascii=False,
            default=_json_serializer,
        )
    except ImportError as e:
        return json.dumps({"success": False, "error": f"ChEMBL dependency missing: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def query_chembl_substructure(
    smiles: str,
    max_results: Optional[int] = None,
) -> str:
    """Query ChEMBL for molecules containing SMILES substructure. Returns JSON text, no file download."""
    try:
        results = substructure_search(smiles, max_results=max_results)
        return json.dumps(
            {"success": True, "smiles": smiles, "results": results},
            ensure_ascii=False,
            default=_json_serializer,
        )
    except ImportError as e:
        return json.dumps({"success": False, "error": f"ChEMBL dependency missing: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def query_chembl_drug(
    chembl_id: str,
    max_results: Optional[int] = None,
) -> str:
    """Query ChEMBL drug info by ChEMBL ID (drug, mechanisms, indications). Returns JSON text, no file download."""
    try:
        drug = get_drug(chembl_id)
        mechanisms = get_mechanisms(chembl_id, max_results=max_results) if drug else []
        indications = get_indications(chembl_id, max_results=max_results) if drug else []
        return json.dumps(
            {"success": True, "drug": drug, "mechanisms": mechanisms, "indications": indications},
            ensure_ascii=False,
            default=_json_serializer,
        )
    except ImportError as e:
        return json.dumps({"success": False, "error": f"ChEMBL dependency missing: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


# ---------- download_chembl_*: save to file, return message ----------


def download_chembl_molecule(chembl_id: str, out_path: str) -> str:
    """Download ChEMBL molecule JSON by ChEMBL ID to file. Returns status message."""
    try:
        text = query_chembl_molecule(chembl_id)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return f"Saved molecule {chembl_id} to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_chembl_similarity(
    smiles: str,
    out_path: str,
    threshold: int = 85,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL similarity search results to JSON file. Returns status message."""
    try:
        text = query_chembl_similarity(smiles, threshold=threshold, max_results=max_results)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return f"Saved similarity results (threshold={threshold}) to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_chembl_substructure(
    smiles: str,
    out_path: str,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL substructure search results to JSON file. Returns status message."""
    try:
        text = query_chembl_substructure(smiles, max_results=max_results)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return f"Saved substructure results to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


def download_chembl_drug(
    chembl_id: str,
    out_path: str,
    max_results: Optional[int] = None,
) -> str:
    """Download ChEMBL drug info (drug, mechanisms, indications) to JSON file. Returns status message."""
    try:
        text = query_chembl_drug(chembl_id, max_results=max_results)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return f"Saved drug {chembl_id} to {out_path}"
    except Exception as e:
        return f"Download failed: {e}"


__all__ = [
    "query_chembl_molecule",
    "query_chembl_similarity",
    "query_chembl_substructure",
    "query_chembl_drug",
    "download_chembl_molecule",
    "download_chembl_similarity",
    "download_chembl_substructure",
    "download_chembl_drug",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ChEMBL operations: query_chembl_* (return JSON) and download_chembl_* (save file)."
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

    out_base = os.path.join("example", "database", "chembl")
    os.makedirs(out_base, exist_ok=True)

    print("=== query_chembl_* (return JSON) ===")
    print("  query_chembl_molecule(...)")
    out = query_chembl_molecule(args.mol_id)
    print(out[:300] + "..." if len(out) > 300 else out)
    print("  query_chembl_similarity(...)")
    out = query_chembl_similarity(args.smiles, threshold=args.threshold)
    print(out[:300] + "..." if len(out) > 300 else out)
    print("  query_chembl_substructure(...)")
    out = query_chembl_substructure("c1ccccc1")
    print(out[:300] + "..." if len(out) > 300 else out)
    print("  query_chembl_drug(...)")
    out = query_chembl_drug(args.mol_id)
    print(out[:300] + "..." if len(out) > 300 else out)

    print("=== download_chembl_* (save to file) ===")
    print("  ", download_chembl_molecule(args.mol_id, os.path.join(out_base, "chembl_molecule_sample.json")))
    print("  ", download_chembl_similarity(args.smiles, os.path.join(out_base, "chembl_similarity_sample.json"), threshold=args.threshold))
    print("  ", download_chembl_substructure("c1ccccc1", os.path.join(out_base, "chembl_substructure_sample.json")))
    print("  ", download_chembl_drug(args.mol_id, os.path.join(out_base, "chembl_drug_sample.json")))

    print(f"Done. Output under {out_base}")
