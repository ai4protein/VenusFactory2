"""
ChEMBL molecule API: get by ID, filter by name/properties. Atomic script.
Skill: src/agent/skills/chembl_database/
"""
from typing import Any, Dict, List, Optional

from .chembl_client import (
    get_client,
    DEFAULT_CHEMBL_MAX_FILTER_RESULTS,
    MAX_CHEMBL_FILTER_RESULTS_CAP,
)


def get_molecule(chembl_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a molecule by ChEMBL ID (e.g. CHEMBL25).
    Returns dict or None if not found.
    """
    client = get_client()
    try:
        return client.molecule.get(chembl_id)
    except Exception:
        return None


def filter_molecules(
    *,
    max_results: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Filter molecules by Django-style kwargs (e.g. pref_name__icontains='aspirin',
    molecule_properties__mw_freebase__lte=500). Returns list of dicts.
    max_results: cap on number of results (default 500, max 5000). None uses default.
    """
    limit = max_results if max_results is not None else DEFAULT_CHEMBL_MAX_FILTER_RESULTS
    limit = min(limit, MAX_CHEMBL_FILTER_RESULTS_CAP)
    client = get_client()
    qs = client.molecule.filter(**kwargs)
    return list(qs[:limit]) if qs else []


__all__ = ["get_molecule", "filter_molecules"]

if __name__ == "__main__":
    print("Testing registered tool: get_molecule", flush=True)
    m = get_molecule("CHEMBL25")
    if m:
        print("  get_molecule(CHEMBL25) pref_name:", m.get("pref_name"))
    else:
        print("  get_molecule(CHEMBL25): not found or client unavailable")
    print("Done.")
