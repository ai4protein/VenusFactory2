"""
ChEMBL target API: get by ID, filter by type/name. Atomic script.
Skill: src/agent/skills/chembl_database/
"""
from typing import Any, Dict, List, Optional

from .chembl_client import (
    get_client,
    DEFAULT_CHEMBL_MAX_FILTER_RESULTS,
    MAX_CHEMBL_FILTER_RESULTS_CAP,
)


def get_target(chembl_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a target by ChEMBL ID (e.g. CHEMBL203). Returns dict or None if not found.
    """
    client = get_client()
    try:
        return client.target.get(chembl_id)
    except Exception:
        return None


def filter_targets(
    *,
    max_results: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Filter targets by Django-style kwargs (e.g. target_type='SINGLE PROTEIN',
    pref_name__icontains='kinase'). Returns list of dicts.
    max_results: cap on number of results (default 500, max 5000). None uses default.
    """
    limit = max_results if max_results is not None else DEFAULT_CHEMBL_MAX_FILTER_RESULTS
    limit = min(limit, MAX_CHEMBL_FILTER_RESULTS_CAP)
    client = get_client()
    qs = client.target.filter(**kwargs)
    return list(qs[:limit]) if qs else []


__all__ = ["get_target", "filter_targets"]

if __name__ == "__main__":
    print("ChEMBL target example:")
    t = get_target("CHEMBL203")
    if t:
        print("  CHEMBL203 pref_name:", t.get("pref_name"))
    print("  filter_targets(pref_name__icontains='EGFR') count:", len(filter_targets(pref_name__icontains="EGFR")))
