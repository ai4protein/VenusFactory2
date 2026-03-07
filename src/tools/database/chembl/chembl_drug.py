"""
ChEMBL drug / mechanism / indication API. Atomic script.
Skill: src/agent/skills/chembl_database/
"""
from typing import Any, Dict, List, Optional

from .chembl_client import (
    get_client,
    DEFAULT_CHEMBL_MAX_FILTER_RESULTS,
    MAX_CHEMBL_FILTER_RESULTS_CAP,
)


def get_drug(chembl_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve drug record by ChEMBL molecule ID. Returns dict or None."""
    client = get_client()
    try:
        return client.drug.get(chembl_id)
    except Exception:
        return None


def get_mechanisms(
    molecule_chembl_id: str,
    max_results: Optional[int] = None,
) -> List[dict]:
    """Return list of mechanism-of-action records for the given molecule ChEMBL ID.
    max_results: cap on number of results (default 500, max 5000). None uses default.
    """
    limit = max_results if max_results is not None else DEFAULT_CHEMBL_MAX_FILTER_RESULTS
    limit = min(limit, MAX_CHEMBL_FILTER_RESULTS_CAP)
    client = get_client()
    qs = client.mechanism.filter(molecule_chembl_id=molecule_chembl_id)
    return list(qs[:limit]) if qs else []


def get_indications(
    molecule_chembl_id: str,
    max_results: Optional[int] = None,
) -> List[dict]:
    """Return list of drug indication records for the given molecule ChEMBL ID.
    max_results: cap on number of results (default 500, max 5000). None uses default.
    """
    limit = max_results if max_results is not None else DEFAULT_CHEMBL_MAX_FILTER_RESULTS
    limit = min(limit, MAX_CHEMBL_FILTER_RESULTS_CAP)
    client = get_client()
    qs = client.drug_indication.filter(molecule_chembl_id=molecule_chembl_id)
    return list(qs[:limit]) if qs else []


__all__ = ["get_drug", "get_mechanisms", "get_indications"]

if __name__ == "__main__":
    print("Testing registered tools: get_drug, get_mechanisms, get_indications", flush=True)
    d = get_drug("CHEMBL25")
    print("  get_drug(CHEMBL25):", d.get("pref_name") if d else "not found")
    mech = get_mechanisms("CHEMBL25")
    print(f"  get_mechanisms(CHEMBL25) -> {len(mech)} items")
    ind = get_indications("CHEMBL25")
    print(f"  get_indications(CHEMBL25) -> {len(ind)} items")
    print("Done.")
