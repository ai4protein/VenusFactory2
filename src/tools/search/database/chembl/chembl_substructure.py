"""
ChEMBL substructure search: find compounds containing a SMILES substructure. Atomic script.
Skill: src/agent/skills/chembl_database/
"""
from typing import List, Optional

from .chembl_client import (
    get_client,
    DEFAULT_CHEMBL_MAX_FILTER_RESULTS,
    MAX_CHEMBL_FILTER_RESULTS_CAP,
)


def substructure_search(
    smiles: str,
    max_results: Optional[int] = None,
) -> List[dict]:
    """
    Find compounds that contain the given SMILES as a substructure.
    Returns list of compound records.
    max_results: cap on number of results (default 500, max 5000). None uses default.
    """
    limit = max_results if max_results is not None else DEFAULT_CHEMBL_MAX_FILTER_RESULTS
    limit = min(limit, MAX_CHEMBL_FILTER_RESULTS_CAP)
    client = get_client()
    qs = client.substructure.filter(smiles=smiles)
    return list(qs[:limit]) if qs else []


__all__ = ["substructure_search"]

if __name__ == "__main__":
    print("Testing registered tool: substructure_search", flush=True)
    results = substructure_search("c1ccccc1")
    print(f"  substructure_search(benzene) -> {len(results)} results")
    print("Done.")
