"""
ChEMBL similarity search: find compounds similar to a SMILES. Atomic script.
Skill: src/agent/skills/chembl_database/
"""
from typing import List, Optional

from .chembl_client import (
    get_client,
    DEFAULT_CHEMBL_MAX_FILTER_RESULTS,
    MAX_CHEMBL_FILTER_RESULTS_CAP,
)


def similarity_search(
    smiles: str,
    threshold: int = 85,
    max_results: Optional[int] = None,
) -> List[dict]:
    """
    Find compounds similar to the given SMILES string.
    threshold: minimum similarity percentage (0–100). Returns list of compound records.
    max_results: cap on number of results (default 500, max 5000). None uses default.
    """
    limit = max_results if max_results is not None else DEFAULT_CHEMBL_MAX_FILTER_RESULTS
    limit = min(limit, MAX_CHEMBL_FILTER_RESULTS_CAP)
    client = get_client()
    qs = client.similarity.filter(smiles=smiles, similarity=threshold)
    return list(qs[:limit]) if qs else []


__all__ = ["similarity_search"]

if __name__ == "__main__":
    print("Testing registered tool: similarity_search", flush=True)
    results = similarity_search("CC(=O)Oc1ccccc1C(=O)O", threshold=70)
    print(f"  similarity_search(aspirin SMILES, 70) -> {len(results)} results")
    print("Done.")
