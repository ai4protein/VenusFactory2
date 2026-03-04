"""
ChEMBL activity API: filter bioactivities by target, molecule, type, value. Atomic script.
Skill: src/agent/skills/chembl_database/
"""
from typing import Any, List, Optional

from .chembl_client import (
    get_client,
    DEFAULT_CHEMBL_MAX_FILTER_RESULTS,
    MAX_CHEMBL_FILTER_RESULTS_CAP,
)


def filter_activities(
    *,
    max_results: Optional[int] = None,
    **kwargs: Any,
) -> List[dict]:
    """
    Filter bioactivity records by Django-style kwargs (e.g. target_chembl_id='CHEMBL203',
    standard_type='IC50', standard_value__lte=100, standard_units='nM').
    Returns list of activity dicts.
    max_results: cap on number of results (default 500, max 5000). None uses default.
    """
    limit = max_results if max_results is not None else DEFAULT_CHEMBL_MAX_FILTER_RESULTS
    limit = min(limit, MAX_CHEMBL_FILTER_RESULTS_CAP)
    client = get_client()
    qs = client.activity.filter(**kwargs)
    return list(qs[:limit]) if qs else []


__all__ = ["filter_activities"]

if __name__ == "__main__":
    print("ChEMBL activity example:")
    acts = filter_activities(
        target_chembl_id="CHEMBL203",
        standard_type="IC50",
        standard_value__lte=100,
        standard_units="nM",
        max_results=100,
    )
    print("  CHEMBL203 IC50 <= 100 nM count:", len(acts))
    if acts:
        print("  first keys:", list(acts[0].keys())[:6])
