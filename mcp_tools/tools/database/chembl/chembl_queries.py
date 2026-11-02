"""
ChEMBL high-level query helpers built on atomic modules (molecule, target, activity, drug, etc.).
Skill: src/agent/skills/chembl_database/
"""
from typing import Any, Dict, List, Optional

from .chembl_molecule import get_molecule, filter_molecules
from .chembl_target import get_target, filter_targets
from .chembl_activity import filter_activities
from .chembl_similarity import similarity_search
from .chembl_substructure import substructure_search
from .chembl_drug import get_drug, get_mechanisms, get_indications


def get_molecule_info(chembl_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve detailed molecule information by ChEMBL ID. Alias for get_molecule."""
    return get_molecule(chembl_id)


def search_molecules_by_name(name_pattern: str) -> List[Dict[str, Any]]:
    """Search molecules by name pattern (case-insensitive contains)."""
    return filter_molecules(pref_name__icontains=name_pattern)


def find_molecules_by_properties(
    max_mw: int = 500,
    min_logp: Optional[float] = None,
    max_logp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Find molecules by physicochemical properties (MW, LogP)."""
    filters: Dict[str, Any] = {"molecule_properties__mw_freebase__lte": max_mw}
    if min_logp is not None:
        filters["molecule_properties__alogp__gte"] = min_logp
    if max_logp is not None:
        filters["molecule_properties__alogp__lte"] = max_logp
    return filter_molecules(**filters)


def get_target_info(target_chembl_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve target information by ChEMBL ID. Alias for get_target."""
    return get_target(target_chembl_id)


def search_targets_by_name(target_name: str) -> List[Dict[str, Any]]:
    """Search targets by name (SINGLE PROTEIN, pref_name contains)."""
    return filter_targets(
        target_type="SINGLE PROTEIN",
        pref_name__icontains=target_name,
    )


def get_bioactivity_data(
    target_chembl_id: str,
    activity_type: str = "IC50",
    max_value: float = 100,
) -> List[dict]:
    """Retrieve bioactivity records for a target (e.g. IC50 <= max_value nM)."""
    return filter_activities(
        target_chembl_id=target_chembl_id,
        standard_type=activity_type,
        standard_value__lte=max_value,
        standard_units="nM",
    )


def find_similar_compounds(smiles: str, similarity_threshold: int = 85) -> List[dict]:
    """Find compounds similar to the given SMILES. Wraps similarity_search."""
    return similarity_search(smiles, threshold=similarity_threshold)


def get_compound_bioactivities(molecule_chembl_id: str) -> List[dict]:
    """Get all bioactivity records for a compound (with pChEMBL value)."""
    return filter_activities(
        molecule_chembl_id=molecule_chembl_id,
        pchembl_value__isnull=False,
    )


def get_drug_info(molecule_chembl_id: str) -> tuple:
    """Return (drug_info, mechanisms, indications) for a molecule. Alias for get_drug + get_mechanisms + get_indications."""
    drug_info = get_drug(molecule_chembl_id)
    mechanisms = get_mechanisms(molecule_chembl_id)
    indications = get_indications(molecule_chembl_id)
    return drug_info, mechanisms, indications


def find_kinase_inhibitors(max_ic50: float = 100) -> List[dict]:
    """Find potent kinase inhibitor activities (IC50 <= max_ic50 nM)."""
    kinase_targets = filter_targets(
        target_type="SINGLE PROTEIN",
        pref_name__icontains="kinase",
    )
    target_ids = [t["target_chembl_id"] for t in kinase_targets[:10]]
    if not target_ids:
        return []
    return filter_activities(
        target_chembl_id__in=target_ids,
        standard_type="IC50",
        standard_value__lte=max_ic50,
        standard_units="nM",
    )


def export_to_dataframe(data: List[dict]):
    """Convert list of ChEMBL records to pandas DataFrame. Returns None if pandas not installed."""
    try:
        import pandas as pd
        return pd.DataFrame(data)
    except ImportError:
        return None


__all__ = [
    "get_molecule_info",
    "search_molecules_by_name",
    "find_molecules_by_properties",
    "get_target_info",
    "search_targets_by_name",
    "get_bioactivity_data",
    "find_similar_compounds",
    "get_compound_bioactivities",
    "get_drug_info",
    "find_kinase_inhibitors",
    "export_to_dataframe",
]

if __name__ == "__main__":
    print("ChEMBL queries example:")
    aspirin = get_molecule_info("CHEMBL25")
    print("  aspirin pref_name:", aspirin.get("pref_name") if aspirin else None)
    print("  get_bioactivity_data(CHEMBL203, IC50, 10) count:", len(get_bioactivity_data("CHEMBL203", "IC50", 10)))
