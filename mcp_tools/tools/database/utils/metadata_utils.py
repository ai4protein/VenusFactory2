"""Parse metadata files (e.g. RCSB entry JSON)."""
import json


def get_uid_from_rcsb_metadata(meta_data_file: str) -> str:
    """
    Get UniProt ID from RCSB PDB metadata JSON file.
    Supports REST API format (entry at root, polymer_entities array) and legacy GraphQL (data.entry).
    """
    with open(meta_data_file) as f:
        data = json.load(f)
    entry = data if data.get("rcsb_id") or data.get("polymer_entities") is not None else (data.get("data") or {}).get("entry")
    if not entry:
        raise KeyError("No entry in RCSB metadata file")
    polymer = entry.get("polymer_entities")
    if isinstance(polymer, list) and polymer:
        pe = polymer[0]
        uniprots = pe.get("uniprots")
        if isinstance(uniprots, dict):
            return uniprots.get("rcsb_id", "")
        if isinstance(uniprots, str):
            return uniprots
    if isinstance(polymer, dict):
        return (polymer.get("uniprots") or {}).get("rcsb_id", "")
    raise KeyError("No polymer_entities/uniprots in RCSB metadata file")
