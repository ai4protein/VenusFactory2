"""Metadata file parsing (e.g. RCSB JSON)."""
import json


def get_uniprot_id_from_rcsb_metadata(meta_data_file: str) -> str:
    """
    Get UniProt ID from RCSB PDB metadata JSON file.
    Expects structure: data.entry.polymer_entities.uniprots.rcsb_id
    """
    with open(meta_data_file, "r") as f:
        data = json.load(f)
    return data["data"]["entry"]["polymer_entities"]["uniprots"]["rcsb_id"]
