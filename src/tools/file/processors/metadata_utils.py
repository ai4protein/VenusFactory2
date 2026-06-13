"""Metadata file parsing (e.g. RCSB JSON)."""
import json
import re
from typing import Any

import requests


_RCSB_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"


def _try_extract_pdb_id(data: dict, meta_data_file: str) -> str | None:
    # The entry-level JSON saved by download_rcsb_entry_metadata_by_pdb_id has
    # ``rcsb_id`` at the entry root.
    for path in (
        ("rcsb_id",),
        ("entry", "id"),
        ("rcsb_entry_container_identifiers", "entry_id"),
    ):
        cur: Any = data
        try:
            for k in path:
                cur = cur[k]
            if isinstance(cur, str) and cur:
                return cur.upper()
        except (KeyError, TypeError):
            continue
    # Last resort: filename like "5XJH_metadata.json" → "5XJH"
    m = re.search(r"([A-Za-z0-9]{4})", meta_data_file)
    return m.group(1).upper() if m else None


def _try_extract_entity_ids(data: dict) -> list[str]:
    ids = (
        (data.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids")
        or (data.get("entry") or {}).get("polymer_entity_ids")
        or []
    )
    return [str(x) for x in ids if x is not None]


def _fetch_uniprots_for_entity(pdb_id: str, entity_id: str) -> list[str]:
    """Hit RCSB REST and extract uniprot_ids from polymer entity details."""
    url = _RCSB_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id)
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return []
    try:
        ent = r.json()
    except ValueError:
        return []
    cids = ent.get("rcsb_polymer_entity_container_identifiers") or {}
    uids = cids.get("uniprot_ids") or []
    return [str(u) for u in uids if u]


def get_uniprot_id_from_rcsb_metadata(meta_data_file: str) -> str:
    """Return the first UniProt accession associated with the PDB entry.

    The download tool ``download_rcsb_entry_metadata_by_pdb_id`` only saves the
    **entry-level** record (cell / citation / exptl / rcsb_entry_info / ...).
    Polymer-entity records (which hold the UniProt cross-refs) come from a
    separate endpoint, so we fetch them on demand here:

        rcsb_entry_container_identifiers.polymer_entity_ids   # in entry JSON
        → https://data.rcsb.org/rest/v1/core/polymer_entity/<PDB>/<ENTITY>
          → rcsb_polymer_entity_container_identifiers.uniprot_ids[0]

    Falls back to the GraphQL-shaped path if the file actually is a GraphQL
    response (kept for backward-compat with the original implementation).
    """
    with open(meta_data_file, "r") as f:
        data = json.load(f)

    # Backward-compat: legacy GraphQL response shape with top-level "data"
    try:
        legacy = data["data"]["entry"]["polymer_entities"]
        if isinstance(legacy, list) and legacy:
            entities = legacy
        elif isinstance(legacy, dict):
            entities = [legacy]
        else:
            entities = []
        for ent in entities:
            uids_obj = ent.get("uniprots") if isinstance(ent, dict) else None
            if isinstance(uids_obj, list):
                for u in uids_obj:
                    if isinstance(u, dict) and u.get("rcsb_id"):
                        return str(u["rcsb_id"])
            elif isinstance(uids_obj, dict) and uids_obj.get("rcsb_id"):
                return str(uids_obj["rcsb_id"])
    except (KeyError, TypeError):
        pass

    # Modern path: entry-level JSON → REST polymer_entity fetches
    pdb_id = _try_extract_pdb_id(data, meta_data_file)
    if not pdb_id:
        raise ValueError(
            "Could not determine PDB ID from metadata file (no rcsb_id, "
            "entry.id, or 4-letter token in filename)."
        )
    entity_ids = _try_extract_entity_ids(data)
    if not entity_ids:
        raise ValueError(
            f"Entry JSON for {pdb_id} has no polymer_entity_ids; cannot resolve "
            "UniProt IDs. Re-download with download_rcsb_entry_metadata_by_pdb_id "
            "or check that the PDB entry actually contains polymer entities."
        )
    for eid in entity_ids:
        uids = _fetch_uniprots_for_entity(pdb_id, eid)
        if uids:
            return uids[0]
    raise ValueError(
        f"No UniProt cross-refs found for {pdb_id} polymer entities "
        f"{entity_ids} via RCSB REST. Entry may be a designed/synthetic protein."
    )
