---
name: rcsb_database
description: RCSB Protein Data Bank (PDB) — experimentally determined 3D biomolecular structures. Search by full-text/sequence/structure/attribute, fetch entry metadata, download coordinate files (PDB/mmCIF). Use when the user provides a PDB ID, asks for structures of a protein, wants to find similar structures by sequence, or needs experimental (not predicted) coordinates. Don't use for AlphaFold predictions (use alphafold_database).
license: Unknown
metadata:
    skill-author: VenusFactory2.
---

# RCSB Protein Data Bank

## Overview

The RCSB PDB hosts >200K experimentally-determined biomolecular structures (X-ray, cryo-EM, NMR, etc.). This skill exposes 3 download tools and 1 search tool.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **download_rcsb_search_by_query** | `query` (JSON string or dict — Search API v2 query block or full payload), `out_dir` (required), `return_type` (default `"entry"`; `entry` \| `assembly` \| `polymer_entity` \| `non_polymer_entity` \| `polymer_instance` \| `mol_definition`), `page_start`/`rows` (pagination; default = return all hits), `sort_by` (e.g. `score`, `rcsb_accession_info.initial_release_date`), `sort_direction` (`asc` \| `desc`), `count_only` (skip results, return only `total_count`), `timeout` (default `60`s) | JSON: `{status, file_info {file_path → rcsb_search_<return_type>.json}, content_preview (first 25 ids), biological_metadata {return_type, total_count, result_count, page_start, rows, sort_by, search_time_ms}}` | Run an RCSB Search API v2 query (text / sequence / structure / attribute). |
| **download_rcsb_entry_metadata_by_pdb_id** | `pdb_id` (required, e.g. `4HHB`), `out_path` (required, JSON path) | rich JSON envelope; saves metadata JSON to `out_path` | Fetch full entry metadata for one PDB ID. |
| **download_rcsb_structure_by_pdb_id** | `pdb_id` (required), `out_dir` (required), `file_type` (default `pdb`; `pdb` \| `cif` \| `xml`) | rich JSON envelope; structure file at `file_info.file_path` | Download coordinate file. |

## When to Use This Skill

- User provides a PDB ID and asks for the structure / metadata
- "Find structures of X" / "find structures similar to this sequence" → `download_rcsb_search_by_query` with the right `service`
- Build a workflow that pipes RCSB structures into PyMOL (`render_protein_structure`) or Foldseek

## Search API v2 Query Cookbook

Pass `query` as a dict / JSON-string of a *query block* (auto-wrapped into `{"query": ...}`) or a full request payload.

**Full-text:**
```python
{"type": "terminal", "service": "full_text", "parameters": {"value": "hemoglobin"}}
```

**By sequence (BLAST-like, fast, e-value cutoff):**
```python
{
  "type": "terminal", "service": "sequence",
  "parameters": {
    "evalue_cutoff": 0.1, "identity_cutoff": 0.95,
    "sequence_type": "protein",
    "value": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFF..."
  }
}
```

**By attribute (e.g. resolution, organism, year):**
```python
{
  "type": "group", "logical_operator": "and",
  "nodes": [
    {"type": "terminal", "service": "text",
     "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less", "value": 2.0}},
    {"type": "terminal", "service": "text",
     "parameters": {"attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                    "operator": "exact_match", "value": "Homo sapiens"}}
  ]
}
```

**Boolean combination:** use `{"type": "group", "logical_operator": "and"|"or", "nodes": [...]}`.

## Return Type → Identifier Shape

| return_type | identifier shape |
|---|---|
| `entry` | `4HHB` (PDB ID) |
| `assembly` | `4HHB-1` |
| `polymer_entity` | `4HHB-1` (entity ID) |
| `polymer_instance` | `4HHB.A` (chain) |
| `mol_definition` | `HEM` (ligand 3-letter code) |

## Common Mistakes

- **Passing a Python query dict expecting URL-encoding**: dicts are serialized & url-encoded automatically; pass the dict, not a hand-crafted URL.
- **Forgetting `return_type`**: defaults to `entry` (PDB IDs). If you want chain identifiers, pass `polymer_instance`.
- **Using `count_only=True` and then trying to read `result_set`**: count-only mode returns `{"total_count": N}` only, no `result_set`.
- **Sequence searches with `identity_cutoff` too strict**: try lower (e.g. 0.3) if you get few hits.

## References

- [RCSB Search API v2 docs](https://search.rcsb.org/)
- [Query attribute schema](https://search.rcsb.org/rcsbsearch/v2/metadata/schema)
- [Data API docs](https://data.rcsb.org/)
