---
name: string_database
description: >-
  STRING PPI networks and enrichment via VenusFactory download tools. Use for interaction networks, partners, GO/KEGG enrichment, homology across 5000+ species. Do NOT use for sequence homology search (protein_sequence_similarity_search) or KEGG pathway entries alone (kegg_database).
license: Unknown
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# STRING Database

## Overview

Hub exposes **`download_string_*`** tools only. Lower-level `query_string_*` helpers in `src/tools/database/string/` are not LangChain tools.

## Project Tools (VenusFactory2)

| Tool | Key args | Purpose |
|------|----------|---------|
| **download_string_map_ids** | `identifiers`, `out_dir`, `species` (default 9606) | Name → STRING ID |
| **download_string_network** | `identifiers`, `out_dir`, `required_score` | PPI TSV |
| **download_string_network_image** | `identifiers`, `out_dir` | Network PNG |
| **download_string_interaction_partners** | `identifiers`, `out_dir`, `limit` | Partners TSV |
| **download_string_enrichment** | `identifiers`, `out_dir` | GO/KEGG/Pfam enrichment |
| **download_string_ppi_enrichment** | `identifiers`, `out_dir` | PPI enrichment JSON |
| **download_string_homology** | `identifiers`, `out_dir` | Homology TSV |

## Workflow

1. `download_string_map_ids` (validate gene/protein names)
2. Network and/or partners + enrichment
3. Plan a score-bar figure even if network PNG exists

## Common mistakes

- Calling `query_string_*` as hub tools
- Wrong `species` taxon (human=9606)
- Score threshold too high → empty network

## References (progressive disclosure)

**Trust order:** `SKILL.md` (hub tools & envelopes) → topic refs → `references/legacy_guide.md` (archived; may be outdated).

```text
read_skill(skill_id="string_database", relative_path="references/legacy_guide.md")
```

Load legacy only after the hub workflow in this file is insufficient.
