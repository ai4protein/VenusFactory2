---
name: kegg_database
description: >-
  KEGG REST access via VenusFactory download tools (academic use). Use for pathway/gene/compound lookups, ID conversion, and DDI. Do NOT use for PPI networks (string_database) or enzyme kinetics (brenda_database). Non-academic use of KEGG requires a commercial license.
license: Non-academic use of KEGG requires a commercial license
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# KEGG Database

## Project Tools (VenusFactory2)

| Tool | Purpose |
|------|---------|
| **download_kegg_info_by_database** | Database info |
| **download_kegg_list_by_database** | List entries |
| **download_kegg_find_by_database** | Text |
| **download_kegg_entry_by_id** | Entry detail |
| **download_kegg_conv_by_id** | ID conversion |
| **download_kegg_link_by_id** | Cross-links |
| **download_kegg_ddi_by_id** | Drug–drug interaction |

All write results under required **`out_path`** (not `out_dir`) and return rich JSON (`status` + `file_info`).

## Workflow

1. `find` / `list` → `entry` → optional `link`/`conv`.
2. Combine with `string_database` for PPI enrichment context; `brenda_database` for kinetics.

## Common mistakes

- Ignoring KEGG academic-use license constraints
- Confusing KEGG gene ids with NCBI Gene ids without `conv`

## References (progressive disclosure)

**Trust order:** `SKILL.md` → topic refs → `references/legacy_guide.md` (archived; may show `query_kegg_*` library APIs).

```text
read_skill(skill_id="kegg_database", relative_path="references/kegg_reference.md")
read_skill(skill_id="kegg_database", relative_path="references/legacy_guide.md")
```

| File | When to load |
|------|----------------|
| `references/kegg_reference.md` | REST field details |
| `references/legacy_guide.md` | Extended tutorials after hub workflow is insufficient |
