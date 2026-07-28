---
name: alphafold_database
description: >-
  AlphaFold DB structures and confidence analytics via VenusFactory tools. Use when the user needs predicted structures by UniProt ID, pLDDT/PAE analysis, or PDB/mmCIF download. Do NOT use for experimental PDB (rcsb_database), local ESMFold without UniProt (predict_structure_esmfold / protein_structure_pipeline), or sequence annotation (uniprot_database).
license: Unknown
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# AlphaFold Database

## Overview

Download AlphaFold predictions to disk and analyze confidence locally. Default model version is **v6**. Large coordinate files must stay on disk — never paste PDB into chat.

## VenusFactory execution

Call hub tools by exact name. Extended Biopython/GCP examples: `read_skill(..., relative_path="references/legacy_guide.md")` or `references/api_reference.md`.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|-------------|
| **download_alphafold_structure_by_uniprot_id** | `uniprot_id`, `out_dir`, `format` (`pdb`\|`cif`, default pdb), `version` (default `v6`), `fragment` | rich JSON `status` + `file_info` | Structure file |
| **download_alphafold_metadata_by_uniprot_id** | `uniprot_id`, `out_dir` | rich JSON + metadata JSON path | Prediction metadata |
| **analyze_alphafold_plddt_by_metadata_file** | `metadata_path` | pLDDT fractions + verdict | Local analysis |
| **analyze_alphafold_pae_by_pae_file** | `pae_path`, optional cutoffs | domains + PAE stats | Local analysis |

There is **no** dedicated PAE download `@tool`. Obtain PAE via metadata `paeDocUrl` + `agent_generated_code`, then analyze.

## Recommended workflow

1. `download_alphafold_metadata_by_uniprot_id` → `analyze_alphafold_plddt_by_metadata_file`
2. `download_alphafold_structure_by_uniprot_id` for coordinates
3. Optional PAE download → `analyze_alphafold_pae_by_pae_file`
4. Optional `render_protein_structure` / figure step for pLDDT plot

## When NOT to use

- Experimental structure → `rcsb_database`
- No UniProt, only raw sequence → `predict_structure_esmfold`
- Full engineering pipeline → `protein_structure_pipeline`

## Common mistakes

- Assuming return shape is `{success, file_path}` — use `status` + `file_info.file_path`
- Using v4 URLs while tools default to **v6**
- Feeding structure PDB into the pLDDT analyzer (needs **metadata JSON**)
- Dumping coordinates into the conversation

## References

- `references/api_reference.md` — REST/URL details
- `references/legacy_guide.md` — archived tutorials (Biopython, GCP bulk)
