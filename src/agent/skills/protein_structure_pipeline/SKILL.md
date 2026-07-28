---
name: protein_structure_pipeline
description: >-
  Protein structure obtain → confidence → visualize pipeline. Use when the user needs a 3D structure from sequence or UniProt ID, AlphaFold/ESMFold retrieval, pLDDT/PAE analysis, or structure rendering. Do NOT use for mutation ranking (zero_shot_mutation_workflow), FoldSeek search (foldseek_structural_similarity), or experimental PDB-only metadata without structure needs (rcsb_database alone may suffice).
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# Protein Structure Pipeline

## Overview

Chains VenusFactory structure tools: AlphaFold DB download + confidence analytics, local ESMFold when no UniProt ID, RCSB for experimental structures, and PyMOL rendering.

## VenusFactory execution

Call hub tools only. Large PDB/mmCIF stay on disk via `file_info.file_path`.

## Project Tools (VenusFactory2)

| Tool | Args | When |
|------|------|------|
| **download_alphafold_structure_by_uniprot_id** | `uniprot_id`, **`out_dir`**, `format` | Known UniProt accession |
| **download_alphafold_metadata_by_uniprot_id** | `uniprot_id`, **`out_dir`** | Need pLDDT metadata JSON |
| **analyze_alphafold_plddt_by_metadata_file** | `metadata_path` | Confidence per residue |
| **analyze_alphafold_pae_by_pae_file** | `pae_path` | Domain/interface confidence |
| **predict_structure_esmfold** | `sequence`, optional `output_dir` | No UniProt / quick local fold |
| **download_rcsb_structure_by_pdb_id** | `pdb_id`, **`out_dir`**, format | Experimental structure |
| **download_rcsb_entry_metadata_by_pdb_id** | `pdb_id`, **`out_path`** | Resolution, method, ligands |
| **render_protein_structure** | `pdb_path`, **`out_dir`**, style options | Publication-quality still |
| **superpose_two_structures** | `pdb_a`, `pdb_b`, **`out_dir`** | Compare models |

## Workflow

### UniProt → AlphaFold (default)

1. `download_alphafold_structure_by_uniprot_id`
2. `download_alphafold_metadata_by_uniprot_id` → `analyze_alphafold_plddt_by_metadata_file`
3. If PAE available, `analyze_alphafold_pae_by_pae_file`
4. Optional `render_protein_structure` (cartoon + pLDDT coloring if supported by tool args)

### Sequence-only → ESMFold

1. Obtain sequence (`read_fasta` / UniProt seq tool).
2. `predict_structure_esmfold`
3. Optional RSA/SASA/SS via `protein_property_prediction` tools.

### Experimental PDB

1. `download_rcsb_structure_by_pdb_id` + metadata.
2. Prefer over AlphaFold when an experimental entry exists for the same construct.

## When NOT to use

- Structural homolog search with active-site masking → `foldseek_structural_similarity`
- Domain annotation without structure → `interpro_domain_annotation`

## Common mistakes

- Using AlphaFold when user already has a high-res PDB.
- Dumping PDB text into chat instead of using `file_info.file_path`.
- Skipping pLDDT before trusting loop regions for mutation design.
