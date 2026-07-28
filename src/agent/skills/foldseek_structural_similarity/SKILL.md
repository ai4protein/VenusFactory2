---
name: foldseek_structural_similarity
description: >-
  FoldSeek structural similarity search against PDB with optional protected-region masking. Use when the user has a PDB and wants fold-level homologs, structural neighbors, or to protect an active site while searching. Do NOT use for sequence BLAST/MMseqs2 (protein_sequence_similarity_search) or MSA (clustalo_msa).
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# FoldSeek Structural Similarity

## Overview

Runs the VenusFactory FoldSeek submit→poll→download pipeline. Coordinates are **1-based inclusive** for `protect_start` / `protect_end` (masked region kept out of the search query).

## Project Tools (VenusFactory2)

| Tool | Args | Returns |
|------|------|---------|
| **download_foldseek_results_by_pdb_file** | `pdb_file_path`, `protect_start`, `protect_end`, optional `out_dir` | rich JSON; m8/FASTA under `file_info` |
| **download_rcsb_structure_by_pdb_id** | hit PDB ids | fetch neighbors |
| **superpose_two_structures** | query + hit PDBs | visual compare |

## Workflow

1. Ensure a local PDB (user upload, AlphaFold, or RCSB).
2. Choose protect region (catalytic site / epitope) or pass a wide range if no mask needed — **both start and end are required by the tool**.
3. Call `download_foldseek_results_by_pdb_file`; wait for submit-poll to finish (network latency expected).
4. Parse hits from `content_preview` / saved m8; fetch interesting hits via RCSB; optional superpose.

## When NOT to use

- Sequence identity / remote homologs without structure → `protein_sequence_similarity_search`
- Align many sequences → `clustalo_msa`

## Common mistakes

- Using 0-based residue indices.
- Confusing FoldSeek with sequence BLAST tools.
- Ignoring timeouts on the webserver queue — report `status:error` suggestion to user.
