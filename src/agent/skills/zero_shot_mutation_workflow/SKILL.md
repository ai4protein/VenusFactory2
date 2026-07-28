---
name: zero_shot_mutation_workflow
description: >-
  Zero-shot mutation engineering with VenusFactory PLMs. Use when the user wants beneficial mutations, directed evolution candidates, or stability/fitness ranking from a FASTA sequence or PDB structure. Do NOT use for ProteinMPNN inverse folding (proteinmpnn_design_workflow), sequence homology search (protein_sequence_similarity_search), or experimental wet-lab protocols alone.
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# Zero-Shot Mutation Workflow

## Overview

Orchestrates VenusFactory mutation tools: sequence-only PLMs (ESM-1v / ESM2 / VenusPLM) or structure-aware PLMs (ESM-IF1 / SaProt / ProtSSN / MIF-ST). Outputs CSV + heatmap paths in the rich JSON envelope — treat scores as **computational hypotheses**, not wet-lab results.

## VenusFactory execution

1. `read_skill` with `skill_id: zero_shot_mutation_workflow` (this file).
2. Call the hub tools below by **exact name** (do not invent Forge/ESM SDK calls).
3. Parse `status`, `file_info.file_path`, and any `data` / heatmap paths.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | When |
|------|------|---------|------|
| **zero_shot_mutation_sequence_prediction** | `sequence` **or** `fasta_file`; `model_name` (default `ESM2-650M`); `backend` (`local`/`pjlab`); optional `out_dir` | status JSON + CSV/heatmap | Only sequence available |
| **zero_shot_mutation_structure_prediction** | `structure_file` (PDB); `model_name` (default `ESM-IF1`); `backend`; optional `out_dir` | status JSON + CSV/heatmap | PDB available (prefer over sequence-only) |
| **read_fasta** | `file_path` | sequence content | Inspect uploaded FASTA |
| **get_seq_from_pdb_chain_a** | `pdb_file` | chain A sequence | Need sequence from structure |
| **render_protein_structure** | `pdb_path`, **`out_dir`**, style options | image under `file_info` | Visualize top mutation sites |

Sequence models: `ESM-1v`, `ESM2-650M`, `ESM-1b`, `VenusPLM`.  
Structure models: `ESM-IF1`, `SaProt`, `ProtSSN`, `MIF-ST` (and related names exposed by the tool).

## Workflow

### A. Sequence-only path

1. Ensure FASTA via `read_fasta` or pass `sequence` directly.
2. Call `zero_shot_mutation_sequence_prediction` with an explicit `out_dir` under the session workspace.
3. Summarize top-ranked substitutions; warn that scores are model-dependent.

### B. Structure path (preferred when PDB exists)

1. Validate PDB path (may come from `predict_structure_esmfold` or `download_alphafold_structure_by_uniprot_id`).
2. Call `zero_shot_mutation_structure_prediction`.
3. Optional: `render_protein_structure` highlighting candidate sites; optional cross-check with `predict_residue_function` (Activity/Binding/Conserved Site).

## When NOT to use

- Designing a new sequence for a fixed backbone → `proteinmpnn_design_workflow`
- Need experimental structure QA / AlphaFold confidence → `alphafold_database` / `protein_structure_pipeline`
- Homolog discovery → `protein_sequence_similarity_search` or `foldseek_structural_similarity`

## Common mistakes

- Calling non-existent ESM Forge APIs instead of hub tools.
- Using structure models without a real PDB path.
- Presenting zero-shot ranks as measured ΔΔG / activity without caveats.
- Omitting `out_dir` when the session needs a stable artifact path.
