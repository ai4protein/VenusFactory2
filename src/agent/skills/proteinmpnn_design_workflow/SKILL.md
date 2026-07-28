---
name: proteinmpnn_design_workflow
description: >-
  ProteinMPNN inverse folding: design or score sequences on a fixed backbone. Use when the user wants sequence design from PDB, interface/binder design, homomer symmetry, or fixed catalytic residues. Do NOT use for zero-shot mutation ranking on a wild-type sequence (zero_shot_mutation_workflow) or de novo fold hallucination without a backbone.
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# ProteinMPNN Design Workflow

## Overview

Orchestrates VenusFactory denovo tools for backbone-conditioned sequence design and scoring. Residue indices in `fixed_residues_json` are **1-indexed**.

## Project Tools (VenusFactory2)

| Tool | Args | When |
|------|------|------|
| **proteinmpnn_sequence_design_from_structure** | `pdb_path`; optional `designed_chains`, `fixed_chains`, `fixed_residues_json`, `homomer`, `num_sequences`, `temperatures`, `out_dir` | Design sequences |
| **proteinmpnn_sequence_scoring_from_structure** | `pdb_path`; optional `fasta_path`, `designed_chains`, `out_dir` | Score designs / native |
| **pdb_chain_sequences** / **check_pdb_apo** | PDB path | Prep / inspect |
| **zero_shot_mutation_structure_prediction** | designed PDB context | Optional post-design ranking |

## Workflow patterns

| Goal | Key args |
|------|----------|
| Single-chain redesign | `pdb_path` only |
| Design chain B against fixed A | `designed_chains=["B"]`, `fixed_chains=["A"]` |
| Keep catalytic residues | `fixed_residues_json='{"A":[57,102,195]}'` |
| Homomer | `designed_chains=["A","B","C"]`, `homomer=true` |

1. Validate PDB (`check_pdb_apo` if ligand-free backbone expected).
2. Design with explicit `out_dir`.
3. Score designs with `proteinmpnn_sequence_scoring_from_structure`.
4. Optional: structure prediction / property checks on top sequences.

## Common mistakes

- Passing nearly all residues as fixed when intending to redesign the whole chain (or the inverse) — check tool log warnings.
- 0-based residue indices in `fixed_residues_json`.
- Expecting affinity/ΔG; ProteinMPNN gives sequence likelihoods, not binding free energy.
