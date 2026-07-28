---
name: protein_property_prediction
description: >-
  Physicochemical properties, surface/SS features, and finetuned protein/residue function prediction. Use when the user asks for solubility, optimal temperature, activity/binding/conserved sites, RSA/SASA/secondary structure, or property tables from FASTA/PDB. Do NOT use for zero-shot mutation ranking (zero_shot_mutation_workflow) or custom model training (train tools / future finetune workflow).
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# Protein Property & Function Prediction

## Overview

Orchestrates VenusFactory `predict` tools: FASTA physicochemical features, PDB RSA/SASA/SS, and finetuned sequence-level / residue-level function heads (Ankh / ESM2 / ProtT5 adapters under `ckpt/`).

## Project Tools (VenusFactory2)

| Tool | Args | When |
|------|------|------|
| **calculate_physchem_from_fasta** | `fasta_file`, optional `out_dir` | Length, MW, pI, gravy, etc. |
| **calculate_rsa_from_pdb** | `pdb_file`, `chain_id` | Relative solvent accessibility |
| **calculate_sasa_from_pdb** | `pdb_file` | Absolute SASA |
| **calculate_ss_from_pdb** | `pdb_file`, `chain_id` | Secondary structure |
| **predict_protein_function** | `fasta_file`, `task`, `model_name` | Solubility, Optimal Temperature, … |
| **predict_residue_function** | `fasta_file`, `task`, `model_name` | Activity / Binding / Conserved Site / Motif |
| **predict_structure_esmfold** | `sequence` | Need PDB before RSA/SASA/SS |

## Workflow

### Properties from sequence

1. `calculate_physchem_from_fasta`
2. Optional `predict_protein_function` with an explicit `task` matching platform dataset names.

### Structure features

1. Obtain PDB (`protein_structure_pipeline` or user upload).
2. Run `calculate_rsa_from_pdb` / `calculate_sasa_from_pdb` / `calculate_ss_from_pdb` as needed.

### Residue sites

1. `predict_residue_function` with task in {`Activity Site`, `Binding Site`, `Conserved Site`, `Motif`}.
2. Cross-check with InterPro (`interpro_domain_annotation`) before proposing edits to catalytic residues.

## Common mistakes

- Inventing `task` names not present in platform `constant.json` / ckpt mapping.
- Running RSA tools on FASTA without a structure.
- Treating finetuned scores as wet-lab labels without reporting model/task.
