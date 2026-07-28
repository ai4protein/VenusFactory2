---
name: venus_finetune_workflow
description: >-
  Fine-tune and run custom protein models on VenusFactory (CSV/HF → config → train → predict). Use when the user brings labeled sequences, wants adapter training (ProtT5/ESM2/Ankh/QLoRA notes), or batch inference with a trained config. Do NOT use for zero-shot mutation without labels (zero_shot_mutation_workflow) or built-in function heads already covered by predict_protein_function.
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# Venus Fine-Tune Workflow

## Overview

Orchestrates training hub tools. Expect CSV columns such as `aa_seq` / `sequence` + `label`. GPU recommended; training can be long — set clear success criteria (metrics file / checkpoint exists).

## Project Tools (VenusFactory2)

| Tool | Args | Purpose |
|------|------|---------|
| **generate_training_config** | `csv_file` **or** `dataset_path`; optional valid/test CSV; `user_requirements`; `output_name` | Build training JSON |
| **train_protein_model** | `config_path` | Run training; **auto-registers** under `ckpt/user_trained/<model_id>/` |
| **register_trained_model** | `config_path`; optional `model_id` / `output_dir` / `model_path` | Explicit re-register (usually unnecessary after train) |
| **list_trained_models** | — | List cross-session registered models |
| **protein_model_predict** | `config_path` **or** `model_id` + `sequence` **or** `csv_file` | Inference |
| **agent_generated_code** | `task_description`, `input_files`, `output_dir` | Splits, metrics plots, CSV cleanup (Expert only; not MCP) |

**Mode:** local only. Online mode disables these tools (and VenusMine / FoldSeek discovery). Science Agent reaches train tools via MCP; Expert via the LangGraph tool hub.

## Workflow

1. Validate/split data with `agent_generated_code` if needed (70/15/15) — Expert path.
2. `generate_training_config` with explicit `user_requirements` (model, epochs, LR, QLoRA…).
3. `train_protein_model` on the returned config path (`dependency:step_N:file_path`). On success, use `registered_model_id` / `registered_config_path` from the result.
4. `protein_model_predict` with the registered `model_id` (or config path) for hold-out or new sequences — works across sessions.
5. Figure: loss/accuracy curves from training metrics (dpi≥300).

## When NOT to use

- Unlabeled mutagenesis ranking → `zero_shot_mutation_workflow`
- Off-the-shelf solubility/temp heads → `predict_protein_function` (`protein_property_prediction`)

## Common mistakes

- Omitting both `csv_file` and `dataset_path`
- Predicting without the **same** `config_path` used for training
- Writing artifacts outside the session `output_dir`
