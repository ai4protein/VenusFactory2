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
| **generate_training_config** | `csv_file` **or** `dataset_path`; optional valid/test CSV; `parameters`; `output_name` | Build training JSON from explicit settings; no nested LLM |
| **train_protein_model** | `config_path` | Run configured training; research contexts stage a candidate for independent review |
| **register_trained_model** | `config_path`; optional `model_id` / `output_dir` / `model_path` | Store a checkpoint; research contexts cannot self-publish |
| **list_trained_models** | — | List cross-session registered models |
| **protein_model_predict** | `config_path` **or** `model_id` + `sequence` **or** `csv_file` | Inference |
| **write_session_file → run_session_python** | Write: `path`, `content`; run: `path`, optional `timeout_seconds` | Splits, metrics plots, CSV cleanup (session sandbox) |

**Mode:** local only. Online mode disables these tools (and VenusMine / FoldSeek discovery). Science Agent reaches train tools via MCP; Expert via the LangGraph tool hub.

## Workflow

1. Inspect the supplied data and group related sequences to avoid leakage. Write the chosen split code with `write_session_file`, then execute `run_session_python` using that saved relative path.
2. Translate the user's requirements into `parameters`, for example `{"plm_model":"ESM2-8M","num_epochs":10,"learning_rate":0.0005,"batch_size":8}`. Call `generate_training_config` and inspect its `resolved_parameters` and detected task type before training. Omitted settings use the returned defaults. Do not pass free-text `user_requirements`.
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

### Session execution
Use `write_session_file(path, content)` to write actual Python source and `run_session_python(path)` to execute it. Paths are relative to the server-assigned session workspace. Installed Python packages are available; repository source paths in this guide are host implementation references, not guest imports. Networking uses registered database/search/download tools; the analysis sandbox has no network access. Verify exit_code and actual output files.
