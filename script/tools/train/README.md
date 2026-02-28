# Training Scripts

This directory contains shell scripts for training protein language models (PLMs) and structure-aware PLMs in VenusFactory. All scripts call `src/train.py` with different configurations.

---

## Table of Contents

- [Common Arguments](#common-arguments)
- [Script Overview](#script-overview)
- [Vanilla / Freeze Training](#train_plm_vanillash---vanilla--freeze-training)
- [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
- [Structure-Aware Training](#structure-aware-training)
- [Dataset and Model Compatibility](#dataset-and-model-compatibility)
- [Quick Test](#quick-test)
- [HuggingFace Mirror](#huggingface-mirror)

---

## Common Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--plm_model` | HuggingFace model name or path | `facebook/esm2_t33_650M_UR50D` |
| `--dataset_config` | Path to dataset JSON config | `data/DeepLocBinary/DeepLocBinary_AlphaFold2_HF.json` |
| `--pdb_dir` | Directory of PDB files (for structure models) | `dataset/DeepET/alphafold_pdb` |
| `--batch_token` | Max tokens per batch (variable-length batching) | `12000` |
| `--batch_size` | Fixed number of samples per batch | `12` |
| `--max_seq_len` | Max sequence length when using `batch_size` | `1024` |
| `--learning_rate` | Learning rate | `5e-4` |
| `--gradient_accumulation_steps` | Gradient accumulation steps | `8` |
| `--num_epochs` | Number of training epochs | `100` |
| `--patience` | Early stopping patience | `10` |
| `--output_dir` | Directory to save checkpoints | `ckpt/debug/DeepLocBinary/esm2_t33_650M_UR50D` |
| `--output_model_name` | Checkpoint filename | `af2_lr5e-4_bt12k_ga8.pt` |
| `--training_method` | Training strategy (see below) | `freeze`, `plm-lora`, etc. |
| `--quick_test` | Truncate train/val/test for a quick run | (flag) |

You must provide either `--batch_token` or `--batch_size`. Dataset configs (e.g. `problem_type`, `metrics`, `monitor`) are read from the JSON file.

---

## Script Overview

| Script | Training method | Use case |
|--------|-----------------|----------|
| `train_plm_vanilla.sh` | `freeze` (default) | Standard PLM + adapter; PLM frozen |
| `train_plm_lora.sh` | `plm-lora` | LoRA on PLM |
| `train_plm_qlora.sh` | `plm-qlora` | Quantized LoRA (4-bit), lower GPU memory |
| `train_plm_dora.sh` | `plm-dora` | DoRA on PLM |
| `train_plm_adalora.sh` | `plm-adalora` | AdaLoRA on PLM |
| `train_plm_ia3.sh` | `plm-ia3` | IA3 on PLM (e.g. T5) |
| `train_plm_ses-adapter.sh` | `ses-adapter` | Sequence + structure (foldseek, ss8) with cross-attention |
| `train_splm.sh` | `freeze` | Structure-aware PLMs: **ProSST**, **SaProt** (need `pdb_dir`) |

---

## train_plm_vanilla.sh — Vanilla / Freeze Training

**What it does:** Trains a small adapter on top of a frozen PLM. Best for standard sequence-only tasks.

**Supported PLMs:** ESM2 (facebook), ProtBert/ProtT5/Ankh (RostLab), etc.

**Batching:**
- **By token:** `--batch_token 12000` (recommended for variable-length sequences).
- **By sample:** `--batch_size 12 --max_seq_len 1024`.

**Example (batch by token):**
```bash
dataset=DeepLocBinary
pdb_type=AlphaFold2
plm_model=esm2_t30_150M_UR50D
python src/train.py \
    --plm_model facebook/$plm_model \
    --dataset_config data/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 8 \
    --batch_token 12000 \
    --output_dir debug/$dataset/$plm_model \
    --output_model_name af2_lr5e-4_bt12k_ga8.pt
```

**Datasets:** Use configs under `data/<Dataset>/` with names like `<Dataset>_<pdb_type>_HF.json` (e.g. `DeepLocBinary_AlphaFold2_HF.json`). Some datasets have no structure (e.g. FLIP_AAV, FLIP_GB1).

---

## Parameter-Efficient Fine-Tuning (PEFT)

Scripts: `train_plm_lora.sh`, `train_plm_qlora.sh`, `train_plm_dora.sh`, `train_plm_adalora.sh`, `train_plm_ia3.sh`.

**Common:** All pass `--training_method <method>` and usually `--lora_target_modules ...`. The PLM is partially fine-tuned instead of fully frozen.

- **plm-lora / plm-qlora / plm-dora / plm-adalora:**  
  - ESM/BERT: `--lora_target_modules query key value`  
  - T5/Ankh: `--lora_target_modules q k v`
- **plm-ia3:** For T5-style models; also set `--feedforward_modules wo` and target `q k v wo` as in the script.

**QLoRA:** Use when GPU memory is limited; loads the base model in 4-bit.

**HuggingFace mirror (optional):**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
Then run the script as usual.

---

## Structure-Aware Training

### train_plm_ses-adapter.sh — SES-Adapter

**What it does:** Uses both **sequence** (from the PLM) and **structure** (foldseek 3Di, ss8) via extra structure embeddings and cross-attention. The dataset (or pipeline) must provide structure columns or they are generated from `pdb_dir`.

**Required:** `--structure_seq foldseek_seq,ss8_seq` (and optionally `--pdb_dir` if those columns are not in the HuggingFace dataset).

**Example:**
```bash
python src/train.py \
    --plm_model facebook/esm2_t30_150M_UR50D \
    --dataset_config data/DeepLocBinary/DeepLocBinary_AlphaFold2_HF.json \
    --structure_seq foldseek_seq,ss8_seq \
    --learning_rate 5e-4 \
    --batch_token 12000 \
    --gradient_accumulation_steps 8 \
    --output_dir debug/DeepLocBinary/esm2_t30_150M_UR50D \
    --output_model_name ses-adapter_AlphaFold2_lr5e-4_bt12k_ga8.pt
```

### train_splm.sh — ProSST & SaProt

**What it does:** Trains **structure-aware PLMs** that take structure as part of the input:

- **ProSST:** Expects `stru_token_{vocab}` (e.g. `stru_token_2048`). If the HuggingFace dataset does not have it, pass `--pdb_dir`; the pipeline will generate ProSST structure tokens from PDBs and merge by name (with containment matching for varied PDB naming).
- **SaProt:** Expects **uppercase amino acid + lowercase Foldseek 3Di** per residue (e.g. `MdEvVp...`). The pipeline needs `foldseek_seq`; if missing, it is generated from `--pdb_dir` and merged.

**ProSST models:** `AI4Protein/ProSST-20`, `-128`, `-512`, `-1024`, `-2048`, `-4096`.

**SaProt models:** `westlake-repl/SaProt_650M_AF2`, `westlake-repl/SaProt_650M_PDB`, `westlake-repl/SaProt_35M_AF2`.

**Tasks:** Both support **protein-level** and **residue-level** tasks (e.g. VenusX residue labels). Use the appropriate dataset config (e.g. `VenusX_Res_Act_MP30_HF.json` for residue-level).

**Examples in the script:**
1. **SaProt, protein-level:** DeepET_Topt + `pdb_dir` for foldseek → combined aa+3Di sequence.
2. **ProSST, residue-level:** VenusX_Res_Act_MP30 + `pdb_dir` for ProSST tokens.

**VenusX:** Dataset configs use `structure_merge_key_format: "{interpro_id}-{uid}"` so PDB names (e.g. `IPR000126_A4Y3F4.pdb`) are matched to HF rows by containment. Only PDBs that match the dataset are processed.

---

## Dataset and Model Compatibility

- **Sequence-only (no structure):** Use `train_plm_vanilla.sh` or any PEFT script with a config that has no structure columns (e.g. FLIP_AAV, FLIP_GB1).
- **With structure (PDB dir):** Use a config that references structure columns and, if needed, set `--pdb_dir`. For ProSST/SaProt, `pdb_dir` is typically required so the pipeline can generate `stru_token_*` or `foldseek_seq` and merge.
- **Residue-level:** Use configs with `problem_type` containing `residue` (e.g. `residue_single_label_classification`) and the correct sequence column (e.g. `seq_full` for VenusX).

Dataset config JSONs under `data/` define `dataset`, `problem_type`, `metrics`, `monitor`, `sequence_column_name`, `label_column_name`, and optionally `pdb_dir`, `structure_seq`, `structure_merge_key_format`, etc.

---

## Quick Test

To run a short training for debugging (small subset of train/val/test):

```bash
python src/train.py \
    ... \
    --quick_test
```

Optionally set in config or CLI: `--max_train_samples`, `--max_validation_samples`, `--max_test_samples` (defaults are applied when `--quick_test` is used). Structure generation (ProSST/SaProt) then runs only on this subset.

---

## HuggingFace Mirror

If downloads from HuggingFace are slow or blocked, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then run any training script. The scripts that use non-default mirrors often include this line at the top.

---

## Summary

| Goal | Script | Required / Notes |
|------|--------|------------------|
| Standard sequence PLM, frozen | `train_plm_vanilla.sh` | `--dataset_config`, `--plm_model`, `--batch_token` or `--batch_size` |
| Fine-tune PLM with LoRA/QLoRA/DoRA/AdaLoRA/IA3 | `train_plm_*.sh` (lora/qlora/dora/adalora/ia3) | `--training_method`, `--lora_target_modules` (and for IA3, `--feedforward_modules`) |
| Sequence + structure (foldseek, ss8) | `train_plm_ses-adapter.sh` | `--structure_seq foldseek_seq,ss8_seq`, optional `--pdb_dir` |
| ProSST (structure tokens) | `train_splm.sh` | `--plm_model` ProSST, `--pdb_dir` if HF has no `stru_token_*` |
| SaProt (aa + 3Di sequence) | `train_splm.sh` | `--plm_model` SaProt, `--pdb_dir` for `foldseek_seq` |

All scripts are meant to be edited (dataset, plm_model, paths, lr, batch size, etc.) and then run from the **repository root** (where `src/train.py` lives).
