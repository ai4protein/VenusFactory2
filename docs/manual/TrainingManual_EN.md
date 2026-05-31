# Custom Model — Training

> **Where to find it (v2):** Sidebar → **Custom Model → Training** (`/custom-model/training`).
> **Local mode only:** training is disabled in online mode (read-only view).

The Training page lets you fine-tune a protein language model (PLM) on your own dataset — or a pre-defined VenusFactory dataset — without writing a single line of training code. It covers the full lifecycle: dataset upload → method / hyperparameter config → training with live curves → test metrics + CSV export.

## 1. Two Modes

| Mode | When to use |
| :--- | :--- |
| **From Scratch** | Fine-tune a fresh PLM. Requires PLM Model, Output Dir, Output Model Name. |
| **Continue Training** | Resume from a checkpoint. Requires Model Folder (under `ckpt/`) + Checkpoint file. |

In either mode, you can train on either a pre-defined dataset or a custom dataset.

## 2. Supported PLMs

| Family | Sizes available | HF example |
| :--- | :--- | :--- |
| **ESM2** | 8M / 35M / 150M / 650M / 3B / 15B | `facebook/esm2_t33_650M_UR50D` |
| **ESM-1v** | 650M × 5 seeds | `facebook/esm1v_t33_650M_UR90S_1` |
| **ProtBert** | 420M (Uniref100 / BFD) | `Rostlab/prot_bert_bfd` |
| **IgBert / IgBert_unpaired** | 420M | `Exscientia/IgBert` |
| **ProtT5** | 3B / 11B (Uniref50 / BFD) | `Rostlab/prot_t5_xl_uniref50` |
| **IgT5 / IgT5_unpaired** | 3B | `Exscientia/IgT5` |
| **Ankh** | 450M / 1.2B | `ElnaggarLab/ankh-base` |
| **ProSST** | 110M × 7 codebooks (20/128/512/1024/2048/4096 + AE) | `AI4Protein/ProSST-2048` |
| **ProPrime** | 690M | `AI4Protein/Prime_690M` |
| **VenusPLM** | 300M | `AI4Protein/VenusPLM-300M` |
| **PETA** | 80M (base / bpe / unigram) | `AI4Protein/PETA-base` |

> **Pick by GPU memory:** <8 GB → ESM2-8M / 35M, ProSST · 8–16 GB → ESM2-150M / 650M, ProtBert · 24 GB+ → ESM2-3B, ProtT5-XL · multi-GPU → ESM2-15B, ProtT5-XXL.

## 3. Supported Fine-tuning Methods

| Method | Description | Data type |
| :--- | :--- | :--- |
| **freeze** | Freeze PLM, train classifier head only | Sequence |
| **full** | Train all parameters | Sequence |
| **plm-lora** | Low-Rank Adaptation | Sequence |
| **plm-dora** | Weight-Decomposed Low-Rank Adaptation | Sequence |
| **plm-adalora** | Adaptive Low-Rank Adaptation | Sequence |
| **plm-ia3** | Infused Adapter (IA³) | Sequence |
| **plm-qlora** | Quantized LoRA (lowest memory) | Sequence |
| **ses-adapter** | Structure-enhanced sequence adapter | Sequence + Structure |

> **ses-adapter** needs the chosen structure sequence types (`foldseek_seq` / `ss8_seq`).
> Structure-aware PLMs (**ProSST / ProtSSN / SaProt**) need a **PDB Dir** so the model can read each entry's structure.

## 4. Evaluation Metrics

| Abbrev. | Metric | Problem types | Direction |
| :--- | :--- | :--- | :---: |
| **Accuracy** | Proportion of correct predictions | Single-/Multi-label classification | ↑ |
| **Recall** | True positive rate | Single-/Multi-label classification | ↑ |
| **Precision** | Positive predictive value | Single-/Multi-label classification | ↑ |
| **F1** | Harmonic mean of precision & recall | Single-/Multi-label classification | ↑ |
| **MCC** | Matthews Correlation Coefficient | Single-/Multi-label classification | ↑ |
| **AUROC** | Area under ROC curve | Single-/Multi-label classification | ↑ |
| **F1_max** | Best F1 across thresholds | Multi-label classification | ↑ |
| **Spearman_corr** | Rank correlation | Regression | ↑ |
| **MSE** | Mean squared error | Regression | ↓ |

---

## 5. Page Walkthrough

### 5.1 Dataset

| Field | Notes |
| :--- | :--- |
| **Dataset Source** | `Pre-defined` (pick from dropdown — auto-fills problem type / num labels / metrics / sequence + label columns) or `Custom`. |
| **HF ID or local upload** (custom) | Either a `username/dataset_name` from Hugging Face or upload `train / valid / test` files (CSV / TSV / XLSX). Workspace picker supported. |
| **Problem Type** | `single_label_classification` · `multi_label_classification` · `regression`. |
| **Num Labels** | Required for classification. |
| **Sequence Column** | Auto-detected from uploaded files; can be overridden. Default convention: `aa_seq`. |
| **Label Column** | Same. Default: `label`. |
| **Metrics** | Multi-select grid; pick metrics relevant to your problem type. |
| **Preview Dataset** | Quick view of train / val / test counts and sample rows. |

### 5.2 Training Mode

| Field | Notes |
| :--- | :--- |
| **PLM Model** | From dropdown — required for "From Scratch". |
| **Training Method** | One of the methods in §3. |
| **Pooling** | `mean` · `attention1d` · `light_attention`. |
| **Structure Seq** | (ses-adapter / structure PLM only) — `foldseek_seq`, `ss8_seq`. |
| **PDB Dir** | (ProSST / ProtSSN / SaProt only) — local directory with one PDB per sample. |
| **LoRA params** | (plm-lora / qlora / adalora / dora / ia3) — `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules` (default `query,key,value`). |
| **Model Folder + Checkpoint** | (Continue Training only) — pick from `ckpt/` listing. |

### 5.3 Batch

| Field | Notes |
| :--- | :--- |
| **Batch Mode** | `Batch Size Mode` (fixed N samples per batch) or `Batch Token Mode` (~N tokens per batch — better for variable-length sequences). |
| **Batch Size / Tokens** | Defaults: 16 samples / 10000 tokens. |

### 5.4 Hyperparameters

| Field | Default | Notes |
| :--- | :--- | :--- |
| Learning Rate | `5e-4` | Lower for full-finetune; higher for LoRA family. |
| Num Epochs | `20` | Early stopping will usually halt sooner. |
| Patience | `10` | Stop after N epochs of no val-metric improvement. |
| Max Seq Length | `1024` | `-1` for no cap. |
| Scheduler | `linear` | `linear` / `cosine` / `step`. |
| Warmup Steps | `0` | Try 5–10% of total steps. |
| Gradient Accumulation Steps | `1` | Simulate larger batches when memory-bound. |
| Max Grad Norm | `-1` | `-1` = no clipping; 1.0–5.0 if training is unstable. |
| Num Workers | `4` | Data loader workers. |
| Monitored Metric | (depends on task) | Drives early stopping + best model selection. |
| Monitored Strategy | `max` | `max` or `min` depending on metric. |

### 5.5 Output

| Field | Notes |
| :--- | :--- |
| **Save Directory** | Default `ckpt/`. |
| **Output Model Name** | Default `demo/best_model.pt`. |
| **Enable W&B Logging** | Optional — set `wandb_project`, `wandb_entity` when on. |

### 5.6 Run & Watch

- **Preview Command** — shows the equivalent CLI invocation (useful for reproducibility / scripting).
- **Start** — kicks off training; the right side panel streams training & validation loss curves, per-epoch metrics, model statistics (total / trainable param count + %), and a progress log.
- **Abort** — gracefully stops the active run.
- **Test Results** — final per-metric numbers + CSV download.

---

## 6. Custom Dataset Format

When you go the Hugging Face / file-upload route, your data should have these columns:

| Column | Required? | Notes |
| :--- | :---: | :--- |
| `aa_seq` (or your chosen sequence column) | ✓ | Amino acid sequence, single-letter codes. |
| `label` (or your chosen label column) | ✓ | Numeric for regression; integer / list for classification. |
| `foldseek_seq` | optional | Needed for `ses-adapter` with `foldseek_seq` enabled. |
| `ss8_seq` | optional | Needed for `ses-adapter` with `ss8_seq` enabled. |
| `name` / `id` | optional | Sample identifier. |

Splits expected: `train`, `validation`, `test`. The Custom dataset upload accepts three separate files; the Pre-defined dataset path expects HuggingFace dataset structure.

### 6.1 Label formats per problem type

**Single-label classification** — `label` is an integer class index (starting from 0):

```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,1
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,0
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,2
```

**Multi-label classification** — `label` is a comma-separated string of present class indices (quoted so commas don't break CSV parsing):

```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,"373,449,584,674,780,883,897,911,1048,1073,1130,1234"
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,"15,42,87,103,256"
```

**Regression** — `label` is a float:

```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,0.75
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,-1.2
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,3.45
```

### 6.2 Adding structure columns (ses-adapter)

When training with `ses-adapter`, add `foldseek_seq` and/or `ss8_seq` as additional columns. Each row must align character-by-character with `aa_seq`:

```csv
name,aa_seq,foldseek_seq,ss8_seq,label
Q9LSD8,MPEEDLVELKFR...,DPPQLWAFAWEA...,LLLLLLEEEEEE...,0
```

### 6.3 Uploading to Hugging Face

1. Create three separate files: `train.csv`, `validation.csv`, `test.csv`.
2. Push them to a new HuggingFace dataset repository.
3. Reference the dataset in the Training page as **Custom Path** = `username/dataset_name`.

---

## 7. Workflow Summary

1. **Dataset** — pre-defined or custom; preview before training.
2. **Model + Method** — pick PLM, fine-tuning method, pooling. Configure LoRA / structure-seq / PDB Dir if needed.
3. **Batch + Hyperparams** — start with defaults, tune only when needed.
4. **Output** — pick save path, optionally enable W&B.
5. **Preview Command** — sanity check.
6. **Start** — monitor curves & metrics; **Abort** if anything looks off.
7. **Test Results** — download CSV; the saved model is now usable from **Custom Model → Evaluation** and **Custom Model → Predict**.
