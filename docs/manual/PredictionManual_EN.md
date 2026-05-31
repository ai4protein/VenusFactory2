# Custom Model — Prediction

> **Where to find it (v2):** Sidebar → **Custom Model → Predict** (`/custom-model/predict`).
> **Local mode only:** prediction is disabled in online mode (read-only view).

The Predict page applies one of your trained models to new sequences — either one sequence at a time, or a whole batch.

---

## 1. Pick a Model

| Field | Notes |
| :--- | :--- |
| **Model Folder** | Pick from the `ckpt/` listing — populated by previous training runs. |
| **Model Path** | The specific checkpoint file (`best_model.pt` by default) inside the chosen folder. Selecting a model auto-fills the rest of the model config. |

Selecting a model **locks** the following fields to the values from the checkpoint config (you can't predict with mismatched settings):

- **PLM** — must match what was used in training
- **Eval Method** — `freeze` · `full` · `plm-lora` · `plm-dora` · `plm-adalora` · `plm-ia3` · `plm-qlora` · `ses-adapter`
- **Pooling** — `mean` · `attention1d` · `light_attention`
- **Problem Type** — `single_label_classification` · `multi_label_classification` · `regression`
- **Num Labels** — relevant for classification

If the Eval Method is `ses-adapter` or the PLM is structure-aware (ProSST / ProtSSN / SaProt), extra inputs appear (Structure Seq picker, PDB Dir, and/or Foldseek / SS8 sequence text areas for Single mode).

---

## 2. Pick a Prediction Mode

Toggle between **Single** and **Batch**.

### 2.1 Single

| Field | Notes |
| :--- | :--- |
| **AA Sequence** | Paste a single amino acid sequence (single-letter codes). |
| **Foldseek Sequence** | (Only when `ses-adapter` + `foldseek_seq` selected) Matching foldseek-encoded string. |
| **SS8 Sequence** | (Only when `ses-adapter` + `ss8_seq` selected) Matching 8-class secondary structure string. |

Click **Predict** to get an instant result.

### 2.2 Batch

| Field | Notes |
| :--- | :--- |
| **Source** | `Upload file` (upload a CSV/TSV/XLSX) · `Paste FASTA` (paste FASTA-formatted content in a textarea) · `Path` (point to an existing file path on the server). |
| **Workspace** | The upload source also supports picking from Workspace. |
| **Batch Size** | Number of sequences scored per forward pass. Lower it if you OOM on long sequences. |

For file inputs (Upload / Path), the expected columns:

| Column | Required? | Notes |
| :--- | :---: | :--- |
| `aa_seq` | ✓ | Amino acid sequence. |
| `id` / `name` | optional | Sample identifier. |
| `foldseek_seq` | optional | Required when `ses-adapter` + `foldseek_seq` is enabled. |
| `ss8_seq` | optional | Required when `ses-adapter` + `ss8_seq` is enabled. |

---

## 3. Run & Watch

- **Preview Command** — equivalent CLI invocation.
- **Start** — kicks off prediction; the page streams a progress bar and a tail of the prediction log.
- **Logs** panel shows per-batch progress and any data issues (bad sequence chars, missing structure-seq, etc.).

---

## 4. Results

| Problem type | What you get |
| :--- | :--- |
| **Single-label classification** | Predicted class + per-class probability distribution. |
| **Multi-label classification** | Per-label 0/1 + per-label probability. |
| **Regression** | Predicted numeric value. |

For batch mode, results are saved as a CSV containing every input sample plus the prediction columns. Download from the results panel.

---

## 5. Workflow Walkthroughs

### 5.1 Single-Sequence Prediction

Quick check on one protein.

1. **Pick the model** — Model Folder + Model Path; the rest of the model config auto-locks.
2. **Switch to Single mode.**
3. **Paste the AA sequence** in the text box (single-letter codes only).
4. (For `ses-adapter` models) paste the matching **Foldseek** and/or **SS8** sequence.
5. **Predict** — result appears in the panel:
   - Single-label classification → predicted class + per-class probabilities.
   - Multi-label classification → 0/1 + probability per label.
   - Regression → predicted numeric value.
6. **Abort** if a run misbehaves.

### 5.2 Batch Prediction

Scale to hundreds or thousands of sequences.

1. **Pick the model** — same as Single.
2. **Switch to Batch mode.**
3. **Choose a Source:**
   - `Upload file` — drag a CSV / TSV / XLSX, or pick from Workspace.
   - `Paste FASTA` — paste a multi-record FASTA into the textarea.
   - `Path` — point at an existing file path on the server (fastest for huge files).
4. **Check the preview** of input rows / records to confirm the columns map correctly.
5. **Set Batch Size** — 16–32 usually works; lower it if you OOM with long sequences.
6. **Start** — the progress bar shows total / processed / ETA.
7. **Download CSV** — every input row plus the prediction columns.

---

## 6. Tips

- **Sequence sanity.** Strip stop codons, line breaks, and non-AA characters before pasting.
- **Long sequences.** If your training capped sequence length, predictions on longer inputs may silently truncate — check the logs.
- **Structure-aware models.** Don't forget the structure-side inputs (PDB Dir for batch, Foldseek/SS8 text for single).
- **Use Batch for >10 sequences.** Single mode rebuilds the GPU pipeline on every call; batch amortizes that cost.
- **Custom workflow:** if you need to mass-screen, generate the input CSV from a database query or a script and feed it via **Source = Path** — much faster than the upload UI.
