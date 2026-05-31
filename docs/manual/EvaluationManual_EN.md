# Custom Model — Evaluation

> **Where to find it (v2):** Sidebar → **Custom Model → Evaluation** (`/custom-model/evaluation`).
> **Local mode only:** evaluation is disabled in online mode (read-only view).

The Evaluation page measures how a trained model actually performs on held-out data. Pick a model checkpoint, choose a dataset (pre-defined or your own), and the page runs the model end-to-end and reports per-metric numbers.

---

## 1. Inputs

| Field | Notes |
| :--- | :--- |
| **Model Folder** | Pick from the `ckpt/` listing — populated by previous training runs. |
| **Model Path** | The specific checkpoint file (`best_model.pt` by default) inside the chosen folder. Selecting a model auto-fills PLM, Eval Method, Pooling, Problem Type, and Num Labels from the saved config. |
| **PLM** | (Locked once a model is chosen) Must match what was used in training. |
| **Eval Method** | (Locked once a model is chosen) `freeze` · `full` · `plm-lora` · `plm-dora` · `plm-adalora` · `plm-ia3` · `plm-qlora` · `ses-adapter`. Must match training. |
| **Pooling** | (Locked once a model is chosen) `mean` · `attention1d` · `light_attention`. Must match training. |
| **Problem Type** | (Locked if Pre-defined dataset) `single_label_classification` · `multi_label_classification` · `regression`. |
| **Num Labels** | (Locked if Pre-defined dataset) Required for classification. |

> If the Eval Method is `ses-adapter` or the PLM is structure-aware (ProSST / ProtSSN / SaProt), an extra **Structure Seq** picker (foldseek_seq / ss8_seq) and a **PDB Dir** field appear.

---

## 2. Dataset

| Mode | Notes |
| :--- | :--- |
| **Pre-defined** | Pick from the dataset config dropdown — VenusFactory auto-fills problem type, num labels, metrics, and column mappings. |
| **Custom** | Either a Hugging Face `username/dataset_name` or upload a test file (CSV / TSV / XLSX / XLS). Workspace picker supported. |

For custom datasets, set:

- **Problem Type** + **Num Labels**
- **Sequence Column** / **Label Column** — auto-detected from the file; can override
- **Metrics** — multi-select

Click **Preview Dataset** to see the train / val / test split counts and sample rows.

---

## 3. Batch

| Field | Default |
| :--- | :--- |
| **Batch Mode** | `Batch Size Mode` or `Batch Token Mode` |
| **Batch Size / Tokens** | 16 / 10000 |

Use **Batch Token Mode** when sequence lengths vary widely.

---

## 4. Run & Watch

- **Preview Command** — equivalent CLI invocation for reproducibility / scripting.
- **Start** — runs evaluation; the page streams a progress bar and a tail of the eval log.
- **Test Results** — per-metric numbers; click the CSV download to save the raw output.

---

## 5. Supported Metrics

| Abbrev. | Metric | Problem types | Direction |
| :--- | :--- | :--- | :---: |
| **Accuracy** | Correct prediction ratio | Single-/Multi-label classification | ↑ |
| **Recall** | True positive rate | Single-/Multi-label classification | ↑ |
| **Precision** | Positive predictive value | Single-/Multi-label classification | ↑ |
| **F1** | Harmonic mean of precision & recall | Single-/Multi-label classification | ↑ |
| **MCC** | Matthews Correlation Coefficient | Single-/Multi-label classification | ↑ |
| **AUROC** | Area under ROC curve | Single-/Multi-label classification | ↑ |
| **F1_max** | Best F1 across thresholds | Multi-label classification | ↑ |
| **Spearman_corr** | Rank correlation | Regression | ↑ |
| **MSE** | Mean squared error | Regression | ↓ |

---

## 6. Workflow Walkthrough

A complete pass from model + dataset to interpreted results.

### 6.1 Prepare Model & Dataset

1. **Pick a trained model.** It should have been saved under `ckpt/` by the Training page. Note the **PLM**, **Training Method**, and **Pooling** that produced it — the checkpoint config will lock these in for you when selected.
2. **Pick the evaluation dataset.**
   - Same dataset as training? Use the test split — measures held-out performance.
   - New dataset? Use a Custom path / upload — measures out-of-distribution generalization.
   - Make sure the column conventions match (sequence column, label column, structure-seq columns for `ses-adapter`).

### 6.2 Configure

1. **Model Folder + Model Path** — pick from `ckpt/` listing. PLM / Eval Method / Pooling auto-lock.
2. **Dataset Source** — Pre-defined or Custom.
3. **Sequence / Label columns** — auto-detected from uploads; override if your file uses non-standard names.
4. **Metrics** — multi-select. For classification, `accuracy + mcc + f1 + precision + recall + auroc` covers most needs; for regression, `mse + spearman_corr`.
5. **Structure Seq + PDB Dir** — required when the model is `ses-adapter` or a structure-aware PLM.
6. **Batch Mode + Size** — Batch Size Mode for uniform lengths, Batch Token Mode for highly variable lengths.
7. **Preview Dataset** — sanity check the split counts and a few sample rows before running.

### 6.3 Run

1. **Preview Command** — see the equivalent CLI invocation; double-check every flag is intended.
2. **Start** — watch the progress bar; the log streams batch-level updates and any data warnings.
3. **Abort** — graceful stop if something looks off.

### 6.4 Read the Results

1. **Metrics table** — focus on whichever metric is most diagnostic for your task (see §5 table for direction).
2. **Download CSV** — gives per-sample predictions, useful for confusion matrices, error analysis, or feeding into downstream scripts.
3. **Decide.** If performance meets your bar, the model is ready for **Custom Model → Predict**. If not, retrain with adjusted method / hyperparams or revisit the dataset (see QA Q14 / Q15).

---

## 7. Tips

- **Match the training config.** The checkpoint config locks PLM / method / pooling automatically — don't try to evaluate with a different method, you'll get noise.
- **Use Pre-defined datasets for benchmarking.** They keep metric configuration honest and reproducible.
- **Custom test files** should share the same format as the training set (sequence + label columns). For `ses-adapter`, include `foldseek_seq` / `ss8_seq` columns if the model uses them.
- **Big test sets?** Lower the batch size if you OOM; eval doesn't need gradients but the activations still occupy GPU memory.
- **Output CSV** contains per-sample predictions — useful for error analysis and confusion matrices in downstream tools.
