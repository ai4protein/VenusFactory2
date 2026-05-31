# VenusFactory2 Frequently Asked Questions (FAQ)

## Installation and Environment

### Q1: How do I install VenusFactory2?

Follow the **Installation** section of `README.md`. The supported paths are:

1. **conda + pip** (default for most users):
   ```bash
   git clone https://github.com/AI4Protein/VenusFactory2.git && cd VenusFactory2
   conda create -n venus python=3.12 && conda activate venus
   pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
       -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
   pip install -r requirements.txt
   ```

2. **uv** (faster, for development): `python install.py --type cu128` then `source .venv/bin/activate`.

3. **Docker**: `cp .env.example .env && docker compose --profile gpu up -d --build`.

Verify with `python scripts/check_env.py`.

### Q2: I hit "Could not find a specific dependency" during installation.

Options, in order:

1. Try installing the problematic dependency individually:
   ```bash
   pip install <name>
   ```
2. If it's CUDA-related, make sure your torch matches your CUDA. The default for VenusFactory2 is **CUDA 12.8 + torch 2.8.0**:
   ```bash
   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   ```
3. Some packages need system libraries. On Ubuntu 24.04 the most common ones are needed by the `pycairo` build (pulled in via `xhtml2pdf` for PDF export):
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libcairo2-dev libxml2-dev pkg-config
   ```
   Alternatively on conda: `conda install -c conda-forge pycairo` to skip the source build entirely.

### Q3: How do I check that CUDA is set up correctly?

1. Driver check: `nvidia-smi` (driver should support CUDA 12.8 → driver ≥ 555.x).
2. PyTorch check:
   ```python
   import torch
   print(torch.__version__)            # should be 2.8.0+cu128
   print(torch.cuda.is_available())    # True
   print(torch.cuda.device_count())    # number of visible GPUs
   print(torch.cuda.get_device_name(0))
   ```
3. End-to-end check: `python scripts/check_env.py` runs the imports above plus a CUDA matmul smoke test.

If `cuda.is_available()` is False, your torch wheel was likely installed without CUDA. Reinstall with the right `--index-url` (see Q2).

## Hardware and Resources

### Q4: I get "CUDA out of memory" during training.

In order of effectiveness:

1. **Reduce batch size** — most direct fix. Halve `Batch Size` (or `Tokens per Batch`).
2. **Use a smaller model** — e.g. switch from `ESM2-650M` to `ESM2-150M` or `ESM2-35M`. ProSST / VenusPLM are also lightweight options.
3. **Use a PEFT method** — `plm-lora`, `plm-qlora`, `plm-dora`, `plm-ia3` train far fewer parameters.
4. **Enable gradient accumulation** — set `gradient_accumulation_steps` to 2 / 4 / 8 to keep effective batch size while shrinking memory.
5. **Reduce `Max Seq Length`** if your data allows it (truncate signal peptides / long disordered tails).

### Q5: How do I pick a batch size?

1. **Start small, grow.** Begin at 4 or 8; raise until you're near GPU memory limit.
2. **Reference range.** For most protein PLMs, batch 16–64 is typical, but it depends heavily on GPU memory and sequence length.
3. **Trade-off.** Larger batches → more stable gradients but may need a higher learning rate.
4. **OOM rule.** If you OOM, halve first; only then think about other tweaks.

## Dataset

### Q6: How do I prepare a custom dataset?

For the **Custom Model → Training** page:

1. **Columns.** At minimum a sequence column (default `aa_seq`) and a label column (default `label`). For `ses-adapter` add `foldseek_seq` and/or `ss8_seq`. Regression: numeric label. Classification: integer (single-label) or list (multi-label).
2. **Splits.** Three files — `train`, `validation`, `test` — uploaded separately, or a HuggingFace dataset with those splits.
3. **Path on HF.** Reference as `username/dataset_name`.
4. **Config.** On the page, set Problem Type, Num Labels, and Metrics; or pick a Pre-defined dataset which auto-fills these.

### Q7: I get a format error when uploading my dataset.

Common issues:

1. **Wrong column names.** Ensure your sequence column is `aa_seq` (or whatever you set in the form) and the label column is `label`.
2. **Bad sequence characters.** Only the 20 standard AA letters (`ACDEFGHIKLMNPQRSTVWY`) plus optionally `X` for unknown. Strip whitespace, line breaks, and other characters.
3. **Encoding.** Save as UTF-8.
4. **Delimiter.** CSV uses comma; TSV uses tab — name your file accordingly.
5. **Missing values.** Drop rows with missing sequences or labels.

### Q8: My dataset is huge and the system is slow.

1. **Subset for prototyping.** Validate the pipeline on 1–5k rows first.
2. **Batch Token Mode** — for variable-length sequences, this packs more efficiently than fixed Batch Size.
3. **Preprocess off-line.** Remove unused columns, deduplicate, and shard into multiple files.
4. **More memory.** If your machine has spare RAM, raise `num_workers` in Training; if it's swapping, lower it.

## Training

### Q9: My training got interrupted. How do I recover?

The training save path is whatever you set in **Save Directory** + **Output Model Name** (default `ckpt/demo/best_model.pt`).

1. On the Training page, switch **Training Mode** from `From Scratch` to `Continue Training`.
2. Pick the **Model Folder** that contains your last checkpoint.
3. Pick the **Checkpoint** file from the dropdown.
4. Hit Start — training resumes from that checkpoint's epoch and optimizer state.

> The system keeps a "best so far" checkpoint per run (filtered by your monitored metric). Periodic step-based snapshots are not enabled by default.

### Q10: Training is too slow.

1. **Use a PEFT method** (`plm-lora`, `plm-qlora`) — orders of magnitude fewer trainable params.
2. **Lower `Max Seq Length`** if your task allows.
3. **Use a smaller PLM** — ESM2-150M is often within 1–2 points of ESM2-650M on classification tasks but trains much faster.
4. **Move data to SSD** — protein datasets with PDB files are often I/O bound.
5. **`Batch Token Mode`** for variable-length data — better GPU utilization than fixed batch size.

### Q11: Loss isn't going down, or I see NaN values.

For loss not decreasing:

- **Learning rate too high** — try `1e-5` instead of `5e-4` if doing `full` fine-tuning.
- **Wrong optimizer** — full fine-tuning typically wants AdamW.
- **Data issues** — check for label noise, mislabeled samples, off-by-one indexing.

For NaN:

- **Gradient explosion** — set **Max Grad Norm** to 1.0–5.0.
- **Learning rate too high** — drop by 10×.
- **fp16 instability** — try fp32 if you suspect numerical underflow.
- **Bad data** — extreme values in regression labels can produce NaN; cap or normalize them.

### Q12: How do I avoid overfitting?

1. **More data / data augmentation.**
2. **Regularization** — dropout (0.1–0.3), weight decay, or early stopping via `Patience`.
3. **Smaller model** — fewer parameters, or use `freeze` to lock the PLM.
4. **Cross-validation** — train multiple folds and pick the median.

## Evaluation

### Q13: Which evaluation metric should I focus on?

| Task | Default focus |
| :--- | :--- |
| Balanced classification | **Accuracy**, **F1** |
| Imbalanced classification | **F1**, **MCC**, **AUROC** |
| Multi-label classification | **F1_max**, per-label AUROC |
| Regression | **Spearman_corr**, **MSE** |

The *most important* metric depends on your downstream use. For drug screening you might prioritize true-positive rate; for ranking candidates, Spearman.

### Q14: Evaluation results are poor. What now?

1. **Data quality** — check for label noise, distribution shift between train and test.
2. **Model + method choices** — try a different PLM, or change from `freeze` to `plm-lora`.
3. **More features** — structure-aware methods (`ses-adapter`, ProSST, ProtSSN) often beat sequence-only on structurally-dependent tasks.
4. **Ensemble** — train 3–5 seeds and average predictions.

### Q15: Test-set performance is much worse than validation.

1. **Distribution shift** — your test set has families / properties not seen in training. Use stratified splits.
2. **Overfit to val** — repeated model selection against val acts like training on it. Hold out a *separate* test set you only touch once.
3. **Data leakage** — duplicates between train and test. Cluster by sequence identity before splitting.
4. **Small test set** — re-run with different seeds and check variance.

## Prediction

### Q16: How do I speed up prediction?

1. **Use Batch mode** in **Custom Model → Predict** — it amortizes GPU setup across many sequences.
2. **Smaller model** — sometimes a `ESM2-150M` model is "good enough" and 4× faster than `650M`.
3. **GPU not CPU** — make sure `torch.cuda.is_available()` returns True.
4. **Lower `Max Seq Length`** if your inputs allow.

### Q17: My predictions are far off from expectations.

Possible causes:

1. **Wrong model / wrong checkpoint** — Predict locks PLM / Method / Pooling from the checkpoint; confirm you picked the right one.
2. **Out-of-distribution sequence** — your input may be much longer, much shorter, or from a different organism / family than training data.
3. **Missing structure inputs** — for `ses-adapter` / ProSST / ProtSSN / SaProt models, you must provide the structure side (PDB Dir or Foldseek/SS8 text).
4. **Sequence formatting** — non-AA characters, lowercase, gaps, or stop codons. Strip them first.

### Q18: How do I batch-predict many sequences efficiently?

Use **Custom Model → Predict** in **Batch** mode:

1. **Prepare the input file** — a CSV/TSV/XLSX with at least:
   - `aa_seq` — the amino acid sequence
   - `id` / `name` — optional identifier
   - `foldseek_seq` / `ss8_seq` — only needed when the model is `ses-adapter` with those structure-seq types enabled
2. **Load the model** — pick the **Model Folder** + **Model Path** (the saved config locks PLM, method, pooling).
3. **Switch to Batch mode** — choose `Upload file` (browser upload), `Paste FASTA`, or `Path` (point at a file already on the server — fastest for very large lists).
4. **Set Batch Size** — 16–32 is a good default. Lower it if you OOM on long sequences; raise it if you have GPU headroom.
5. **Start** — the page streams a progress bar and a tail of the prediction log.
6. **Result CSV** — contains every input sample plus the prediction columns; download from the results panel.

## Model and Result Issues

### Q19: Which pre-trained model should I choose?

| Situation | Recommended |
| :--- | :--- |
| General-purpose, balanced compute / quality | **ESM2-650M** |
| Limited GPU (<8 GB) | **ESM2-8M / 35M / 150M**, **ProSST**, **VenusPLM**, **PETA** |
| Long-context / generation | **ProtT5-XL** |
| Structure-aware (PDB available) | **ProSST-2048**, **ProtSSN**, **SaProt**, **VenusREM** |
| Antibody sequences | **IgBert**, **IgT5** |
| Largest available, multi-GPU | **ESM2-15B**, **ProtT5-XXL** |

When picking:

- **Data volume:** with limited training data, smaller models often generalize better (less overfitting risk).
- **Sequence length:** for very long proteins, prefer models that natively support long context.
- **Resources:** smaller PLM + a PEFT method (e.g. `plm-lora`) is usually the best resource-quality trade-off.
- **Task type:** structure-aware models help on structure-dependent tasks (binding, stability); pure-sequence models are fine for solubility / localization.

When in doubt, train 2-3 candidates and pick the one that wins on the validation set.

### Q20: How do I read the training-loss curve in the Training page?

The Training page streams **Train Loss**, **Val Loss**, and **Val Metrics** charts in real time.

| Pattern | Likely meaning | What to try |
| :--- | :--- | :--- |
| Both losses go down and converge cleanly | Healthy run — let it finish | — |
| Train loss ↓ / Val loss ↑ | Overfitting | Higher dropout, weight decay, lower `Num Epochs`, smaller `Patience`, smaller model |
| Both losses plateau high | Underfitting | Higher learning rate, larger model, more epochs |
| Curve is wildly noisy | Learning rate too high | Drop LR by 5–10×, set `Max Grad Norm` to 1.0–5.0 |
| Val < Train | Often normal (dropout / data split effect); occasionally indicates split contamination | Check that train and val are truly disjoint |
| Sudden spike then NaN | Gradient explosion | Set `Max Grad Norm`, lower LR, check for extreme labels |

If val metric stops improving before the epoch cap, the **Patience** early-stop will halt the run automatically.

### Q21: How do I save and share my trained model?

Models are saved to `Save Directory / Output Model Name` (default `ckpt/demo/best_model.pt`). The folder contains:

| File | Purpose |
| :--- | :--- |
| `*.pt` | Model weights (the file you trained) |
| `config.json` / `adapter_config.json` | Run configuration: PLM, method, pooling, problem type, num labels, LoRA params — **Custom Model → Evaluation / Predict reads this back** |
| tokenizer files | Inherited from the base PLM |

**To share:**

1. **Hugging Face Hub** — easiest. Create a model repository, upload the folder, fill in a model card describing the training data, architecture, metrics, and a usage example.
2. **Local export** — `tar -czf my_model.tar.gz ckpt/demo/`. Recipient should also receive a note on which PLM was the base + the training method, so they can plug into **Custom Model → Predict** the same way.
3. **Document everything** — training data source, hyperparameters, validation / test metrics, intended use, and known limitations.

## Interface and Operation Issues

### Q22: The WebUI is slow or the page crashes.

1. **Browser:** Chrome / Edge tend to have the best compatibility with the React + Molstar viewer. Clear cache, disable heavy extensions.
2. **Resources:** make sure your machine has free RAM. Close other GPU-heavy apps. If on a remote server, check `top` / `nvidia-smi`.
3. **Network:** for remote deployments, an unstable SSH tunnel or reverse proxy can cause API timeouts. Test with `curl http://<host>:7861/health`.
4. **Restart:** kill `python src/webui_v2.py` and start it again. In Docker: `docker compose --profile gpu restart`.
5. **Build fresh frontend:** if some pages render but others are stuck on a spinner, rebuild the React bundle (`cd frontend && npm run build`) — older artifacts can mismatch a newer backend.

### Q23: Training process becomes unresponsive midway.

Most common causes:

1. **OOM kill:** the Linux OOM killer terminated the Python process. Check `dmesg | tail -30` for `Killed process`. Fix: lower batch size, use a smaller PLM, or use `plm-qlora`.
2. **GPU OOM that didn't crash but stalled:** rare but possible with non-blocking CUDA errors. `nvidia-smi` will show 0% util and unfree memory. Restart the process.
3. **Browser disconnect:** the UI shows "stopped" but the backend may still be running. The training process keeps going regardless — check the latest checkpoint in `ckpt/`.
4. **Network / SSH tunnel drop:** if you launched via SSH without `tmux` / `nohup` / `screen`, the process was killed when the shell died. Always run inside `tmux` for long jobs.
5. **API rate limits** (when training uses external services like W&B): the run can hang waiting on the API. Disable W&B with the toggle or check `wandb` status.

For all of the above, the best safeguard is **Continue Training** mode — restart from the last checkpoint with one click.

## Agent / Chat

### Q24: The Agent stops mid-plan.

- Check the **Execution Status** column on the right. A red status usually points at the failing tool.
- The most common causes: missing API key (set it in **Settings** or `.env`), provider rate limit, or a tool that ran out of memory.
- You can re-run the failing step: use **Modify & Re-execute** at the iteration checkpoint, or edit the plan and continue.

### Q25: Agent ran out of quota in online mode.

Online deployments enforce a per-user daily chat quota (visible in the quota pill near the input). When exhausted:

- Wait for the daily reset, or
- Use **Local mode** with your own API keys — no quota.

### Q26: My custom OpenAI-style model isn't showing up.

- Custom models are **local-mode only** and stored in your browser's `localStorage` (`vf2_custom_openai_style_models`).
- If you cleared site data, re-add them via the chat-page model picker.
- In Online mode they're filtered out by design.

## WebUI / Deployment

### Q27: What's the difference between WebUI v1 and v2?

- **v1** (`python src/webui.py`): legacy Gradio interface on port 7860.
- **v2** (`python src/webui_v2.py`): current FastAPI + React interface on port 7861 — this is what all the manuals describe.

v2 is the default for new deployments. v1 is kept around for users who depended on the old Gradio MCP integration.

### Q28: I started v2 but pages are blank.

You need to build the React frontend first:

```bash
cd frontend && npm install && npm run build && cd ..
python src/webui_v2.py --host 0.0.0.0 --port 7861
```

Without `frontend/dist/`, v2 has no UI to serve.

### Q29: WebUI v2 starts but pages can't reach the API.

Check the bound host and port:

- `--host 0.0.0.0` to accept external connections (default `0.0.0.0`).
- The default port is `7861`. If something else uses it, override with `--port` or `VENUS_PORT` (Docker).
- For external access through a reverse proxy, set `WEBUI_V2_CORS_ORIGINS` in `.env` to include your proxy URL.

### Q30: How do I reset the environment after a bad install?

```bash
# conda path
conda deactivate
conda env remove -n venus
conda create -n venus python=3.12
# then re-run the install steps from Q1

# uv path
rm -rf .venv
python install.py --type cu128
```

For Docker: `docker compose --profile gpu down -v && docker compose --profile gpu up -d --build`.
