---
license: mit
tags:
  - biology
  - protein
  - venusfactory
  - checkpoints
library_name: venus-factory
---

# VenusFactory2 Checkpoints

Pretrained / finetuned weights for [VenusFactory2](https://github.com/AI4Protein/VenusFactory2).

These files are **not** stored in the GitHub repository. VenusFactory2 downloads them into a local `ckpt/` directory.

## Install / download

```bash
# From a VenusFactory2 checkout
python scripts/download_ckpts.py --preset predict-core   # demo + ankh-large adapters
python scripts/download_ckpts.py --preset proteinmpnn    # ProteinMPNN
python scripts/download_ckpts.py --preset all            # everything
python scripts/download_ckpts.py --include 'DeepSol/**'  # one task
```

Or with the Hugging Face CLI:

```bash
huggingface-cli download AI4Protein/VenusFactory2-ckpts \
  --local-dir ckpt \
  --include "DeepSol/**" "demo/**"
```

## Auto-download

VenusFactory2 enables on-demand fetch by default (`VENUS_CKPT_AUTO_DOWNLOAD=1`).
The first time a tool needs a missing adapter, it pulls from this repo into `ckpt/`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `VENUS_CKPT_REPO_ID` | `AI4Protein/VenusFactory2-ckpts` | This repo |
| `VENUS_CKPT_DIR` | `ckpt` | Local cache root |
| `VENUS_CKPT_REVISION` | `main` | Branch / tag / commit |
| `VENUS_CKPT_AUTO_DOWNLOAD` | `1` | Fetch missing weights automatically |

## Layout

```text
demo/
DeepSol/<backbone>/*.pt|*.json
ProteinMPNN/vanilla_model_weights/*.pt
...
manifest.json
assets/frpc/frpc_linux_amd64_v0.3   # Gradio share helper (not a model weight)
```

See `manifest.json` for the full file list, sizes, and sha256 checksums.

## Gradio ``frpc``

```bash
python scripts/download_frpc.py --to-gradio-cache
```

This binary is also ignored by git (same idea as checkpoints).

## Presets

- `demo` — demo solubility adapter
- `predict-core` — demo + all task `ankh-large` adapters
- `predict-all` — demo + all finetuned task folders
- `proteinmpnn` — ProteinMPNN weights
- `all` — entire repository
