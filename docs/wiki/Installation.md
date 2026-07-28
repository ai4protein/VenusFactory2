# VenusFactory2 Installation Guide

[← Documentation map](../README.md) · [Wiki home](./Home.md) · [中文](./Installation_CN.md)

This is the **canonical** install page (the root README only keeps quick commands). Day-to-day setup:

```bash
git clone --recurse-submodules https://github.com/AI4Protein/VenusFactory2.git && cd VenusFactory2
# Or after a plain clone: git submodule update --init --recursive
python scripts/setup_quickstart.py
```

Agent skills optionally load the [scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) git submodule under `third_party/scientific-agent-skills`. Without it, VenusFactory2 skills still work.

The CLI presents choices; options marked **[recommended]** are defaults. Press Enter to accept the recommended all-in-one plan (deps + frontend + `predict-core` weights).

Non-interactive (CI / scripts):

```bash
python scripts/setup_quickstart.py -y
```

If an older install is detected (`./.venv`, built frontend, …), the interactive menu offers:

- **Reuse** the existing `.venv` (recommended when torch works)
- **Remove `.venv` and reinstall** (recommended for broken/stale envs)
- **Force reinstall packages** (keep the directory, reinstall wheels)
- **Full wipe** (delete `.venv` + `frontend/dist` + `node_modules`, then reinstall)

Non-interactive examples:

```bash
python scripts/setup_quickstart.py -y --clean-venv
python scripts/setup_quickstart.py -y --clean-venv --clean-frontend
```

---

## Prerequisites

| Component | Requirement |
|:----------|:------------|
| Python | **≥3.12** (default `uv venv` creates 3.12; `--env system` also needs ≥3.12) |
| Node.js | 25.x (WebUI v2 frontend build) |
| Disk | Hundreds of MB to a few GB depending on weights |
| GPU | Optional; falls back to CPU wheels |

Linux PDF export (`pycairo`) may need system headers on Debian/Ubuntu:

```bash
sudo apt-get install -y libcairo2-dev libxml2-dev pkg-config
# or: conda install -c conda-forge pycairo
```

---

## What the one-click installer does

Recommended plan in `scripts/setup_quickstart.py`:

1. **Detect platform**: NVIDIA GPU → `cu128`, else / macOS → `cpu`
2. **Install deps**: `install.py` + `uv` into `./.venv` (or current conda/system interpreter)
3. **Build frontend**: `npm install && npm run build` under `frontend/`
4. **Download weights**: default `predict-core` (~163 MB)
5. **Verify**: `python scripts/check_env.py`

Optional: Gradio Share `frpc` (not required for local UI).

### Weight presets

Hosted at [AI4Protein/VenusFactory2-ckpts](https://huggingface.co/AI4Protein/VenusFactory2-ckpts) (**not** in git).

| Preset | Approx. size | Notes |
|:-------|:-------------|:------|
| `demo` | ~0.4 MB | Smoke test |
| `predict-core` | ~163 MB | **Recommended**: demo + ankh-large adapters |
| `proteinmpnn` | ~64 MB | Sequence design |
| `predict-all` | ~405 MB | All finetuned adapters |
| `all` | ~469 MB | Full hub snapshot |

```bash
python scripts/download_ckpts.py --list-presets
python scripts/download_ckpts.py --preset proteinmpnn
```

On-demand download is on by default (`VENUS_CKPT_AUTO_DOWNLOAD=1`).  
Slow HF access: `export HF_ENDPOINT=https://hf-mirror.com`  
Quiet notices: `VENUS_DOWNLOAD_QUIET=1`

```bash
python scripts/download_frpc.py --to-gradio-cache   # Gradio --share only
```

---

## CLI flags (non-interactive)

```bash
python scripts/setup_quickstart.py -y                          # recommended all-in-one
python scripts/setup_quickstart.py --install-deps --torch-type cpu
python scripts/setup_quickstart.py --install-deps --env system # current interpreter
python scripts/setup_quickstart.py --preset demo
python scripts/setup_quickstart.py --skip-ckpts
python scripts/setup_quickstart.py --with-frpc
python scripts/setup_quickstart.py --dry-run -y                # preview only
```

Lower-level dependency installer (default `--type auto`):

```bash
python install.py
python install.py --env system --type cu128
python install.py --type cpu --skip-frpc
```

---

## Manual install (without the one-click script)

### macOS (Apple Silicon)

```bash
conda create -n venus python=3.12 && conda activate venus
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install torch_scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install -r requirements_for_macOS.txt
cd frontend && npm install && npm run build && cd ..
python scripts/download_ckpts.py --preset predict-core
```

### Linux / Windows — CUDA 12.8

```bash
conda create -n venus python=3.12 && conda activate venus
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install -r requirements.txt
```

### Linux / Windows — CUDA 11.8

```bash
conda create -n venus python=3.12 && conda activate venus
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install -r requirements.txt
```

> The auto installer currently ships only `cu128` / `cpu` profiles. Use this section for CUDA 11.8.

### CPU only

```bash
conda create -n venus python=3.12 && conda activate venus
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install -r requirements.txt
```

---

## Verify & launch

```bash
python scripts/check_env.py
source .venv/bin/activate   # if you used the default venv install
python src/webui_v2.py --host 0.0.0.0 --port 7861
# → http://localhost:7861
```

Other entry points:

```bash
python src/webui.py --mode all
python src/webui_v2.py --host 0.0.0.0 --port 7861 --online
python src/api_server.py
```

Config template: copy `.env.example` → `.env`.  
Skip default kimi-code engine: `export KIMI_EXTERNAL=1`.

---

## Related

- Chinese: [Installation_CN.md](./Installation_CN.md)
- Checkpoints / presets: [Checkpoints.md](./Checkpoints.md) · [ckpt/README.md](../../ckpt/README.md)
- Wiki home: [Home.md](./Home.md)
- Feature manuals: [docs/manual/](../manual/)
