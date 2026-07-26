# Checkpoints

Back to [Wiki home](./Home.md) · [中文](./Checkpoints_CN.md) · upstream notes [ckpt/README.md](../../ckpt/README.md)

Weights are hosted at [AI4Protein/VenusFactory2-ckpts](https://huggingface.co/AI4Protein/VenusFactory2-ckpts) and are **not** in git. Local cache defaults to `ckpt/`.

## One-click default

`scripts/setup_quickstart.py` downloads **`predict-core`** (~163 MB) in the recommended plan.

## Presets

| Preset | Approx. size | Contents |
|:-------|:-------------|:---------|
| `demo` | ~0.4 MB | Tiny solubility demo |
| `predict-core` | ~163 MB | **Recommended**: demo + per-task `ankh-large` |
| `proteinmpnn` | ~64 MB | ProteinMPNN |
| `predict-all` | ~405 MB | All finetuned task folders |
| `all` | ~469 MB | Full hub snapshot |

```bash
python scripts/download_ckpts.py --list-presets
python scripts/download_ckpts.py --preset predict-core
python scripts/download_ckpts.py --preset proteinmpnn
python scripts/download_ckpts.py --include 'DeepSol/**'
```

## Common task folders (examples)

Adapters are stored per task (with backbones such as `ankh-large`, `esm2_t33_650M_UR50D`, …):

- DeepSol / DeepSoluE / ProtSolM — solubility
- DeepLocBinary / DeepLocMulti — localization
- Thermostability / DeepET_Topt — thermostability / Topt
- MetalIonBinding / SortingSignal / DLKcat / EpHod
- VenusVaccine_* — vaccine binary tasks
- VenusX_Res_* — residue-level tasks (Motif / Act / Bind / Evo, …)
- ProteinMPNN — sequence design
- `demo/` — demo weights

Full file list, sizes, and sha256: `ckpt/manifest.json` (after download or on HF).

## Environment variables

| Variable | Default | Meaning |
|:---------|:--------|:--------|
| `VENUS_CKPT_REPO_ID` | `AI4Protein/VenusFactory2-ckpts` | HF repo |
| `VENUS_CKPT_DIR` | `ckpt` | Local root |
| `VENUS_CKPT_REVISION` | `main` | Branch / tag |
| `VENUS_CKPT_AUTO_DOWNLOAD` | `1` | Fetch missing weights |
| `VENUS_DOWNLOAD_QUIET` | `0` | Suppress notices |
| `HF_ENDPOINT` | (empty) | Mirror, e.g. `https://hf-mirror.com` |

## Gradio frpc

Only for `--share`: `python scripts/download_frpc.py --to-gradio-cache`  
See [Installation.md](./Installation.md).
