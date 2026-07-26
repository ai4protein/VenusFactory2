# 权重 / Checkpoints

回到 [Wiki 首页](./Home.md) · [English](./Checkpoints.md) · 上游说明 [ckpt/README.md](../../ckpt/README.md)

权重托管在 [AI4Protein/VenusFactory2-ckpts](https://huggingface.co/AI4Protein/VenusFactory2-ckpts)，**不进 git**。本地目录默认 `ckpt/`。

## 一键安装默认

`scripts/setup_quickstart.py` 推荐方案会下载 **`predict-core`**（约 163 MB）。

## Preset

| Preset | 约大小 | 内容 |
|:-------|:-------|:-----|
| `demo` | ~0.4 MB | 冒烟用溶解度 demo |
| `predict-core` | ~163 MB | **推荐**：demo + 各任务 `ankh-large` |
| `proteinmpnn` | ~64 MB | ProteinMPNN |
| `predict-all` | ~405 MB | 全部微调任务目录 |
| `all` | ~469 MB | hub 全量 |

```bash
python scripts/download_ckpts.py --list-presets
python scripts/download_ckpts.py --preset predict-core
python scripts/download_ckpts.py --preset proteinmpnn
python scripts/download_ckpts.py --include 'DeepSol/**'
```

## 常见任务目录（示例）

微调适配器按任务分子目录存放（每个任务下可有多个 backbone，如 `ankh-large` / `esm2_t33_650M_UR50D` 等）：

- DeepSol / DeepSoluE / ProtSolM — 溶解度
- DeepLocBinary / DeepLocMulti — 定位
- Thermostability / DeepET_Topt — 热稳定 / 最适温度
- MetalIonBinding / SortingSignal / DLKcat / EpHod
- VenusVaccine_* — 疫苗相关二分类
- VenusX_Res_* — 残基级任务（Motif / Act / Bind / Evo 等）
- ProteinMPNN — 序列设计
- `demo/` — 演示权重

完整文件列表、体积与 sha256 见 `ckpt/manifest.json`（下载后或 HF 仓库内）。

## 环境变量

| 变量 | 默认 | 含义 |
|:-----|:-----|:-----|
| `VENUS_CKPT_REPO_ID` | `AI4Protein/VenusFactory2-ckpts` | HF 仓库 |
| `VENUS_CKPT_DIR` | `ckpt` | 本地根目录 |
| `VENUS_CKPT_REVISION` | `main` | 分支/标签 |
| `VENUS_CKPT_AUTO_DOWNLOAD` | `1` | 缺失时自动拉取 |
| `VENUS_DOWNLOAD_QUIET` | `0` | 关闭下载提示 |
| `HF_ENDPOINT` | （空） | 镜像，如 `https://hf-mirror.com` |

## Gradio frpc

仅 `--share` 需要：`python scripts/download_frpc.py --to-gradio-cache`  
详见 [Installation_CN.md](./Installation_CN.md)。
