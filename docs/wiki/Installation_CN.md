# VenusFactory2 安装详解

[← 文档总览](../README.md) · [Wiki 首页](./Home.md) · [English](./Installation.md)

本文是**权威**安装页（根 README 只保留最短命令）。日常上手：

```bash
git clone --recurse-submodules https://github.com/AI4Protein/VenusFactory2.git && cd VenusFactory2
# 若已普通 clone：git submodule update --init --recursive
python scripts/setup_quickstart.py
```

Agent 技能可选加载 git submodule `third_party/scientific-agent-skills`（[scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)）。未初始化时仍可使用 VenusFactory2 自有技能。

终端会给出选项，带 **[recommended]** 的为推荐项；直接回车即可按推荐方案安装（依赖 + 前端 + `predict-core` 权重）。

非交互（CI / 脚本）：

```bash
python scripts/setup_quickstart.py -y
```

若检测到旧环境（`./.venv`、已构建前端等），交互菜单会提示：

- **Reuse** 现有 `.venv`（推荐，若 torch 可用）
- **Remove `.venv` and reinstall**（推荐用于损坏/过旧环境）
- **Force reinstall packages**（保留目录，强制重装依赖）
- **Full wipe**（删除 `.venv` + `frontend/dist` + `node_modules` 后重装）

非交互示例：

```bash
python scripts/setup_quickstart.py -y --clean-venv                 # 删掉旧 venv 再装
python scripts/setup_quickstart.py -y --clean-venv --clean-frontend
```

---

## 前置条件

| 组件 | 要求 |
|:-----|:-----|
| Python | **≥3.12**（`uv venv` 默认创建 3.12；`--env system` 也需 ≥3.12） |
| Node.js | 25.x（构建 WebUI v2 前端） |
| 磁盘 | 推荐方案约需数百 MB～数 GB（权重可选） |
| GPU | 可选；无 GPU 会自动走 CPU 轮子 |

Linux 上若需 PDF 导出（`pycairo`），Debian/Ubuntu 可先装：

```bash
sudo apt-get install -y libcairo2-dev libxml2-dev pkg-config
# 或 conda: conda install -c conda-forge pycairo
```

---

## 一键安装会做什么

`scripts/setup_quickstart.py` 推荐方案默认包括：

1. **扫描平台**：有 NVIDIA GPU → `cu128`，否则 / macOS → `cpu`
2. **安装依赖**：通过 `install.py` + `uv` 装入 `./.venv`（也可选当前 conda/system 解释器）
3. **构建前端**：`frontend/` 下 `npm install && npm run build`
4. **下载权重**：默认 `predict-core`（约 163 MB，见下表）
5. **环境检查**：`python scripts/check_env.py`

可选：Gradio Share 用的 `frpc`（本地 UI **不需要**）。

### 权重 preset

托管仓库：[AI4Protein/VenusFactory2-ckpts](https://huggingface.co/AI4Protein/VenusFactory2-ckpts)（**不进 git**）。

| Preset | 约大小 | 说明 |
|:-------|:-------|:-----|
| `demo` | ~0.4 MB | 冒烟 |
| `predict-core` | ~163 MB | **推荐**：demo + 各任务 ankh-large |
| `proteinmpnn` | ~64 MB | 序列设计 |
| `predict-all` | ~405 MB | 全部微调适配器 |
| `all` | ~469 MB | hub 全量 |

```bash
python scripts/download_ckpts.py --list-presets
python scripts/download_ckpts.py --preset proteinmpnn
```

默认开启按需下载：`VENUS_CKPT_AUTO_DOWNLOAD=1`。  
HF 访问慢时可：`export HF_ENDPOINT=https://hf-mirror.com`  
关闭进度提示：`VENUS_DOWNLOAD_QUIET=1`

```bash
python scripts/download_frpc.py --to-gradio-cache   # 仅 Gradio --share
```

---

## 命令行参数（非交互）

```bash
python scripts/setup_quickstart.py -y                          # 推荐全量一键
python scripts/setup_quickstart.py --install-deps --torch-type cpu
python scripts/setup_quickstart.py --install-deps --env system # 装到当前解释器
python scripts/setup_quickstart.py --preset demo
python scripts/setup_quickstart.py --skip-ckpts
python scripts/setup_quickstart.py --with-frpc
python scripts/setup_quickstart.py --dry-run -y                # 只预览
```

底层依赖安装也可单独调用（默认 `--type auto`）：

```bash
python install.py                 # → ./.venv
python install.py --env system --type cu128
python install.py --type cpu --skip-frpc
```

---

## 手动安装（不使用一键脚本）

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

> 一键安装器当前自动配置只内置 `cu128` / `cpu`。CUDA 11.8 请用本节手动步骤。

### 仅 CPU

```bash
conda create -n venus python=3.12 && conda activate venus
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install -r requirements.txt
```

---

## 验证与启动

```bash
python scripts/check_env.py
source .venv/bin/activate   # 若使用默认 venv 安装
python src/webui_v2.py --host 0.0.0.0 --port 7861
# → http://localhost:7861
```

其他入口：

```bash
python src/webui.py --mode all                              # Gradio v1
python src/webui_v2.py --host 0.0.0.0 --port 7861 --online
python src/api_server.py                                    # → :5000/docs
```

配置模板：`.env.example` → 复制为 `.env` 后按需修改。  
Agent 若跳过默认 kimi-code：`export KIMI_EXTERNAL=1`。

---

## 相关文档

- 英文版：[Installation.md](./Installation.md)
- 权重 / preset 详解：[Checkpoints_CN.md](./Checkpoints_CN.md) · [ckpt/README.md](../../ckpt/README.md)
- Wiki 首页：[Home.md](./Home.md)
- 功能手册：[docs/manual/](../manual/)
