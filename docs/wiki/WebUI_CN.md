# WebUI 模块地图（v2）

[← Wiki 首页](./Home.md) · [English](./WebUI.md)

当前主界面是 **FastAPI + React**（`python src/webui_v2.py`，默认 `:7861`）。  
旧版 Gradio（`python src/webui.py`，`:7860`）仅作兼容，手册以 v2 为准。

## 启动

一键安装后通常已构建前端：

```bash
source .venv/bin/activate
python src/webui_v2.py --host 0.0.0.0 --port 7861
```

仅在改过 `frontend/` 或缺少 `frontend/dist` 时：

```bash
cd frontend && npm install && npm run build && cd ..
```

Online：`python src/webui_v2.py --host 0.0.0.0 --port 7861 --online`

## 模块 ↔ 手册

| 侧栏 / 模块 | 路由前缀（示意） | 详细手册 |
|:------------|:-----------------|:---------|
| **Agent** | `/agent`, chat | [AgentManual_CN.md](../manual/AgentManual_CN.md) |
| **Quick Tools** | `/quick-tools/*` | [QuickTools_CN.md](../manual/QuickTools_CN.md) |
| **Advanced Tools** | `/advanced-tools/*` | [AdvancedToolsManual_CN.md](../manual/AdvancedToolsManual_CN.md) |
| **Custom Model** · Training | `/custom-model/training` | [TrainingManual_ZH.md](../manual/TrainingManual_ZH.md) |
| **Custom Model** · Evaluation | `/custom-model/evaluation` | [EvaluationManual_ZH.md](../manual/EvaluationManual_ZH.md) |
| **Custom Model** · Predict | `/custom-model/predict` | [PredictionManual_ZH.md](../manual/PredictionManual_ZH.md) |
| **Download** | `/download/*` | [DownloadManual_ZH.md](../manual/DownloadManual_ZH.md) |
| **Manual** | `/manual/*` | 内嵌上述手册 |
| **Workspace / Settings / Leaderboards / Report** | 对应页面 | Report → [ReportManual_CN.md](../manual/ReportManual_CN.md) |

### Quick Tools（6）

Directed Evolution · Sequence Design · Protein Discovery · Protein Function · Functional Residue · Physicochemical Property  

### Advanced Tools

与 Quick Tools 同类任务，提供更多模型 / 参数控制。

## 配置

- 模板：`.env.example` → `.env`
- Agent / LLM：优先界面 **Settings**；也可写环境变量（见 `.env.example`）
- Online：`WEBUI_V2_MODE=online`、`WEBUI_V2_SESSION_TOKEN_SECRET`、`WEBUI_V2_*_LIMIT`
- 跳过默认 kimi-code：`export KIMI_EXTERNAL=1`

## 权重

功能预测等任务用到的适配器见 [Checkpoints_CN.md](./Checkpoints_CN.md)。本地缺失时默认按需从 HF 拉取（`VENUS_CKPT_AUTO_DOWNLOAD=1`）。
