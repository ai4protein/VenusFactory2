# WebUI module map (v2)

[← Wiki home](./Home.md) · [中文](./WebUI_CN.md)

The primary UI is **FastAPI + React** (`python src/webui_v2.py`, default `:7861`).  
Legacy Gradio (`python src/webui.py`, `:7860`) is compatibility-only; manuals describe v2.

## Launch

After one-click install the frontend is usually already built:

```bash
source .venv/bin/activate
python src/webui_v2.py --host 0.0.0.0 --port 7861
```

Rebuild only if you changed `frontend/` or `frontend/dist` is missing:

```bash
cd frontend && npm install && npm run build && cd ..
```

Online: `python src/webui_v2.py --host 0.0.0.0 --port 7861 --online`

## Modules ↔ manuals

| Module | Route prefix (approx.) | How-to |
|:-------|:-----------------------|:-------|
| **Agent** | `/agent`, chat | [AgentManual_EN.md](../manual/AgentManual_EN.md) |
| **Quick Tools** | `/quick-tools/*` | [QuickTools_EN.md](../manual/QuickTools_EN.md) |
| **Advanced Tools** | `/advanced-tools/*` | [AdvancedToolsManual_EN.md](../manual/AdvancedToolsManual_EN.md) |
| **Custom Model** · Training | `/custom-model/training` | [TrainingManual_EN.md](../manual/TrainingManual_EN.md) |
| **Custom Model** · Evaluation | `/custom-model/evaluation` | [EvaluationManual_EN.md](../manual/EvaluationManual_EN.md) |
| **Custom Model** · Predict | `/custom-model/predict` | [PredictionManual_EN.md](../manual/PredictionManual_EN.md) |
| **Download** | `/download/*` | [DownloadManual_EN.md](../manual/DownloadManual_EN.md) |
| **Manual** | `/manual/*` | In-app copies of the manuals |
| **Workspace / Settings / Leaderboards / Report** | matching pages | Report → [ReportManual_EN.md](../manual/ReportManual_EN.md) |

### Quick Tools (6)

Directed Evolution · Sequence Design · Protein Discovery · Protein Function · Functional Residue · Physicochemical Property  

### Advanced Tools

Same task families as Quick Tools with more model/parameter control.

### Chat — dual modes

Agent chat (`/agent`) uses a composer toggle (**Science Agent** / **Science Expert**; choice is stored in the browser).

| Mode | Engine | Notes |
|:-----|:-------|:------|
| **Science Agent** | kimi-code | Tool-first agent loop (MCP / bash); collapsible thinking blocks and inline tool cards. Backed by the kimi-code daemon (see Configuration). Online hides the model picker. |
| **Science Expert** | LangGraph (`graph`) | Multi-role pipeline **PI → CB → MLS → SC**; plan review in Plan Editor before run. Tuned for fewer pauses: execution-style tasks skip literature research; after plan confirm, steps default to auto-run; sub-reports default to auto-advance (plan confirm is still required). **Local**: pick the LLM in the model selector. **Online**: no model selector — fixed backend model (client cannot change it). |

Details: [AgentManual_EN.md](../manual/AgentManual_EN.md).

## Configuration

- Template: `.env.example` → `.env`
- Agent / LLM: prefer UI **Settings**; env vars documented in `.env.example`
- Online: `WEBUI_V2_MODE=online`, `WEBUI_V2_SESSION_TOKEN_SECRET`, `WEBUI_V2_*_LIMIT`
- Online without sudo (kimi-code): default `auto` uses `bwrap` sandbox; or set `KIMI_ONLINE_SPAWN_MODE=bwrap`
- Skip default kimi-code: `export KIMI_EXTERNAL=1`

## Weights

Adapters used by function/residue tools: [Checkpoints.md](./Checkpoints.md). Missing files auto-fetch when `VENUS_CKPT_AUTO_DOWNLOAD=1`.
