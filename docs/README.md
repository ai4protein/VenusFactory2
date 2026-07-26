# VenusFactory2 documentation map

Docs are split into **three layers**. Prefer the top layer first; go deeper only when needed.

```text
README(.md / _CN.md)     ← 30-second: what it is + one-click install + launch
       ↓
docs/wiki/               ← repo facts: install, models, ckpts, WebUI map, CLI/API
       ↓
docs/manual/             ← feature how-tos for each WebUI module (in-app Manual tab too)
```

| Layer | Path | Audience | Contains |
|:------|:-----|:---------|:---------|
| Landing | [`README.md`](../README.md) / [`README_CN.md`](../README_CN.md) | Everyone | Product one-liner, prerequisites, `setup_quickstart.py`, launch, wiki index |
| Wiki | [`docs/wiki/`](./wiki/) | Operators / power users | Install details, architecture, models/datasets/ckpts, WebUI map, Agent/CLI/API overview |
| Manuals | [`docs/manual/`](./manual/) | End users in the UI | Step-by-step for Quick Tools, Advanced, Training, Agent, Download, FAQ |

**Not user docs (ignore unless developing):** `src/agent/skills/**/SKILL.md` (agent tool prompts), HF card text in [`ckpt/README.md`](../ckpt/README.md) (mirrors the weight repo; VenusFactory users should start from [wiki/Checkpoints](./wiki/Checkpoints.md)).

## Wiki index

Start at **[wiki/Home.md](./wiki/Home.md)**.

| Page | Purpose |
|:-----|:--------|
| [Installation](./wiki/Installation.md) | One-click + manual CUDA/CPU, clean reinstall, troubleshooting |
| [WebUI](./wiki/WebUI.md) | Current FastAPI/React modules and how they map to manuals |
| [Overview](./wiki/Overview.md) | Architecture & capabilities |
| [Models](./wiki/Models.md) / [Datasets](./wiki/Datasets.md) / [Checkpoints](./wiki/Checkpoints.md) | Catalogs |
| [Usage](./wiki/Usage.md) | Agent keys, CLI under `script/tools/`, API, PEFT cheat-sheet |

Chinese twins use `_CN` suffix (some older manuals still use `_ZH` — same language).

## Current product surface (source of truth)

| Entry | Command / path |
|:------|:---------------|
| Primary UI | `python src/webui_v2.py --host 0.0.0.0 --port 7861` |
| Legacy Gradio | `python src/webui.py --mode all` → `:7860` |
| REST | `python src/api_server.py` → `:5000/docs` |
| Install | `python scripts/setup_quickstart.py` |
| Online demo | https://venusfactory.bio/ |
| CLI scripts | `script/tools/{train,evaluate,database,...}/` |

Frontend modules (v2): Agent · Quick Tools · Advanced Tools · Custom Model · Download · Manual · Workspace · Settings · Leaderboards · Report.
