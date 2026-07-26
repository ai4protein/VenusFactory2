# Agent · CLI · API · training methods

Back to [Wiki home](./Home.md) · [中文](./Usage_CN.md)

## Agent-0.1

Natural-language orchestration (LangGraph + LangChain). Example:

```
You: "Design thermostable mutations for PDB:1ABC"
→ Download structure → ESM-2 scan → Stability score → Ranked report
```

| Category | Capabilities |
|:---------|:-------------|
| Analysis | Mutation · function/stability · structure |
| Data | Multi-DB search · conversion · batch |
| Planning | Multi-step automation · tool routing |
| Research | Literature · family analysis · reports |

Needs an LLM API key (UI Settings or `.env`; see `.env.example`).  
Manual: [AgentManual_EN.md](../manual/AgentManual_EN.md)

## CLI (examples)

Most scripts live under `script/tools/`:

```bash
bash script/tools/train/train_plm_lora.sh \
  --model facebook/esm2_t33_650M_UR50D \
  --dataset DeepSol --batch_size 32

bash script/tools/evaluate/eval.sh \
  --model_path ckpt/DeepSol/best_model \
  --test_dataset DeepSol
```

Database helpers: `script/tools/database/`.

## REST API

```bash
python src/api_server.py   # → http://localhost:5000/docs
```

```bash
curl -X POST http://localhost:5000/api/mutation/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTAYIA...", "mutations": ["A23V", "K45R"]}'
```

Treat OpenAPI at `/docs` as source of truth (routes may evolve).

## Python (illustrative)

```python
from src.tools.mutation import predict_mutation_effects

results = predict_mutation_effects(
    sequence="MKTAYIAKQR...",
    mutations=["A5V", "K9R"],
    model="esm2",
)
```

Confirm import paths against the current codebase.

## PEFT methods

| Method | Memory | Speed | Quality | Best for |
|:-------|:------:|:-----:|:-------:|:---------|
| LoRA | Low | Fast | Good | General |
| QLoRA | Very low | Slow | Good | Tight VRAM |
| DoRA | Low | Medium | Better | Improved LoRA |
| AdaLoRA | Low | Medium | Better | Adaptive rank |
| SES-Adapter | Medium | Medium | Better | Selective tuning |
| IA3 | Very low | Fast | Good | Lightweight |
| Freeze | Low | Fast | Good | Simple tuning |

See [TrainingManual_EN.md](../manual/TrainingManual_EN.md).
