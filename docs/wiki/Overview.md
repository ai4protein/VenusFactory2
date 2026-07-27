# Overview & architecture

Back to [Wiki home](./Home.md) · [中文](./Overview_CN.md)

## Capabilities

| Task | Approach | Typical time |
|:-----|:---------|:-------------|
| Mutation effects | ESM-2, ProSST, ProtSSN (zero-shot) | <1 min |
| Protein function | 30+ fine-tuned adapters | <30 sec |
| Custom training | 7 PEFT methods (LoRA, QLoRA, …) | 10–60 min |
| Data download | AlphaFold, UniProt, RCSB, KEGG, … | Real-time |
| Literature / search | AI-assisted retrieval | <2 min |

## Why VenusFactory2?

| Agent-first | Three interfaces | Zero to results |
|:------------|:-----------------|:----------------|
| Natural language → multi-step flows; Web chat **Science Agent** (kimi-code) or **Science Expert** (LangGraph) | Web / REST / CLI | Upload → predict |
| 40+ models + 11 databases | Same power, different entry points | Or fine-tune in minutes |

## Architecture

```
Interfaces: Web UI | REST API | CLI
        ↓
   Agent layer (LangGraph + LangChain)
        ↓
   Apps: Train | Eval | Predict | Tools
        ↓
   Core tools (mutation, DB, search, design, …)
        ↓
   Resources: 40+ models | 30+ datasets | 11+ databases
```

## Tool categories

| Category | Notes | Agent | CLI |
|:---------|:------|:-----:|:---:|
| Mutation | ESM-1v/2, ProSST, ProtSSN, MIF-ST | ✅ | ✅ |
| Prediction | 30+ fine-tuned models | ✅ | ✅ |
| Database | 11 integrations | ✅ | ✅ |
| Search | PubMed, FDA, patents, … | ✅ | ✅ |
| Training | LoRA / QLoRA / DoRA… | ✅ | ✅ |
| File | Format conversion | ✅ | ✅ |
| Denovo | Protein design | ✅ | ✅ |
| Discovery | Discovery workflows | ✅ | ✅ |
| Visualize | 3D viewer | ✅ | ✅ |

## Resources (pointers)

- **Models**: [Models.md](./Models.md)
- **Datasets**: [Datasets.md](./Datasets.md)
- **Checkpoints**: [Checkpoints.md](./Checkpoints.md)
- **Databases**: AlphaFold · RCSB · UniProt · NCBI · KEGG · STRING · BRENDA · ChEMBL · HPA · FDA · Foldseek

## Recent news (excerpt)

- [2026-04-01] [venusfactory.bio](https://venusfactory.bio/)
- [2026-03-27] Technical report [arXiv:2603.27303](https://arxiv.org/abs/2603.27303)
- [2026-01-23] [VenusX (ICLR2026)](https://openreview.net/forum?id=zcmL592XRG)
- [2025-04-19] [VenusREM](https://github.com/ai4protein/VenusREM) #1 ProteinGym / VenusMutHub
