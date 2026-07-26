# Supported models

Back to [Wiki home](./Home.md) · [中文](./Models_CN.md)

VenusFactory2 integrates **40+** protein language models (PLMs) for zero-shot mutation, fine-tuning, and inference.

## Venus series (Liang Lab)

- ProSST-20 / 128 / 512 / 1024 / 2048 / 4096 (~110M)
- ProPrime-690M
- VenusPLM-300M
- PETA-base / bpe / unigram (~80M)

## ESM series (Meta)

- ESM2: 8M, 35M, 150M, 650M, 3B, 15B
- ESM-1v: 5 models (~650M each)

## ProtBert / ProtT5 / Ankh / antibody

- ProtBert-Uniref100 / BFD (~420M)
- IgBert (~420M)
- ProtT5-XL / XXL (~3B–11B)
- Ankh-base / large (~450M–1.2B)
- IgT5 (antibody)

## GPU selection guide

| VRAM | Suggested models |
|:-----|:-----------------|
| <8GB | ESM2-8M/35M, ProSST |
| 8–16GB | ESM2-150M/650M, ProtBert |
| 24GB+ | ESM2-3B, ProtT5-XL |
| Multi-GPU | ESM2-15B, ProtT5-XXL |

## By task

| Task | Common choices |
|:-----|:---------------|
| Classification | ESM2, ProtBert |
| Structure-related | Ankh |
| Generation | ProtT5 |
| Antibody | IgBert / IgT5 |
| Lightweight | ProSST, PETA |

## Fine-tuned adapters (ckpt)

Task-specific adapters (solubility, localization, …) live under `ckpt/`. See [Checkpoints.md](./Checkpoints.md).
