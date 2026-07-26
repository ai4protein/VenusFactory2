# 支持的模型

回到 [Wiki 首页](./Home.md) · [English](./Models.md)

仓库集成 **40+** 蛋白质语言模型（PLM），可用于零样本突变、微调训练与推理。

## Venus 系列（Liang Lab）

- ProSST-20 / 128 / 512 / 1024 / 2048 / 4096（约 110M）
- ProPrime-690M
- VenusPLM-300M
- PETA-base / bpe / unigram（约 80M）

## ESM 系列（Meta）

- ESM2：8M, 35M, 150M, 650M, 3B, 15B
- ESM-1v：5 个模型（各约 650M）

## ProtBert / ProtT5 / Ankh / 抗体

- ProtBert-Uniref100 / BFD（约 420M）
- IgBert（约 420M）
- ProtT5-XL / XXL（约 3B–11B）
- Ankh-base / large（约 450M–1.2B）
- IgT5（抗体相关）

## GPU 选型建议

| 显存 | 建议 |
|:-----|:-----|
| <8GB | ESM2-8M/35M, ProSST |
| 8–16GB | ESM2-150M/650M, ProtBert |
| 24GB+ | ESM2-3B, ProtT5-XL |
| 多卡 | ESM2-15B, ProtT5-XXL |

## 按任务

| 任务 | 常用模型 |
|:-----|:---------|
| 分类 | ESM2, ProtBert |
| 结构相关 | Ankh |
| 生成 | ProtT5 |
| 抗体 | IgBert / IgT5 |
| 轻量 | ProSST, PETA |

## 微调权重（ckpt）

平台还为功能/定位/溶解度等任务提供 **微调适配器**（按任务目录存放），详见 [Checkpoints_CN.md](./Checkpoints_CN.md)。
