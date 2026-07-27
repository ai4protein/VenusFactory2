# 功能概览与架构

回到 [Wiki 首页](./Home.md) · [English](./Overview.md)

## 核心能力

| 任务 | 方案 | 量级耗时 |
|:-----|:-----|:---------|
| 突变效应 | ESM-2, ProSST, ProtSSN（零样本） | <1 分钟 |
| 蛋白功能 | 30+ 微调适配器 | <30 秒 |
| 定制训练 | 7 种 PEFT（LoRA, QLoRA 等） | 10–60 分钟 |
| 数据下载 | AlphaFold, UniProt, RCSB, KEGG 等 | 实时 |
| 文献 / 检索 | AI 辅助搜索与分析 | <2 分钟 |

## 为什么用 VenusFactory2？

| Agent 优先 | 三种接口 | 从零到结果 |
|:-----------|:---------|:-----------|
| 自然语言 → 多步自动化；Web 对话可选 **Science Agent**（kimi-code）或 **Science Expert**（LangGraph） | Web / REST / CLI | 上传即可预测 |
| 40+ 模型 + 11 数据库 | 同一能力，不同入口 | 也可分钟级微调 |

## 架构

```
接口: Web UI | REST API | CLI
        ↓
   Agent 层 (LangGraph + LangChain)
        ↓
   应用: 训练 | 评估 | 预测 | 工具
        ↓
   核心工具（突变、数据库、搜索、设计等）
        ↓
   资源: 40+ 模型 | 30+ 数据集 | 11+ 数据库
```

## 工具类别

| 类别 | 说明 | Agent | CLI |
|:-----|:-----|:-----:|:---:|
| 突变 | ESM-1v/2, ProSST, ProtSSN, MIF-ST | ✅ | ✅ |
| 预测 | 30+ 微调模型 | ✅ | ✅ |
| 数据库 | 11 个集成源 | ✅ | ✅ |
| 搜索 | PubMed, FDA, 专利等 | ✅ | ✅ |
| 训练 | LoRA / QLoRA / DoRA… | ✅ | ✅ |
| 文件 | 格式转换 | ✅ | ✅ |
| Denovo | 蛋白设计 | ✅ | ✅ |
| 发现 | 新蛋白相关流程 | ✅ | ✅ |
| 可视化 | 3D 查看 | ✅ | ✅ |

## 集成资源（摘要）

- **模型**：见 [Models_CN.md](./Models_CN.md)
- **数据集**：见 [Datasets_CN.md](./Datasets_CN.md)
- **权重**：见 [Checkpoints_CN.md](./Checkpoints_CN.md)
- **数据库**：AlphaFold · RCSB · UniProt · NCBI · KEGG · STRING · BRENDA · ChEMBL · HPA · FDA · Foldseek

## 最新动态（摘录）

- [2026-04-01] [venusfactory.bio](https://venusfactory.bio/)
- [2026-03-27] 技术报告 [arXiv:2603.27303](https://arxiv.org/abs/2603.27303)
- [2026-01-23] [VenusX (ICLR2026)](https://openreview.net/forum?id=zcmL592XRG)
- [2025-04-19] [VenusREM](https://github.com/ai4protein/VenusREM) #1 ProteinGym / VenusMutHub
