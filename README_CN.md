<div align="right">
  <a href="README.md">English</a> | <a href="README_CN.md">简体中文</a>
</div>

<p align="center">
  <img src="img/banner_2503.png" width="70%" alt="VenusFactory Banner">
</p>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/AI4Protein/VenusFactory?style=flat-square)](https://github.com/AI4Protein/VenusFactory/stargazers) [![GitHub forks](https://img.shields.io/github/forks/AI4Protein/VenusFactory?style=flat-square)](https://github.com/AI4Protein/VenusFactory/network/members) [![GitHub issues](https://img.shields.io/github/issues/AI4Protein/VenusFactory?style=flat-square)](https://github.com/AI4Protein/VenusFactory/issues) [![GitHub license](https://img.shields.io/github/license/AI4Protein/VenusFactory?style=flat-square)](https://github.com/AI4Protein/VenusFactory/blob/main/LICENSE)

[![Python Version](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/) [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen?style=flat-square)](https://venusfactory.readthedocs.io/) [![Downloads](https://img.shields.io/github/downloads/AI4Protein/VenusFactory/total?style=flat-square)](https://github.com/AI4Protein/VenusFactory/releases) [![Youtube](https://img.shields.io/badge/Youtube-VenusFactory-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=MT6lPH5kgCc&ab_channel=BxinZhou) [![Demo on OpenBayes](https://img.shields.io/badge/Demo-OpenBayes贝式计算-blue)](https://openbayes.com/console/public/tutorials/O3RCA0XUKa0)


</div>

最新消息：
- [2026-01-23] 🚀 **更新:** 在VenusFactory中新增30+下游任务预测功能。
- [2025-08-10] 🎉 VenusFactory发布了免费使用的网站 [venusfactory.cn/playground/](https://venusfactory.cn/playground/).
- [2025-06-30] 🚀 **更新:** 新增突变零样本预测功能，支持结构和序列模型的高通量突变效应评分。
- [2025-04-19] 🎉 **祝贺!** [VenusREM](https://github.com/ai4protein/VenusREM) 在 [ProteinGym](https://proteingym.org/benchmarks) 和 [VenusMutHub](https://lianglab.sjtu.edu.cn/muthub/) 排行榜中均取得第一名！
- [2025-03-26] 新增 [VenusPLM-300M](https://huggingface.co/AI4Protein/VenusPLM-300M) 模型，基于**VenusPod**独立开发，由[**Hong Liang**](https://lianglab.sjtu.edu.cn/)课题组开发。
- [2025-03-17] 新增 [Venus-PETA、Venus-ProPrime、Venus-ProSST 模型](https://huggingface.co/AI4Protein)，更多详情请参考[支持的模型](#-支持的模型)
- [2025-03-05] 🎉 **祝贺!** 课题组最新的研究成果**VenusMutHub**被[**Acta Pharmaceutica Sinica B**](https://www.sciencedirect.com/science/article/pii/S2211383525001650) 正式接收，并发布了系列[**排行榜**](https://lianglab.sjtu.edu.cn/muthub/)！

<details>
<summary>📨 欢迎加入我们的微信交流群</summary>

<p align="center">
  <img src="img/wechat.png" width="80%" alt="WeChat Group">
</p>
</details>

<details>
<summary>
  📝 您的反馈至关重要！诚邀您扫描下方任一二维码填写问卷。
</summary>
<div style="display: flex; justify-content: space-evenly; align-items: center; padding-top: 15px; text-align: center;">
  
  <div style="margin: 10px;">
    <img src="img/Questionnaire/Google_QA.png" width="200" alt="谷歌问卷二维码">
    <p style="margin-top: 5px; font-size: 14px;">谷歌问卷</p>
  </div>
  
  <div style="margin: 10px;">
    <img src="img/Questionnaire/XWJ_QA.jpg" width="200" alt="问卷星问卷二维码">
    <p style="margin-top: 5px; font-size: 14px;">问卷星</p>
  </div>

</div>
</details>

## ✏️ 目录

- [功能特点](#-功能特点)
- [支持的模型](#-支持的模型)
- [支持的训练方法](#-支持的训练方法)
- [支持的数据集](#-支持的数据集)
- [支持的评估指标](#-支持的评估指标)
- [Agent-0.1: 您的AI助手](#-Agent-0.1:您的AI助手)
- [快速工具：您的快速预测之选](#-快速工具：您的快速预测之选)
- [高级工具：用于深度蛋白质分析](#-高级工具：用于深度蛋白质分析)
- [环境要求](#-环境要求)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [命令行使用](#-命令行使用)
- [引用](#-引用)
- [致谢](#-致谢)

## 📑 功能特点

🙌     ***VenusFactory*** 是一个统一的蛋白质工程开放平台，同时支持**图形用户界面（GUI）**和**命令行操作**。它通过简化的无代码工作流，实现了**数据检索、模型训练、评估和部署**。

## 核心亮点

-   **本地私有部署**：支持本地化私有部署，确保数据安全
-   **领先模型集成**：集成了超过40种最先进的深度学习模型，降低了科研门槛，加速了人工智能在生命科学领域的应用
-   **AI 智能助手**：**Agent-0.1** 作为一个智能AI助手，为蛋白质工程任务提供专业的解答和分析
-   **高效工作流**：**快速工具（Quick Tools）**为蛋白质功能和突变效果评分等常见任务提供快速、无代码的预测
-   **深度分析**：**高级工具（Advanced Tools）**为基于序列和基于结构的零样本突变预测提供强大而深入的分析
-   **多样化的蛋白质语言模型**：Venus系列、ESM系列、ProtTrans系列、Ankh系列等
-   **全面的监督数据集**：定位（Localization）、适应度（Fitness）、溶解度（Solubility）、稳定性（Stability）等
-   **简便快捷的数据收集器**：AlphaFold2数据库、RCSB、InterPro、UniProt等
-   **实验监控**：Wandb、本地监控
-   **友好的用户界面**：Gradio UI

<p align="center">
  <video width="80%" controls>
    <source src="./img/venusfactory.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

## 🧬 支持的模型 (零样本预测)

### 序列-结构模型

[ProSST, NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf), [ProtSSN, eLife2025](https://elifesciences.org/articles/98033), [MIF-ST, PEDS2022](https://academic.oup.com/peds/article-abstract/doi/10.1093/protein/gzad015/7330543?redirectedFrom=fulltext)

### 结构模型

[MIF, PEDS2022](https://academic.oup.com/peds/article-abstract/doi/10.1093/protein/gzad015/7330543?redirectedFrom=fulltext)

### 序列模型

[ESM2, Science2023](https://www.science.org/doi/10.1126/science.ade2574), [ESM-1v, NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf)


## 🤖 支持的模型 (监督任务)

### 预训练蛋白质语言模型

<details>
<summary>Venus系列模型：特定任务架构</summary>

| 模型 | 大小 | 参数量 | GPU内存 | 特点 | 模板 |
|-------|------|------------|------------|----------|----------|
| ProSST-20 | 20 | 110M | 4GB+ | 突变预测 | [AI4Protein/ProSST-20](https://huggingface.co/AI4Protein/ProSST-20) |
| ProSST-128 | 128 | 110M | 4GB+ | 突变预测 | [AI4Protein/ProSST-128](https://huggingface.co/AI4Protein/ProSST-128) |
| ProSST-512 | 512 | 110M | 4GB+ | 突变预测 | [AI4Protein/ProSST-512](https://huggingface.co/AI4Protein/ProSST-512) |
| ProSST-1024 | 1024 | 110M | 4GB+ | 突变预测 | [AI4Protein/ProSST-1024](https://huggingface.co/AI4Protein/ProSST-1024) |
| ProSST-2048 | 2048 | 110M | 4GB+ | 突变预测 | [AI4Protein/ProSST-2048](https://huggingface.co/AI4Protein/ProSST-2048) |
| ProSST-4096 | 4096 | 110M | 4GB+ | 突变预测 | [AI4Protein/ProSST-4096](https://huggingface.co/AI4Protein/ProSST-4096) |
| ProPrime-690M | 690M | 690M | 16GB+ | OGT预测 | [AI4Protein/Prime_690M](https://huggingface.co/AI4Protein/Prime_690M) |

> 💡 这些模型在特定任务上表现出色或提供独特的架构优势
</details>

<details>
<summary>Venus-PETA 模型：分词变体</summary>

#### BPE 分词系列
| 模型 | 词表大小 | 参数量 | GPU内存 | 模板 |
|-------|------------|------------|------------|----------|
| PETA-base | base | 80M | 4GB+ | [AI4Protein/deep_base](https://huggingface.co/AI4Protein/deep_base) |
| PETA-bpe-50 | 50 | 80M | 4GB+ | [AI4Protein/deep_bpe_50](https://huggingface.co/AI4Protein/deep_bpe_50) |
| PETA-bpe-100 | 100 | 80M | 4GB+ | [AI4Protein/deep_bpe_100](https://huggingface.co/AI4Protein/deep_bpe_100) |
| PETA-bpe-200 | 200 | 80M | 4GB+ | [AI4Protein/deep_bpe_200](https://huggingface.co/AI4Protein/deep_bpe_200) |
| PETA-bpe-400 | 400 | 80M | 4GB+ | [AI4Protein/deep_bpe_400](https://huggingface.co/AI4Protein/deep_bpe_400) |
| PETA-bpe-800 | 800 | 80M | 4GB+ | [AI4Protein/deep_bpe_800](https://huggingface.co/AI4Protein/deep_bpe_800) |
| PETA-bpe-1600 | 1600 | 80M | 4GB+ | [AI4Protein/deep_bpe_1600](https://huggingface.co/AI4Protein/deep_bpe_1600) |
| PETA-bpe-3200 | 3200 | 80M | 4GB+ | [AI4Protein/deep_bpe_3200](https://huggingface.co/AI4Protein/deep_bpe_3200) |

#### Unigram 分词系列
| 模型 | 词表大小 | 参数量 | GPU内存 | 模板 |
|-------|------------|------------|------------|----------|
| PETA-unigram-50 | 50 | 80M | 4GB+ | [AI4Protein/deep_unigram_50](https://huggingface.co/AI4Protein/deep_unigram_50) |
| PETA-unigram-100 | 100 | 80M | 4GB+ | [AI4Protein/deep_unigram_100](https://huggingface.co/AI4Protein/deep_unigram_100) |
| PETA-unigram-200 | 200 | 80M | 4GB+ | [AI4Protein/deep_unigram_200](https://huggingface.co/AI4Protein/deep_unigram_200) |
| PETA-unigram-400 | 400 | 80M | 4GB+ | [AI4Protein/deep_unigram_400](https://huggingface.co/AI4Protein/deep_unigram_400) |
| PETA-unigram-800 | 800 | 80M | 4GB+ | [AI4Protein/deep_unigram_800](https://huggingface.co/AI4Protein/deep_unigram_800) |
| PETA-unigram-1600 | 1600 | 80M | 4GB+ | [AI4Protein/deep_unigram_1600](https://huggingface.co/AI4Protein/deep_unigram_1600) |
| PETA-unigram-3200 | 3200 | 80M | 4GB+ | [AI4Protein/deep_unigram_3200](https://huggingface.co/AI4Protein/deep_unigram_3200) |

> 💡 不同的分词策略可能更适合特定任务
</details>

<details>
<summary>ESM 系列模型：Meta AI 的蛋白质语言模型</summary>

| 模型 | 大小 | 参数量 | GPU内存 | 训练数据 | 模板 |
|-------|------|------------|------------|---------------|----------|
| ESM2-8M | 8M | 8M | 2GB+ | UR50/D | [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D) |
| ESM2-35M | 35M | 35M | 4GB+ | UR50/D | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) |
| ESM2-150M | 150M | 150M | 8GB+ | UR50/D | [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D) |
| ESM2-650M | 650M | 650M | 16GB+ | UR50/D | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |
| ESM2-3B | 3B | 3B | 24GB+ | UR50/D | [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D) |
| ESM2-15B | 15B | 15B | 40GB+ | UR50/D | [facebook/esm2_t48_15B_UR50D](https://huggingface.co/facebook/esm2_t48_15B_UR50D) |

> 💡 ESM2 模型是最新一代，性能优于 ESM-1b/1v
</details>

<details>
<summary>BERT 系列模型：基于 Transformer 编码器架构</summary>

| 模型 | 大小 | 参数量 | GPU内存 | 训练数据 | 模板 |
|-------|------|------------|------------|---------------|----------|
| ProtBert-Uniref100 | 420M | 420M | 12GB+ | UniRef100 | [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert) |
| ProtBert-BFD | 420M | 420M | 12GB+ | BFD100 | [Rostlab/prot_bert_bfd](https://huggingface.co/Rostlab/prot_bert_bfd) |
| IgBert | 420M | 420M | 12GB+ | 抗体 | [Exscientia/IgBert](https://huggingface.co/Exscientia/IgBert) |
| IgBert-unpaired | 420M | 420M | 12GB+ | 抗体 | [Exscientia/IgBert_unpaired](https://huggingface.co/Exscientia/IgBert_unpaired) |

> 💡 BFD 训练的模型在结构相关任务上表现更好
</details>

<details>
<summary>T5 系列模型：编码器-解码器架构</summary>

| 模型 | 大小 | 参数量 | GPU内存 | 训练数据 | 模板 |
|-------|------|------------|------------|---------------|----------|
| ProtT5-XL-UniRef50 | 3B | 3B | 24GB+ | UniRef50 | [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) |
| ProtT5-XXL-UniRef50 | 11B | 11B | 40GB+ | UniRef50 | [Rostlab/prot_t5_xxl_uniref50](https://huggingface.co/Rostlab/prot_t5_xxl_uniref50) |
| ProtT5-XL-BFD | 3B | 3B | 24GB+ | BFD100 | [Rostlab/prot_t5_xl_bfd](https://huggingface.co/Rostlab/prot_t5_xl_bfd) |
| ProtT5-XXL-BFD | 11B | 11B | 40GB+ | BFD100 | [Rostlab/prot_t5_xxl_bfd](https://huggingface.co/Rostlab/prot_t5_xxl_bfd) |
| Ankh-base | 450M | 450M | 12GB+ | 编码器-解码器 | [ElnaggarLab/ankh-base](https://huggingface.co/ElnaggarLab/ankh-base) |
| Ankh-large | 1.2B | 1.2B | 20GB+ | 编码器-解码器 | [ElnaggarLab/ankh-large](https://huggingface.co/ElnaggarLab/ankh-large) |

> 💡 T5 模型可用于编码和生成任务
</details>

### 模型选择指南

<details>
<summary>如何选择合适的模型？</summary>

1. **基于硬件限制：**
   - 低配GPU (<8GB)：ESM2-8M、ESM2-35M、ProSST
   - 中配GPU (8-16GB)：ESM2-150M、ESM2-650M、ProtBert系列
   - 高配GPU (24GB+)：ESM2-3B、ProtT5-XL、Ankh-large
   - 多GPU：ESM2-15B、ProtT5-XXL

2. **基于任务类型：**
   - 序列分类：ESM2、ProtBert
   - 结构预测：ESM2、Ankh
   - 生成任务：ProtT5
   - 抗体设计：IgBert、IgT5
   - 轻量级部署：ProSST、PETA-base

3. **基于训练数据：**
   - 通用蛋白质任务：ESM2、ProtBert
   - 结构感知任务：Ankh
   - 抗体特异性：IgBert、IgT5
   - 自定义分词需求：PETA系列

</details>

> 🔍 所有模型都可通过Hugging Face Hub获取，使用其模板可轻松加载。

## 🔬 支持的训练方法

<details>
<summary>支持的训练方法</summary>

| 方法 | 全量微调 | 冻结微调 | SES-Adapter | AdaLoRA | QLoRA | LoRA | DoRA | IA3 |
|------|---------|----------|-------------|----------|--------|------|------|-----|
| 监督微调 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
</details>

## 📚 支持的数据集

<details><summary>预训练数据集</summary>

| 数据集 | 数据来源 |
|-------|----------|
| [CATH_V43_S40](https://huggingface.co/datasets/tyang816/cath) | 结构数据集
| [AGO_family](https://huggingface.co/datasets/tyang816/Ago_database_PDB) | 结构数据集

</details>

<details><summary>零样本数据集</summary>

| 数据集 | 任务 | 数据来源 |
|-------|----------|----------|
| [VenusMutHub](https://huggingface.co/datasets/AI4Protein/VenusMutHub) | 突变 | 蛋白质序列
| [ProteinGym](https://proteingym.org/) | 突变 | 蛋白质序列
</details>

<details><summary>监督微调数据集（氨基酸序列/foldseek序列/二级结构序列）</summary>

| 数据集 | 任务 | 数据层次 | 问题类型 | 数据来源 |
|-------|------|------------|----------|----------|
| DeepLocBinary | 定位 | 蛋白质级别 | 单标签分类 | [DeepLocBinary_AlphaFold2](https://huggingface.co/datasets/AI4Protein/DeepLocBinary_AlphaFold2), [DeepLocBinary_ESMFold](https://huggingface.co/datasets/AI4Protein/DeepLocBinary_ESMFold) |
| DeepLocMulti | 定位 | 蛋白质级别 | 多标签分类 | [DeepLocMulti_AlphaFold2](https://huggingface.co/datasets/AI4Protein/DeepLocMulti_AlphaFold2), [DeepLocMulti_ESMFold](https://huggingface.co/datasets/AI4Protein/DeepLocMulti_ESMFold) |
| DeepLoc2Multi | 定位 | 蛋白质级别 | 多标签分类 | [DeepLoc2Multi_AlphaFold2](https://huggingface.co/datasets/AI4Protein/DeepLoc2Multi_AlphaFold2), [DeepLoc2Multi_ESMFold](https://huggingface.co/datasets/AI4Protein/DeepLoc2Multi_ESMFold) |
| DeepSol | 溶解度 | 蛋白质级别 | 单标签分类 | [DeepSol_AlphaFold2](https://huggingface.co/datasets/AI4Protein/DeepSol_AlphaFold2), [DeepSol_ESMFold](https://huggingface.co/datasets/AI4Protein/DeepSol_ESMFold) |
| DeepSoluE | 溶解度 | 蛋白质级别 | 单标签分类 | [DeepSoluE_ESMFold](https://huggingface.co/datasets/AI4Protein/DeepSoluE_ESMFold) |
| ProtSolM | 溶解度 | 蛋白质级别 | 单标签分类 | [ProtSolM_ESMFold](https://huggingface.co/datasets/AI4Protein/ProtSolM_ESMFold) |
| eSOL | 溶解度 | 蛋白质级别 | 回归 | [eSOL_AlphaFold2](https://huggingface.co/datasets/AI4Protein/eSOL_AlphaFold2), [eSOL_ESMFold](https://huggingface.co/datasets/AI4Protein/eSOL_ESMFold) |
| DeepET_Topt | 最适酶活 | 蛋白质级别 | 回归 | [DeepET_Topt_AlphaFold2](https://huggingface.co/datasets/AI4Protein/DeepET_Topt_AlphaFold2), [DeepET_Topt_ESMFold](https://huggingface.co/datasets/AI4Protein/DeepET_Topt_ESMFold) |
| EC | 功能 | 蛋白质级别 | 多标签分类 | [EC_AlphaFold2](https://huggingface.co/datasets/AI4Protein/EC_AlphaFold2), [EC_ESMFold](https://huggingface.co/datasets/AI4Protein/EC_ESMFold) |
| GO_BP | 功能 | 蛋白质级别 | 多标签分类 | [GO_BP_AlphaFold2](https://huggingface.co/datasets/AI4Protein/GO_BP_AlphaFold2), [GO_BP_ESMFold](https://huggingface.co/datasets/AI4Protein/GO_BP_ESMFold) |
| GO_CC | 功能 | 蛋白质级别 | 多标签分类 | [GO_CC_AlphaFold2](https://huggingface.co/datasets/AI4Protein/GO_CC_AlphaFold2), [GO_CC_ESMFold](https://huggingface.co/datasets/AI4Protein/GO_CC_ESMFold) |
| GO_MF | 功能 | 蛋白质级别 | 多标签分类 | [GO_MF_AlphaFold2](https://huggingface.co/datasets/AI4Protein/GO_MF_AlphaFold2), [GO_MF_ESMFold](https://huggingface.co/datasets/AI4Protein/GO_MF_ESMFold) |
| MetalIonBinding | 结合 | 蛋白质级别 | 单标签分类 | [MetalIonBinding_AlphaFold2](https://huggingface.co/datasets/AI4Protein/MetalIonBinding_AlphaFold2), [MetalIonBinding_ESMFold](https://huggingface.co/datasets/AI4Protein/MetalIonBinding_ESMFold) |
| Thermostability | 稳定性 | 蛋白质级别 | 回归 | [Thermostability_AlphaFold2](https://huggingface.co/datasets/AI4Protein/Thermostability_AlphaFold2), [Thermostability_ESMFold](https://huggingface.co/datasets/AI4Protein/Thermostability_ESMFold) |

> 💡 每个数据集都提供了使用 AlphaFold2 和 ESMFold 生成的结构序列版本
</details>

<details><summary>监督微调数据集（氨基酸序列）</summary>

| 数据集 | 任务 | 数据层次 | 问题类型 | 数据来源 |
|-------|------|------------|----------|----------|
| Demo_Solubility | 溶解度 | 蛋白质级别 | 单标签分类 | [Demo_Solubility](https://huggingface.co/datasets/AI4Protein/Demo_Solubility) |
| DeepLocBinary | 定位 | 蛋白质级别 | 单标签分类 | [DeepLocBinary](https://huggingface.co/datasets/AI4Protein/DeepLocBinary) |
| DeepLocMulti | 定位 | 蛋白质级别 | 单标签分类 | [DeepLocMulti](https://huggingface.co/datasets/AI4Protein/DeepLocMulti) |
| DeepLoc2Multi | 定位 | 蛋白质级别 | 多标签分类 | [DeepLoc2Multi](https://huggingface.co/datasets/AI4Protein/DeepLoc2Multi) |
| DeepSol | 溶解度 | 蛋白质级别 | 单标签分类 | [DeepSol](https://huggingface.co/datasets/AI4Protein/DeepSol) |
| DeepSoluE | 溶解度 | 蛋白质级别 | 单标签分类 | [DeepSoluE](https://huggingface.co/datasets/AI4Protein/DeepSoluE) |
| ProtSolM | 溶解度 | 蛋白质级别 | 单标签分类 | [ProtSolM](https://huggingface.co/datasets/AI4Protein/ProtSolM) |
| eSOL | 溶解度 | 蛋白质级别 | 回归 | [eSOL](https://huggingface.co/datasets/AI4Protein/eSOL) |
| DeepET_Topt | 最适酶活 | 蛋白质级别 | 回归 | [DeepET_Topt](https://huggingface.co/datasets/AI4Protein/DeepET_Topt) |
| EC | 功能 | 蛋白质级别 | 多标签分类 | [EC](https://huggingface.co/datasets/AI4Protein/EC) |
| GO_BP | 功能 | 蛋白质级别 | 多标签分类 | [GO_BP](https://huggingface.co/datasets/AI4Protein/GO_BP) |
| GO_CC | 功能 | 蛋白质级别 | 多标签分类 | [GO_CC](https://huggingface.co/datasets/AI4Protein/GO_CC) |
| GO_MF | 功能 | 蛋白质级别 | 多标签分类 | [GO_MF](https://huggingface.co/datasets/AI4Protein/GO_MF) |
| MetalIonBinding | 结合 | 蛋白质级别 | 单标签分类 | [MetalIonBinding](https://huggingface.co/datasets/AI4Protein/MetalIonBinding) |
| Thermostability | 稳定性 | 蛋白质级别 | 回归 | [Thermostability](https://huggingface.co/datasets/AI4Protein/Thermostability) |
| PaCRISPR | CRISPR | 蛋白质级别 | 回归 | [PaCRISPR](https://huggingface.co/datasets/AI4Protein/PaCRISPR) |
| PETA_CHS_Sol | 溶解度 | 蛋白质级别 | 回归 | [PETA_CHS_Sol](https://huggingface.co/datasets/AI4Protein/PETA_CHS_Sol) |
| PETA_LGK_Sol | 溶解度 | 蛋白质级别 | 回归 | [PETA_LGK_Sol](https://huggingface.co/datasets/AI4Protein/PETA_LGK_Sol) |
| PETA_TEM_Sol | 溶解度 | 蛋白质级别 | 回归 | [PETA_TEM_Sol](https://huggingface.co/datasets/AI4Protein/PETA_TEM_Sol) |
| SortingSignal | 信号肽 | 蛋白质级别 | 回归 | [SortingSignal](https://huggingface.co/datasets/AI4Protein/SortingSignal) |
| FLIP_AAV | 突变 | 蛋白质点位 | 回归 |  |
| FLIP_AAV_one-vs-rest | 突变 | 蛋白质点位 | 回归 | [FLIP_AAV_one-vs-rest](https://huggingface.co/datasets/AI4Protein/FLIP_AAV_one-vs-rest) |
| FLIP_AAV_two-vs-rest | 突变 | 蛋白质点位 | 回归 | [FLIP_AAV_two-vs-rest](https://huggingface.co/datasets/AI4Protein/FLIP_AAV_two-vs-rest) |
| FLIP_AAV_mut-des | 突变 | 蛋白质点位 | 回归 | [FLIP_AAV_mut-des](https://huggingface.co/datasets/AI4Protein/FLIP_AAV_mut-des) |
| FLIP_AAV_des-mut | 突变 | 蛋白质点位 | 回归 | [FLIP_AAV_des-mut](https://huggingface.co/datasets/AI4Protein/FLIP_AAV_des-mut) |
| FLIP_AAV_seven-vs-rest | 突变 | 蛋白质点位 | 回归 | [FLIP_AAV_seven-vs-rest](https://huggingface.co/datasets/AI4Protein/FLIP_AAV_seven-vs-rest) |
| FLIP_AAV_low-vs-high | 突变 | 蛋白质点位 | 回归 | [FLIP_AAV_low-vs-high](https://huggingface.co/datasets/AI4Protein/FLIP_AAV_low-vs-high) |
| FLIP_AAV_sampled | 突变 | 蛋白质点位 | 回归 | [FLIP_AAV_sampled](https://huggingface.co/datasets/AI4Protein/FLIP_AAV_sampled) |
| FLIP_GB1 | 突变 | 蛋白质点位 | 回归 | |
| FLIP_GB1_one-vs-rest | 突变 | 蛋白质点位 | 回归 | [FLIP_GB1_one-vs-rest](https://huggingface.co/datasets/AI4Protein/FLIP_GB1_one-vs-rest) |
| FLIP_GB1_two-vs-rest | 突变 | 蛋白质点位 | 回归 | [FLIP_GB1_two-vs-rest](https://huggingface.co/datasets/AI4Protein/FLIP_GB1_two-vs-rest) |
| FLIP_GB1_three-vs-rest | 突变 | 蛋白质点位 | 回归 | [FLIP_GB1_three-vs-rest](https://huggingface.co/datasets/AI4Protein/FLIP_GB1_three-vs-rest) |
| FLIP_GB1_low-vs-high | 突变 | 蛋白质点位 | 回归 | [FLIP_GB1_low-vs-high](https://huggingface.co/datasets/AI4Protein/FLIP_GB1_low-vs-high) |
| FLIP_GB1_sampled | 突变 | 蛋白质点位 | 回归 | [FLIP_GB1_sampled](https://huggingface.co/datasets/AI4Protein/FLIP_GB1_sampled) |
| TAPE_Fluorescence | 突变 | 蛋白质点位 | 回归 | [TAPE_Fluorescence](https://huggingface.co/datasets/AI4Protein/TAPE_Fluorescence) |
| TAPE_Stability | 突变 | 蛋白质点位 | 回归 | [TAPE_Stability](https://huggingface.co/datasets/AI4Protein/TAPE_Stability) |


> 💡 不同数据集的序列结构不同，例如 ``DeepLocBinary_ESMFold`` 和 ``DeepLocBinary_AlphaFold2`` 共享相同的氨基酸序列，因此如果您只想使用 ``aa_seqs``，两者都可以使用！

</details>


## 📈 支持的评估指标
<details>
<summary>支持的评估指标</summary>

| 名称 | Torchmetrics | 问题类型 |
|------|--------------|----------|
| accuracy | Accuracy | 单标签分类/多标签分类 |
| recall | Recall | 单标签分类/多标签分类 |
| precision | Precision | 单标签分类/多标签分类 |
| f1 | F1Score | 单标签分类/多标签分类 |
| mcc | MatthewsCorrCoef | 单标签分类/多标签分类 |
| auc | AUROC | 单标签分类/多标签分类 |
| f1_max | F1ScoreMax | 多标签分类 |
| spearman_corr | SpearmanCorrCoef | 回归 |
| mse | MeanSquaredError | 回归 |

</details>

## 🤖 Agent-0.1: 您的AI助手

`Agent-0.1` 是集成在 VenusFactory 平台中的智能AI助手，旨在回答有关蛋白质工程和生物信息学的问题，并提供深入分析。它扮演着专业专家的角色，帮助生物学家和AI研究人员简化他们的研究工作流程。

### 主要功能：

- **零样本预测**：直接利用尖端的基于序列（例如 ESM-2, ESM-1v, ESM-1b）和基于结构（例如 SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST）的模型，执行零样本突变预测。
- **蛋白质功能预测**：精确预测各种蛋白质功能，包括溶解度、定位、金属离子结合、稳定性、分选信号和最适温度。
- **清晰洞见**：始终对您的查询提供清晰、可操作的洞见。

> 💡 **注意**：此功能需要API密钥才能访问，目前处于测试阶段（Beta）。

## ⚡️ 快速工具：您的快速预测之选
**快速工具**专为需要快速、高效和直接分析而无需大量配置的用户设计。它提供了两个关键预测任务的无代码入口

**定向进化：AI驱动的突变预测**
该工具可用于快速评估和分析蛋白质突变。只需上传一个PDB文件或粘贴PDB内容，平台将提供关于单个或多个突变对蛋白质影响的见解

**蛋白质功能预测**
该模块利用预训练模型，根据给定的氨基酸序列预测各种蛋白质功能。您可以上传一个FASTA文件或直接粘贴序列来预测溶解度、定位等特性

---

## 🧪 高级工具：用于深度蛋白质分析
**高级工具**专为需要更精细控制和更深层分析的研究人员而构建。它通过允许您选择两种不同模型类型，提供了强大的零样本预测能力

**基于序列的模型**
该子模块侧重于使用强大的仅基于序列的模型（如**ESM-2**）进行高通量突变效果评分。您可以上传一个FASTA文件或粘贴一个蛋白质序列来执行大规模预测和突变评分

**基于结构的模型**
对于需要深入了解蛋白质三维几何形状的任务，该工具利用了**结构感知模型**（如**ESM-IF1**）。通过上传一个PDB文件或粘贴其内容，您可以执行复杂的零样本预测，这些预测会考虑蛋白质的空间上下文


## ✈️ 环境要求

### 硬件要求
- 推荐：NVIDIA RTX 3090 (24GB) 或更好
- 实际要求取决于您选择的蛋白质语言模型

### 软件要求
- [Anaconda3](https://www.anaconda.com/download) 或 [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 3.12

## 📦 安装指南
<details><summary> 在macOS上开始</summary>

## 为了获得最佳性能和体验，我们推荐使用 ​带有M系列芯片的Mac设备​（如 M1、M2、M3 等）

## 1️⃣ 克隆仓库

首先，从Github获取VenusFactory的代码：

```bash
git clone https://github.com/AI4Protein/VenusFactory.git
cd VenusFactory
```

## 2️⃣ 创建Conda环境

确保已安装Anaconda或Miniconda。然后，创建一个名为`venus`的新环境，使用Python 3.12：

```bash
conda create -n venus python=3.12
conda activate venus
```

## 3️⃣ 安装PyTorch和PyG依赖项

```bash
# 安装PyTorch
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# 安装PyG依赖项
pip install torch_scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

## 4️⃣ 安装其他依赖项

使用`requirements_for_macOS.txt`安装剩余依赖项:
```bash
pip install -r requirements_for_macOS.txt
```
</details>

<details><summary> 在Windows或Linux上开始(使用CUDA 12.X)</summary>

## 我们推荐使用CUDA 12.2


## 1️⃣ 克隆仓库

首先，从Github获取VenusFactory的代码：

```bash
git clone https://github.com/AI4Protein/VenusFactory.git
cd VenusFactory
```

## 2️⃣ 创建Conda环境

确保已安装Anaconda或Miniconda。然后，创建一个名为`venus`的新环境，使用Python 3.12：


```bash
conda create -n venus python=3.12
conda activate venus
```

## 3️⃣ 安装PyTorch和PyG依赖项

```bash
# 安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装PyG依赖项
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

## 4️⃣ 安装其他依赖项

使用`requirements.txt`安装剩余依赖项:
```bash
pip install -r requirements.txt
```
</details>

<details><summary> 在Windows或Linux上开始(使用CUDA 11.X)</summary>

## 我们推荐使用CUDA 11.8或更高版本，因为它们支持更高版本的PyTorch，提供更好的体验。


## 1️⃣ 克隆仓库

首先，从Github获取VenusFactory的代码：

```bash
git clone https://github.com/AI4Protein/VenusFactory.git
cd VenusFactory
```

## 2️⃣ 创建Conda环境

确保已安装Anaconda或Miniconda。然后，创建一个名为`venus`的新环境，使用Python 3.12：


```bash
conda create -n venus python=3.12
conda activate venus
```

## 3️⃣ 安装PyTorch和PyG依赖项

```bash
# 安装PyTorch
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118

# 安装PyG依赖项
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

## 4️⃣ 安装其他依赖项

使用`requirements.txt`安装剩余依赖项:
```bash
pip install -r requirements.txt
```
</details>

<details><summary> 在Windows或Linux上开始(使用CPU)</summary>

## 1️⃣ 克隆仓库

首先，从Github获取VenusFactory的代码：

```bash
git clone https://github.com/AI4Protein/VenusFactory.git
cd VenusFactory
```

## 2️⃣ 创建Conda环境

确保已安装Anaconda或Miniconda。然后，创建一个名为`venus`的新环境，使用Python 3.12：


```bash
conda create -n venus python=3.12
conda activate venus
```

## 3️⃣ 安装PyTorch和PyG依赖项

```bash
# 安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装PyG依赖项
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

## 4️⃣ 安装其他依赖项

使用`requirements.txt`安装剩余依赖项:
```bash
pip install -r requirements.txt
```
</details>


## 🚀 快速开始

### 启动 Venus Web UI

使用我们基于 [Gradio](https://github.com/gradio-app/gradio) 的直观图形界面快速开始：

```bash
python ./src/webui.py
```

您可以：
- 配置并运行微调实验
- 监控训练进度
- 评估模型
- 可视化结果

### 使用各个标签页

我们提供详细的指南帮助您浏览每个标签页。

<details>
<summary>1. 训练标签页：训练您自己的蛋白质语言模型</summary>

![Model_Dataset_Config](img/Train/Model_Dataset_Config.png)

从下拉菜单中选择蛋白质语言模型。上传您的数据集或选择可用数据集，并选择适合您问题类型的评估指标。

![Training_Parameters](img/Train/Training_Parameters.png)
选择训练方法（Freeze、SES-Adapter、LoRA、QLoRA等）并配置训练参数（批量大小、学习率等）。

![Preview_Command](img/Train/Preview_Command.png)
![Training_Progress](img/Train/Training_Progress.png)
![Best_Model](img/Train/Best_Model.png)
![Monitor_Figs](img/Train/Monitor_Figs.png)
点击"开始训练"并实时监控进度。

<p align="center">
  <img src="img/Train/Metric_Results.png" width="60%" alt="Metric_Results">
</p>

点击"下载CSV"下载测试指标结果。
</details>

<details>
<summary>2. 评估标签页：在基准测试中评估您的训练模型</summary>

![Model_Dataset_Config](img/Eval/Model_Dataset_Config.png)

通过指定模型路径加载您的训练模型。选择训练时使用的相同蛋白质语言模型和模型配置。选择测试数据集并配置批量大小。选择适合您问题类型的评估指标。最后，点击"开始评估"查看性能指标。
</details>

<details>
<summary>3. 预测标签页：使用您的训练模型进行样本预测</summary>

![Predict_Tab](img/Predict/Predict_Tab.png)

通过指定模型路径加载您的训练模型。选择训练时使用的相同蛋白质语言模型和模型配置。

单序列预测：在文本框中输入蛋白质序列。

批量预测：上传包含序列的CSV文件。

![Batch](img/Predict/Batch.png)

点击"预测"生成并查看结果。
</details>

<details>
<summary>4. 下载标签页：高效收集来自不同来源的数据</summary>

- **AlphaFold2结构**：输入UniProt ID下载蛋白质结构
- **UniProt**：使用关键词或ID搜索蛋白质信息
- **InterPro**：获取蛋白质家族和结构域信息
- **RCSB PDB**：下载实验蛋白质结构
</details>

<details>
<summary>5. 手册标签页：详细文档和指南</summary>

选择语言（英文/中文）。

使用目录导航文档并找到分步指南。
</details>

## 🧬 命令行使用

对于偏好命令行界面的用户，我们提供全面的脚本解决方案。

<details>
<summary>训练方法：适应不同需求的各种微调方法</summary>

### 全模型微调
```bash
# 冻结微调：训练特定层同时冻结其他层
bash ./script/train/train_plm_vanilla.sh
```

### 参数高效微调 (PEFT)
```bash
# SES-Adapter：选择性和高效的适配器微调
bash ./script/train/train_plm_ses-adapter.sh

# AdaLoRA：自适应低秩适配
bash ./script/train/train_plm_adalora.sh

# QLoRA：量化低秩适配
bash ./script/train/train_plm_qlora.sh

# LoRA：低秩适配
bash ./script/train/train_plm_lora.sh

# DoRA：双低秩适配
bash ./script/train/train_plm_dora.sh

# IA3：通过抑制和放大内部激活的注入适配器
bash ./script/train/train_plm_ia3.sh
```

#### 训练方法比较
| 方法 | 内存使用 | 训练速度 | 性能 |
|------|----------|----------|------|
| Freeze | 低 | 快 | 良好 |
| SES-Adapter | 中等 | 中等 | 更好 |
| AdaLoRA | 低 | 中等 | 更好 |
| QLoRA | 非常低 | 较慢 | 良好 |
| LoRA | 低 | 快 | 良好 |
| DoRA | 低 | 中等 | 更好 |
| IA3 | 非常低 | 快 | 良好 |

</details>

<details>
<summary>模型评估：全面的评估工具</summary>

### 基础评估
```bash
# 在测试集上评估模型性能
bash ./script/eval/eval.sh
```

### 可用指标
- 分类：准确率、精确率、召回率、F1、MCC、AUC
- 回归：MSE、Spearman相关系数
- 多标签：F1-max

### 可视化工具
- 训练曲线
- 混淆矩阵
- ROC曲线
- 性能比较图

</details>

<details>
<summary>结构序列工具：处理蛋白质结构信息</summary>

### ESM结构序列
```bash
# 使用ESM-3生成结构序列
bash ./script/get_get_structure_seq/get_esm3_structure_seq.sh
```

### 二级结构
```bash
# 预测蛋白质二级结构
bash ./script/get_get_structure_seq/get_secondary_structure_seq.sh
```

特点：
- 支持多种序列格式
- 批处理能力
- 与流行的结构预测工具集成

</details>

<details>
<summary>数据收集工具：多源蛋白质数据获取</summary>

### 格式转换
```bash
# 将CIF格式转换为PDB
bash ./script/search/convert/maxit.sh
```

### 元数据收集
```bash
# 从RCSB PDB下载元数据
bash ./script/search/download/metadata/download_rcsb.sh
```

### 序列数据
```bash
# 从UniProt下载蛋白质序列
bash ./script/search/sequence/download_uniprot_seq.sh
```

### 结构数据
```bash
# 从AlphaFold2数据库下载
bash ./script/search/download/structure/download_alphafold.sh

# 从RCSB PDB下载
bash ./script/search/download/structure/download_rcsb.sh
```

特点：
- 自动批量下载
- 断点续传
- 数据完整性验证
- 多源支持
- 自定义搜索条件

#### 支持的数据库
| 数据库 | 数据类型 | 访问方式 | 速率限制 |
|--------|----------|----------|----------|
| AlphaFold2 | 结构 | REST API | 是 |
| RCSB PDB | 结构 | FTP/HTTP | 否 |
| UniProt | 序列 | REST API | 是 |
| InterPro | 结构域 | REST API | 是 |

</details>

<details>
<summary>使用示例：常见场景和解决方案</summary>

### 训练示例
```bash
# 使用ESM2训练蛋白质溶解度预测器
bash ./script/train/train_plm_lora.sh \
    --model "facebook/esm2_t33_650M_UR50D" \
    --dataset "DeepSol" \
    --batch_size 32 \
    --learning_rate 1e-4
```

### 评估示例
```bash
# 评估训练好的模型
bash ./script/eval/eval.sh \
    --model_path "path/to/your/model" \
    --test_dataset "DeepSol_test"
```

### 数据收集示例
```bash
# 下载UniProt ID列表对应的结构
bash ./script/search/download/structure/download_alphafold.sh \
    --input uniprot_ids.txt \
    --output ./structures
```

</details>

> 💡 所有脚本都支持额外的命令行参数进行自定义。使用任何脚本的 `--help` 选项查看可用选项。

## 🙌 引用

如果您使用了我们的代码或数据，请引用我们的工作：

```bibtex
@inproceedings{tan-etal-2025-venusfactory,
    title = "{V}enus{F}actory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning",
    author = "Tan, Yang and Liu, Chen and Gao, Jingyuan and Wu, Banghao and Li, Mingchen and Wang, Ruilin and Zhang, Lingrong and Yu, Huiqun and Fan, Guisheng and Hong, Liang and Zhou, Bingxin",
    editor = "Mishra, Pushkar and Muresan, Smaranda and Yu, Tao",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-demo.23/",
    doi = "10.18653/v1/2025.acl-demo.23",
    pages = "230--241",
    ISBN = "979-8-89176-253-4",
}
```

## 🎊 致谢

感谢 [Liang's Lab](https://ins.sjtu.edu.cn/people/lhong/index.html) 的支持。 
