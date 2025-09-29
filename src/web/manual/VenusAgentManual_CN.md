## VenusAgent: AI 助理与自动化工作流引擎

**VenusAgent** 是一个强大的引擎，它建立在**多智能体架构**之上，将传统的预测界面转变为**对话式工作流**。通过使用 **Planner (规划器)**、**Workers (执行器)** 和 **Analysis modules (分析模块)**，它帮助用户通过自然语言管理、执行和解释复杂的蛋白质工程任务。

---

## 1. 简介与核心功能

**VenusAgent** 的核心价值在于自动化实验设计和支持复杂、跨模块的工作流。

- **多智能体架构**: 包含一个用于任务分解的 **Planner**，用于执行特定预测模型的 **Workers**，以及用于结果合成的 **Analysis modules**。
- **多任务整合**: 能够整合 VenusFactory 的所有关键预测功能：**Function Prediction (功能预测)**、**Residue Prediction (残基预测)**、**Mutation Prediction (突变预测)** 和 **Physicochemical Analysis (理化性质分析)**。
- **上下文记忆**: 在同一会话线程中保留对先前步骤和数据的记忆。
- **数据集成**: 支持直接上传和引用文件（**FASTA**、**PDB**）。

---

## 2. 界面概览

VenusAgent 界面是一个简洁的聊天环境，专为流畅的对话式交互而设计。

| 元素 | 描述 |
| :--- | :--- |
| **New Chat (新建聊天)** Button | 开始一个新的会话线程，清除先前的上下文。 |
| **Delete Chat (删除聊天)** Button | 永久删除当前选定的会话记录。 |
| **Conversations (会话列表)** | 显示过去的聊天会话历史，允许用户继续未完成的工作流。 |
| **Chat Input Bar (聊天输入栏)** | 输入请求、问题或命令的主要区域。 |
| **Paperclip/Attachment Icon (附件图标)** | 允许用户直接将**数据文件**（FASTA, PDB）上传到聊天上下文中供 AI 分析。 |

---

## 3. 使用指南：对话式工作流

Agent 擅长理解和执行蛋白质工程中常见的**目标驱动型**任务，并由内部智能体架构进行智能规划。

### 3.1 核心工作流能力

| 工作流类别 | Agent 处理的任务示例 | 对应的核心模型 |
| :--- | :--- | :--- |
| **Function/Structural Localization (功能/结构定位)** | “这个蛋白质是位于细胞核的可能性有多大？” “这个锌指蛋白上的 **DNA binding site (DNA 结合位点)**在哪里？” | **Protein Function Prediction (蛋白质功能预测)** (Localization, Sorting Signal)；**Functional Residue Prediction (功能残基预测)** (Binding Site) |
| **Rational Design & Optimization (理性设计与优化)** | “找出 5 个同时增加**活性**和**稳定性**的最佳突变。” “这个序列的**不稳定性指数** $>40$，推荐 3 个稳定化突变。” | **Intelligent Directed Evolution (智能定向进化)** (Activity, Stability)；**Physical & Chemical Properties (理化性质)** |
| **Sequence & Structure Retrieval (序列与结构检索)** | “我的实验使用 UniProt ID P05798；检索它的序列和结构。” “使用 InterPro 查询这个序列的**结构域**信息。” | **InterPro Query (InterPro 查询)**；**UniProt Query (UniProt 查询)** |
| **Integrated Analysis (集成分析)** | “分析这个序列的溶解性和热稳定性，并根据结果给出实验方案建议。” | 自动整合多个预测模型并运行 **Analysis module (分析模块)**。 |

### 3.2 高级工作流命令

| 命令类型 | 示例请求 | Agent 执行的动作 (规划和执行) |
| :--- | :--- | :--- |
| **Complex Task Decomposition (复杂任务分解)** | “分析这个序列 (P60002.fasta) 的稳定性，并找出 5 个最能**增加稳定性**的突变。” | **Planner** 将请求分解为：1. 运行功能预测 (稳定性)；2. 运行定向进化 (Workers 执行)；3. 运行分析模块进行结果排序和总结。 |
| **Conditional Logic (条件逻辑)** | “如果**不稳定性指数**大于 40，推荐 5 个稳定化突变。” | 运行**理化性质分析**，并且仅在条件满足时才继续执行**突变预测**。 |
| **Report Synthesis (报告合成)** | “总结突变扫描中排名前 3 的结果，并解释它们被预测稳定的原因。” | 执行 **Analysis module**，将原始数据合成并解读为简洁的自然语言回复。 |

---

## 4. 最佳实践与注意事项

- **保持目标导向**: 直接说明您的最终实验目标（例如：“我需要一个溶解性更好的变体”），而不是罗列要按的按钮。
- **利用记忆**: 在同一聊天中，您无需重复上传序列；只需在后续请求中引用它即可。
- **信任规划器 (Planner)**: 提交您的高级目标，让内部的 **Planner** 自动为您编排所需的预测和分析步骤。
- **文件上下文**: 在请求基于结构的预测之前，务必先上传必需的 **PDB 文件**。
- **注意事项 (重要)**: **VenusAgent 不保留永久的用户数据或会话记忆。** 当您关闭浏览器或开始一个新的聊天时，之前的对话记录和上下文将被清除。**请务必自行保存**任何重要的对话记录和预测结果。