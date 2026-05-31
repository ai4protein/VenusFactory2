# Advanced Tools — 完全掌控模型与数据集

**Advanced Tools** 与 Quick Tools 任务面相同，但你可以挑选**到底用哪个模型**（部分任务还能挑数据集）。在默认参数不够好、想横向对比模型、或者需要结构感知版本的突变 / 发现任务时使用。

## 工具一览

| 工具 | 路由 | 你能挑什么 |
| :--- | :--- | :--- |
| Directed Evolution | `/advanced-tools/directed-evolution` | Sequence-based vs Structure-based · 主干模型 |
| Sequence Design | `/advanced-tools/sequence-design` | 全部 ProteinMPNN 推理参数 |
| Protein Discovery | `/advanced-tools/protein-discovery` | VenusMine 超参 |
| Protein Function | `/advanced-tools/protein-function` | PLM + 1..N 个微调数据集 checkpoint |
| Functional Residue | `/advanced-tools/functional-residue` | 残基级预测的 PLM |

> Physicochemical Property **只在 Quick Tools** —— 这是确定性计算，没有可调参数。

---

## 通用布局

| 区域 | 内容 |
| :--- | :--- |
| **左栏 — 输入 + 配置** | 序列 / PDB 输入、模型选择器、任务 / 数据集选择器、可选 **Enable AI analysis** 开关、**Start Prediction**。 |
| **右栏 — 结果面板** | 状态、原始表、可选热图或预测图、可选 AI Expert Analysis、**Download Results**。 |

**在线模式：** Protein Discovery 只读；序列输入受在线 FASTA 上限限制。

---

## 1. Directed Evolution（定向进化）

两种预测模式，顶部切换：

### 1.1 Sequence Model

| 字段 | 说明 |
| :--- | :--- |
| **输入** | FASTA：粘贴 / 上传 / Workspace |
| **Select Model** | VenusPLM · ESM2-650M · ESM-1v |

### 1.2 Structure Model

| 字段 | 说明 |
| :--- | :--- |
| **输入** | `.pdb`：上传 / Workspace |
| **Select Model** | VenusREM · ProSST-2048 · ProtSSN · ESM-IF1 · SaProt · MIF-ST |

两种模式都有：**Enable AI analysis** 开关，**LLM Provider** 下拉（DeepSeek / ChatGPT / Gemini）。

**输出：** 排序后的突变表 + 二维预测热图（Y = 排序位置，X = 替换氨基酸），可选 AI Expert Analysis，下载。

---

## 2. Sequence Design（序列设计）

完整控制版的 ProteinMPNN。需要可复现 benchmark 或特定设计约束时使用。

| 组 | 字段 |
| :--- | :--- |
| **结构** | 上传 `.pdb`、Workspace 选择 |
| **模型 & 采样** | Model Family（Soluble / Vanilla / CA）、Model Name（默认 `v_48_020`，原生高分辨率结构用 `v_48_002`）、Omit AAs（默认 `X`）、Temperatures（如 `0.1` 或 `0.1,0.2`）、Number of sequences（默认 8，在线模式有上限）、Design Diversity |
| **链** | Designed Chains、Fixed Chains、Fixed Residues（`A12,C13` 或 `A:12,13;B:5-8`）、Homomer tying 开关 |
| **运行时** | Seed（默认 0）、Batch Size（默认 1）、Max Length（默认 200000） |
| **高级规则（文本）** | Tied Positions、Omit AA Rules、AA Bias、Bias-By-Residue、PSSM Rules —— 文本输入，后端自动转 JSONL |
| **PSSM 数值** | `pssm_multi`、`pssm_threshold`、`pssm_log_odds_flag`、`pssm_bias_flag` |

**模型选择建议：**
- 默认 `v_48_020`（自动 noise 0.20）—— 适用于 AlphaFold / AI 生成的骨架
- 原生高分辨率结构 `v_48_002`（自动 noise 0.02）

**输出：** 状态总结、FASTA 表（header / sequence / length / score）、原始 JSON、下载 FASTA。

要默认安全的一键用法，去 **Quick Tools → Sequence Design**。

---

## 3. Protein Discovery（VenusMine 蛋白发现）

完整控制 VenusMine 流水线参数（结构对齐 + 序列相似 + 冗余消除 + 表征排序 + 进化树）。

| 字段 | 默认 |
| :--- | :--- |
| **Protected Region Start / End** | 1 / 100 |
| **MMseqs Threads** | 96 |
| **MMseqs Iterations** | 3 |
| **MMseqs Max Sequences** | 100 |
| **Cluster Min Seq Identity** | 0.5 |
| **Cluster Threads** | 96 |
| **Tree Top-N Threshold** | 10 |
| **E-value Threshold** | 1e-5 |

**输入：** `.pdb`：上传 / Workspace。

**输出：** 聚类表 + 进化树 + 可下载的归档包。

**在线模式：** 所有控件禁用（只读）。

---

## 4. Protein Function（蛋白功能）

让同一个蛋白级预测在多个微调数据集间交叉验证。

| 字段 | 说明 |
| :--- | :--- |
| **输入** | FASTA：粘贴 / 上传 / Workspace |
| **Model** | PLM 主干（默认 ESM2-650M） |
| **Task** | 要预测的蛋白级属性 — 选项含义见下。 |
| **Datasets** | 多选格子，列出与所选任务相关的微调数据集（如溶解性任务下的 DeepSol + ProtSolM + eSOL） |
| **Enable AI analysis** | 可选 |

**任务选项：**

- **Solubility** — 表达后是否可溶。
- **Localization** — 亚细胞定位。
- **Metal ion binding** — 金属结合能力。
- **Stability** — 对热 / 化学变性的抵抗力。
- **Sorting signal** — 是否含信号肽 / 靶向 motif。
- **Optimum temperature** — 最大活性所需温度。

**输出：** Raw 表多一列 **Dataset**（每个数据集 × 序列一行），预测图（如亚细胞定位的柱状图），可选 AI Expert Analysis，下载。

---

## 5. Functional Residue（功能残基）

可配置 PLM 主干的残基级预测。

| 字段 | 说明 |
| :--- | :--- |
| **输入** | FASTA：粘贴 / 上传 / Workspace |
| **Model** | PLM 主干（默认 ESM2-650M） |
| **Task** | 要预测的残基级位点类型 — 选项含义见下。 |
| **Enable AI analysis** | 可选 |

**任务选项：**

- **Activity Site** — 负责催化 / 生物功能的关键残基。
- **Binding Site** — 与配体、离子或其他分子结合的关键残基。
- **Conserved Site** — 进化中高度保留的残基。
- **Motif** — 具有特定结构 / 功能作用的短模式。

**输出：** 残基级表（Position、Residue、Predicted Label、Probability）、一维预测热图、可选 AI Expert Analysis、下载。

---

## 什么时候用 Quick vs Advanced

| 如果你… | 用 |
| :--- | :--- |
| 只想要结果、最快路径 | **Quick Tools** |
| 想对同一个输入横向比 2-3 个 PLM | **Advanced Tools** |
| 需要结构感知的突变打分（PDB） | **Advanced Tools → Directed Evolution → Structure Model** |
| 需要同一任务跨数据集投票 | **Advanced Tools → Protein Function** |
| 要为 benchmark 调 ProteinMPNN | **Advanced Tools → Sequence Design** |
| 要调 VenusMine 超参 | **Advanced Tools → Protein Discovery** |
| 只想要 pI / SASA / 二级结构 | **Quick Tools → Physicochemical Property**（Advanced 没有对应） |
