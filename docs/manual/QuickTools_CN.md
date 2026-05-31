# Quick Tools — 零配置快速分析

**Quick Tools** 是跑单次预测最快的方式：预选模型、默认参数、最少的开关。选工具、丢数据、点 **Start Prediction**。

要完全掌控模型 / 数据集，去 **Advanced Tools**（同样的五大任务，加上更多选项）。

## 工具一览

| # | 工具 | 路由 | 输入 | 告诉你什么 |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Directed Evolution | `/quick-tools/directed-evolution` | FASTA 或 PDB | 针对目标功能的最佳单点突变 |
| 2 | Sequence Design | `/quick-tools/sequence-design` | PDB | 给定骨架的候选序列（ProteinMPNN） |
| 3 | Protein Discovery | `/quick-tools/protein-discovery` | PDB | 通过 VenusMine 找结构 / 序列同源 |
| 4 | Protein Function | `/quick-tools/protein-function` | FASTA | 蛋白级属性（溶解性、定位…） |
| 5 | Functional Residue | `/quick-tools/functional-residue` | FASTA | 残基级活性 / 结合 / 保守 / motif |
| 6 | Physicochemical Property | `/quick-tools/physicochemical-property` | FASTA 或 PDB | MW、pI、SASA、二级结构等 |

---

## 通用布局

所有 Quick Tools 页面布局一致：

| 区域 | 内容 |
| :--- | :--- |
| **左栏 — 输入 + 配置** | 序列 / PDB 输入（粘贴 / 上传 / Workspace / 用示例）、任务下拉、可选的 **Enable AI analysis** 开关（启用后会出现 LLM Provider）、**Start Prediction** 按钮。 |
| **右栏 — 结果面板** | 状态、原始结果表、可选的热图、可选的 AI Expert Analysis、**Download Results**。 |

**在线模式：** 序列输入受在线 FASTA 上限限制（默认粘贴 50 残基）。Protein Discovery 在在线模式下是**只读**的。

---

## 1. Directed Evolution（定向进化）

针对目标功能给所有单点突变打分。

| 字段 | 说明 |
| :--- | :--- |
| **序列输入** | 粘贴 FASTA、上传 `.fasta` / `.fa` / `.pdb`、从 Workspace 选、或用示例。 |
| **Select Protein Function** | 突变打分的目标功能（见下）。 |
| **Enable AI analysis** | 可选；启用后多一个 AI Expert 总结标签。 |

**功能选项：**

- **Activity** — 突变对催化或生物活性的影响。
- **Binding** — 突变对蛋白结合配体或互作伙伴能力的影响。
- **Expression** — 突变对宿主细胞中表达量的影响。
- **Organismal Fitness** — 突变对整个生物体存活 / 生长能力的影响。
- **Stability** — 突变对热力学或构象稳定性的影响。

**输出：**
- **Raw 表** — 按预测分排序的突变
- **Prediction heatmap** — 二维矩阵：Y = 排序后的位置，X = 替换氨基酸；颜色越深 = 增益越强
- **AI Expert Analysis**（启用时）— 自然语言解读
- **Download Results**

---

## 2. Sequence Design（序列设计）

基于结构用 ProteinMPNN 生成候选序列，默认参数对生物用户友好。

| 字段 | 说明 |
| :--- | :--- |
| **PDB 输入** | 上传 `.pdb`、从 Workspace 选、或用示例。 |
| **Model Family** | Soluble（默认 — 大多数场景推荐）、Vanilla（膜蛋白）、CA（仅 Cα 粗粒度）。 |
| **Designed Chains** | 可选，例如 `A` 或 `A,B`。留空 = 设计所有链。 |
| **Fixed Residues** | 可选固定位点语法，例如 `A12,C13` 或 `A:12,13;B:5-8`。 |
| **Number of sequences** | 4 / 8 / 16 / 32（在线模式受上限限制）。 |
| **Design Diversity** | Low / Medium / High（映射到 ProteinMPNN 采样温度）。 |
| **Enable AI analysis** | 可选。 |

默认底层用 `v_48_020` + `backbone_noise=0.20`，适用于 AlphaFold 风格的骨架和常规重设计。

**输出：**
- **Table** — 生成的序列，含 header、长度、score
- **Raw** — 完整 JSON
- **AI Expert**（启用时）
- **Download Result** — FASTA 文件

需要更细的 ProteinMPNN 调参？去 **Advanced Tools → Sequence Design**。

---

## 3. Protein Discovery（蛋白发现）

一键 VenusMine 流水线，做结构同源搜索 + 聚类。

| 字段 | 说明 |
| :--- | :--- |
| **PDB 输入** | 上传 `.pdb`、从 Workspace 选、或用示例。 |
| **高级参数** | Quick 模式不暴露 — 使用后端默认值。 |

点开始按钮等结果。输出与 Advanced 后端产物兼容（tree / labels / archive 下载字段）。

**在线模式：** 整张表单只读。

需要调参（protected region、MMseqs 线程 / 迭代、聚类相似度、e-value）请去 **Advanced Tools → Protein Discovery**。

---

## 4. Protein Function（蛋白功能）

从 FASTA 序列预测蛋白级属性。

| 字段 | 说明 |
| :--- | :--- |
| **序列输入** | 粘贴 / 上传 FASTA、从 Workspace 选、或用示例。 |
| **Select Task** | 要预测的属性（见下）。 |
| **Enable AI analysis** | 可选。 |

**任务选项：**

- **Solubility** — 表达后是否可能可溶（对纯化关键）。
- **Localization** — 在细胞中的最终位置（核 / 胞质 / 线粒体 / …）。
- **Metal ion binding** — 是否能结合特定金属离子。
- **Stability** — 对热或化学变性的内在稳定性。
- **Sorting signal** — 是否含信号肽，将蛋白引导到特定细胞器 / 分泌途径。
- **Optimum temperature** — 蛋白发挥最大功能活性所需的温度范围。

**输出：**
- **Raw 表** — 蛋白名、序列、预测类别、置信度（0–1）
- **AI Expert Analysis**（启用时）
- **Download Results**

运行后状态提示： *"All predictions completed. Results were aggregated using soft voting."*

---

## 5. Functional Residue（功能残基）

沿序列预测残基级功能位点。

| 字段 | 说明 |
| :--- | :--- |
| **序列输入** | 粘贴 / 上传 FASTA、从 Workspace 选、或用示例。 |
| **Select Task** | 要预测的残基级位点类型（见下）。 |
| **Enable AI analysis** | 可选。 |

**任务选项：**

- **Activity Site** — 负责催化 / 生物功能的关键残基。
- **Binding Site** — 与配体、离子或其他分子结合的关键残基。
- **Conserved Site** — 进化中高度保留的残基；通常对结构或功能关键。
- **Motif** — 序列中形成特定结构 / 功能特征的短氨基酸模式。

**输出：**
- **Raw 表** — Position、Residue、Predicted Label (0/1)、Probability (0–1)
- **Prediction heatmap** — 沿残基轴的一维概率分布带
- **AI Expert Analysis**（启用时）
- **Download Results**

---

## 6. Physicochemical Property（理化性质）

计算生物物理性质——部分仅需序列，部分仅需结构。

| 性质 | 需要的输入 |
| :--- | :--- |
| **Physical and chemical properties** | FASTA — MW、pI、芳香性、不稳定指数、GRAVY、预测二级结构组成 |
| **Relative solvent accessible surface area** | PDB — 每残基相对溶剂可及表面积 |
| **SASA value** | PDB — 总 SASA（Å²） |
| **Secondary structure** | PDB — 每残基 DSSP 编码（H、E…） |

选择仅需 PDB 的任务、且上传了多链 `.pdb` 时，会多出一个 **PDB Chain** 选择器。仅 PDB 任务下粘贴文本框会被禁用。

**输出：** 任务相关的表格；本模块**没有** AI Expert 标签。用 **Download Results** 导出。

---

## 小贴士

- **PDB-only 任务必须传 PDB。** 别把 FASTA 丢进 SASA / 二级结构任务，跑不出结果。
- **传大文件前先用示例验证。** 用 example 按钮 sanity check。
- **盯住状态胶囊。** 多数任务秒级返回，但 Directed Evolution 时间正比于 `序列长度 × 20`。
- **AI 总结可选关掉。** 只要原始数字、想要更快出结果时可以不勾。
