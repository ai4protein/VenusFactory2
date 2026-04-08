# VenusFactory2 Advanced Tools 用户指南

## 1. Advanced Tools 简介

**Advanced Tools (高级工具)** 让您对蛋白质预测有更多控制权。Quick Tools 为了速度使用预选模型，而 Advanced Tools 让您准确选择使用哪些模型，并比较不同方法的结果。

**在以下情况使用 Advanced Tools**：
- 想比较不同模型的预测结果（ESM2、ProtBert、ProtT5 等）
- 除了基于序列的分析外，还需要基于结构的分析
- 正在为特定用例优化，想测试不同的模型配置
- 想通过交叉检查多个数据集来验证结果

权衡：更多灵活性意味着更多选择，但您会获得更定制化、可能更准确的结果。

### 1.1 智能定向进化

该模块利用 AI **精确预测突变效果**，以指导蛋白质的定向进化实验。用户可根据数据和所需精度选择不同的模型类型。

**核心优势：** 用户可选择底层 **PLM**（例如 **ESM2**、**VenusPLM**）并决定是否整合**结构信息**。

* **Sequence-based (基于序列)**:
    * **描述:** 利用用户选择的 PLM，直接从氨基酸序列中预测突变对蛋白质功能的影响。
    * **应用场景:** 适用于**缺乏结构信息**或需要**大规模、高通量筛选**的场景。
* **Structure-based (基于结构)**:
    * **描述:** 将上传的 **PDB 结构文件**与基于结构的算法结合，对突变效果进行更精细的评估。
    * **应用场景:** 适用于需要高精度分析突变对**残基相互作用**和**结构稳定性**影响的理性设计。

### 1.2 蛋白质功能预测

该功能允许用户利用不同的 PLM 模型和多个数据集对蛋白质的整体性质进行预测，旨在获得更可靠的结果。

**核心优势：** 用户可自由选择 **PLM**（例如 **ESM2-650M**），并能同时选择**多个已微调的数据集模型**（例如 **DeepSol, ProtSolM**）进行预测。

* **描述:** 用户通过选择不同的 PLM 和数据集，预测给定氨基酸序列的整体功能或性质（例如溶解性、定位）。
* **应用场景:** **多模型交叉验证**是其关键优势。通过比较不同模型在同一任务上的结果，显著**提高预测可靠性和鲁棒性**。

### 1.3 功能残基预测

该模块允许用户选择特定的 PLM 模型，对序列中的关键功能残基进行高精度定位和预测。

**核心优势：** 用户可自由选择底层 **PLM**，以专门**优化残基级别预测性能**。

* **描述:** 专注于识别单个氨基酸残基的功能角色，例如**活性位点**或**结合位点**。
* **应用场景:** 通过选择最优 PLM，用户可以**准确标记功能位点**并获得**残基级别预测概率**，有助于建立精确的结构-功能图谱。

### 1.4 蛋白质酶挖掘
该模块基于用户提供的蛋白质结构或序列信息，在大规模蛋白质数据库中进行功能相似酶的系统性挖掘与筛选，用于发现潜在同源或类功能酶分子。

**核心优势：** 用户可以自己设置 **实验参数**，一键进行蛋白质酶挖掘。

* **描述：** VenusMine通过结合结构比对、序列相似性搜索与功能特征匹配，实现多层级的酶筛选。
* **应用场景：** 酶功能发现，从未知结构中挖掘潜在功能相似酶；酶工程设计，为蛋白质定向进化提供候选模版；
---


## 2. Advanced Tools 配置与结果

### 2.1 模型配置与数据输入区域

该区域用于定义预测任务、选择底层模型和数据集，并提供待分析的蛋白质序列。

**Model Configuration (模型配置):**

* **Select Sequence-based Model / Select Structure-based Model:** 这是 Advanced Tools 的关键配置。用户可**自由选择底层 PLM**（例如 **VenusPLM, ESM2-650M**）或结构模型进行推理，而非使用预设模型。
* **Select Task / Select Protein function / Select Properties of Protein** (具体选项由工具功能决定): 这是一个下拉菜单，用于选择预测任务类型：
    * directed mutation: **Select Protein function (选择蛋白质功能)**
    * Protein function prediction: **Select Task (选择任务)**
    * Functional residual prediction: **Select Task (选择任务)**
* **Select Datasets:** 在特定任务中，用户可同时选择**多个已微调的数据集模型**（例如 **DeepSol, ProtSolM**）进行预测，启用多模型交叉验证。

**Data Input (数据输入):** 该区域允许用户通过文件上传或粘贴提供序列数据，并显示已上传序列的内容。

* **Upload Protein File:** 选中的标签页，用于通过文件上传序列/结构文件（根据任务可能支持 **.pdb**）。
* **Paste Protein Content:** 替代标签页，用于直接在文本框中粘贴 **FASTA 格式**的序列内容。
* **Uploaded Protein Sequence:** 此区域显示从输入文件或粘贴内容中读取的**原始氨基酸序列**。

**Configure AI Analysis (Optional) (配置AI分析 - 可选):**

* **AI Settings:** 用于选择或配置 AI 分析的具体参数。
* **Enable AI Summary:** 激活此选项后，系统将在预测完成后生成**文本总结和分析**。
* **Start Prediction:** 启动预测任务。

### 2.2 结果展示功能

该模块显示任务完成后的预测结果，并提供结果解读和下载选项。

**Status (状态):** 提供**实时反馈**。由于模型配置灵活，此区域也可能显示详细的诊断和警告信息。

**Result Display Tabs (结果展示标签页):** 结果以表格形式展示，并分为以下部分：

* **Raw Results (原始结果):**
    * 由于 Advanced Tools 允许选择多个数据集，结果表格可能包含额外的 **Dataset (数据集)** 列，展示不同模型在同一序列上的预测结果。
    * 根据任务和模型配置，生成相应的初始预测结果。
* **Prediction Plots (预测图):** 由**蛋白质功能预测**任务提供。它以可视化方式显示预测结果（例如，用条形图显示亚细胞定位的预测概率）。
* **AI Expert Analysis (AI 专家分析):** 如果 **Enable AI Summary** 被启用，此标签页显示由 AI 生成的**文本分析报告**。
* **Prediction heatmap (预测热图):** **智能定向进化**和**功能残基预测**任务提供此图表，用于可视化残基级或突变效应预测结果。
* **Download Results (下载结果):** 允许用户将所有详细预测数据下载到本地计算机。

---

## 3. Advanced Tools - 基于序列的模型

### 3.1 智能定向进化

该模块允许用户利用**自由选择的蛋白质语言模型 (PLM)** 精确预测突变对蛋白质功能的影响，以实现定制化的定向进化指导。

#### 3.1.1 任务配置与数据输入

**Model Configuration (模型配置):**

* **Select Sequence-based Model (选择基于序列的模型):** 这是核心。用户可自由选择不同的底层 PLM 引擎作为预测核心。**可选模型**包括但不限于：**VenusPLM, ESM2-650M, ESM-1v, or ESM-1b**。用户可根据性能或偏好选择。

**Data Input (数据输入):** 该区域允许用户提供序列数据。

* **Upload Protein File:** 选中的标签页，用于通过文件上传序列，支持 **.fasta, .fa, or .pdb** 等格式。
* **Paste Protein Content:** 替代标签页，用于直接在文本框中输入 **FASTA 格式**的序列内容。
* **Uploaded Protein Sequence:** 此区域显示从输入文件或粘贴内容中读取的**原始氨基酸序列**。

**Configure AI Analysis (Optional) (配置AI分析 - 可选):**

* 用户可选择启用 **Enable AI Summary**，在预测完成后获得专业 **AI 生物专家**对结果的**文本评估**。

#### 3.1.2 执行预测

1.  **Ensure all model configuration parameters are set correctly (确保所有模型配置参数设置正确)**。
2.  Click the **"Start Prediction"** button to start the prediction process (点击“Start Prediction”按钮开始预测流程)。
3.  The system will display **prediction progress and status information** (系统将显示预测进度和状态信息)。
4.  To abort the prediction, click the **"Abort"** button (如需中止预测，点击“Abort”按钮)。

#### 3.1.3 结果展示功能

**Status (状态):** 提供**实时反馈** (例如：“Prediction completed successfully!”)。

**Raw Results (原始结果):** 该表格清晰列出模型预测的**最优突变体**及其性能指标。表格通常包括 **Mutant (突变体)** 名称、**Prediction Rank (预测排名)** 和 **Prediction Score (预测分数)**。用户可根据此分数选择高排名的突变体进行实验验证。

**Prediction Heatmap (预测热图):** 此可视化以**二维矩阵**形式展示突变对每个残基位置的**潜在影响**。

* **Y轴** 显示**残基位置**；**X轴** 显示**突变的氨基酸类型**。
* **颜色强度**代表**归一化影响**。**深色 (High/高)** 表示**更强的正面增强效果**。**推理基于用户选择的 PLM 模型**，有助于快速识别**“热点”残基**。

**AI Expert Analysis (AI 专家分析):** 专业 **AI 生物专家**评估结果，提供**文本形式**的深入解读和实验建议。

**Download Results (下载结果):** 允许用户下载所有详细预测数据。

---

## 4. Sequence Design

该模块在 Advanced Tools 中明确提供 **ProteinMPNN**，并开放完整推理参数，支持结构条件序列设计。

### 4.1 输入与核心配置

* **结构输入：** 上传 `.pdb` 文件（或使用示例 PDB）。
* **Model Family 选择：** `Soluble` / `Vanilla` / `CA`（手动选择 + 推荐引导）。
  * **Soluble：** 蛋白发现与常规序列设计推荐。
  * **Vanilla：** 膜蛋白设计推荐。
  * **CA：** 仅用于只有 Cα 粗粒化坐标的场景。
* **Designed Chains / Fixed Chains：** 分别定义要设计的链和作为上下文固定的链。
* **Fixed Residues（文本语法）：** 使用可读语法指定固定位点，例如 `A12,C13` 或 `A:12,13;B:5-8`。
* **Homomer 模式：** 为同源多聚体启用 tied-position 约束。
* **Number of Sequences / Temperatures：** 设置生成数量与采样多样性。

### 4.2 ProteinMPNN 全参数控制

Advanced Sequence Design 页面提供完整参数入口，包含：

* **模型与采样：** `model_name`, `omit_aas`（`backbone_noise` 由模型自动映射）
* **运行控制：** `seed`, `batch_size`, `max_length`
* **可选高级规则（文本输入）：** tied positions、omit-AA、AA bias、bias-by-residue、PSSM 规则，后端自动转换为 JSONL
* **PSSM 数值控制：** `pssm_multi`, `pssm_threshold`, `pssm_log_odds_flag`, `pssm_bias_flag`

模型推荐：
* **常规首选：** `v_48_020`（自动使用 0.20A 噪音策略）
* **仅高分辨率天然结构：** `v_48_002`（自动使用 0.02A 噪音策略）

### 4.3 结果输出

* **Summary：** 运行状态与完成信息。
* **Table：** FASTA 预览（header、sequence、length，以及可用时的 score 字段）。
* **Raw：** 完整 JSON 结果，便于自动化处理。
* **Download Result：** 下载 ProteinMPNN 生成的 FASTA 文件。

### 4.4 使用建议

当你需要对 ProteinMPNN 推理行为进行精细控制（例如方法学对比、参数扫描、可复现实验流程）时，请使用该模块。  
如果你更希望快速得到默认配置结果，请使用 **Quick Tools / Sequence Design**。

---

## 5. Advanced Tools - 蛋白质功能预测

该模块允许用户**自由选择底层 PLM 模型**并同时选择**多个已微调数据集**，以执行多维度预测，获取更可靠的结果。

### 5.1 任务配置与序列输入

**Model Configuration (模型配置):**

* **Select Model (选择模型):** 这是一个关键配置项，用户可自由选择底层 PLM 引擎。
    * **可选模型:** **ESM2-650M, Ankh-large, ProtBert, ProtT5-xl-uniref50**。
* **Select Task (选择任务):** 用户必须选择具体的预测目标。
    * **可选任务:** **Solubility (溶解性), Localization (亚细胞定位), Metal ion binding (金属离子结合), Stability (稳定性), Sorting signal (分选信号), Optimum temperature (最适温度)**。
* **Select Datasets (选择数据集):** 用户可同时选择多个已微调的数据集模型进行预测。
    * **可选数据集:** **DeepSol, DeepSoluE, ProtSolM**。

**Data Input (数据输入):**

* **Upload FASTA File:** 选中的标签页，支持 **FASTA 格式**文件上传。
* **Paste FASTA Content:** 替代标签页，用于直接输入序列。
* **Uploaded Protein Sequence:** 显示原始氨基酸序列。

**Configure AI Analysis (Optional) (配置AI分析 - 可选):**

* 用户可选择启用 **Enable AI Summary**，在预测完成后获得专业 **AI 生物专家**的文本评估。

### 5.2 执行预测

1.  **Ensure all model configuration parameters are set correctly (确保所有模型配置参数设置正确)**。
2.  Click the **"Start Prediction"** button to start the prediction process (点击“Start Prediction”按钮开始预测流程)。
3.  The system will display **prediction progress and status information** (系统将显示预测进度和状态信息)。
4.  To abort the prediction, click the **"Abort"** button (如需中止预测，点击“Abort”按钮)。

### 5.3 结果展示功能

**Status (状态):** 提供实时反馈 (例如：“All predictions completed!”)。

**Raw Results (原始结果):** 该表格展示**多模型/多数据集预测**的核心输出。

* **表格内容:** **Dataset (数据集), Protein Name (蛋白质名称), Sequence (序列), Predicted Class (预测类别), Confidence Score (置信度得分)**。

**Prediction Plots (预测图):** 提供预测结果的**可视化图表**。

**AI Expert Analysis (AI 专家分析):** 专业 **AI 生物专家**评估结果，提供**文本形式**的解读和建议。

**Download Results (下载结果):** 允许用户下载所有详细预测数据。

## 6. Protein Discovery (VenusMine)

VenusMine 用于从一个已知蛋白结构出发，在大规模结构与序列数据库中发现潜在同源、相似结构或功能相关的候选蛋白。该模块适合用于酶挖掘、功能相似蛋白发现、候选模板筛选，以及为后续定向进化或序列设计提供起始序列集合。

### 6.1 输入与核心配置

**Input Configuration（输入配置）：**

* **Upload PDB Structure：** 上传待分析的 `.pdb` 结构文件。VenusMine 以结构为起点，先通过 FoldSeek 执行结构相似性搜索，再进入序列层面的扩展筛选。
* **Protected Region（保护区域）：** 设置需要保留或重点关注的结构/功能片段位置。
  * **Start Position：** 保护区域起始残基位置，默认 `1`。
  * **End Position：** 保护区域结束残基位置，默认 `100`。
  * 系统会优先保留 FoldSeek 命中中覆盖该区域的序列，因此该参数适合用于保护活性口袋、结合片段或关键结构域。

**MMseqs2 Search Parameters（MMseqs2 搜索参数）：**

* **Threads：** MMseqs2 搜索使用的线程数，默认 `96`。如果机器资源较少，可适当降低。
* **Iterations：** 迭代搜索轮数，默认 `3`。更高的迭代数可能发现更远缘的序列，但耗时更长。
* **Max Sequences：** 每轮最多保留的命中序列数量，默认 `100`，可在 `100-5000` 范围内调整。
* **Database：** 默认数据库为 **UniRef90**。如果没有显式配置数据库路径，系统会尝试自动下载到 `data/VenusMine/uniref90.fasta.gz`。


**Clustering Parameters（聚类参数）：**

* **Min Sequence Identity：** MMseqs2 聚类的最低序列一致性阈值，默认 `0.5`。数值越高，聚类越严格，冗余去除越保守。
* **Threads：** 聚类阶段使用的线程数，默认 `96`。

**Tree Building Parameters（进化树构建参数）：**

* **Top N Results：** 最终用于树构建和展示的候选序列数量，默认 `10`。如果想查看更大范围的候选序列，可提高该值。
* **E-value Threshold：** 用于过滤序列搜索结果的 E-value 阈值，默认 `1e-5`。阈值越小，结果越严格。


### 6.2 结果展示

**Structure Visualization（结构可视化）：** 显示上传结构的占位或结构查看区域，帮助确认当前分析对象。

**Phylogenetic Tree（系统发育树）：** 展示候选蛋白与参考序列之间的层次聚类关系。用户可以通过该图快速观察候选序列是否形成明显分支、是否靠近参考功能序列，以及哪些序列可能值得优先验证。

**Sequence Labels（序列标签）：** 以表格形式列出发现的序列、来源类型、标签和聚类信息。该表格适合用于后续人工筛选、候选序列整理或导出到其他分析流程。

**Complete Results（完整结果）：** 提供 ZIP 下载，包含本次 VenusMine 运行产生的主要文件，例如树图、标签表、中间 FASTA、搜索结果和日志等。

**Processing Log（处理日志）：** 实时显示 VenusMine 各步骤状态、数据库路径、命中数量、聚类结果和错误信息。若运行失败，应优先查看此处定位问题，例如 MMseqs2 未安装、UniRef90 数据库缺失、FoldSeek 下载失败或保护区域没有匹配序列。


---

## 7. Functional Residue

该模块允许用户**自由选择底层 PLM 模型**，以高精度定位和预测蛋白质序列中的关键功能残基。

### 7.1 任务配置与序列输入

**Model Configuration (模型配置):**

* **Select Model (选择模型):** 用户可自由选择底层 PLM 引擎。
    * **可选模型:** **ESM2-650M, Ankh-large, ProtT5-xl-uniref50**。
* **Select Task (选择任务):** 用户必须选择具体的预测目标。
    * **Activity Site (活性位点), Binding Site (结合位点), Conserved Site (保守位点), Motif (基序)**。

**Data Input (数据输入):**

* **Upload FASTA File:** 选中的标签页，支持 **FASTA 格式**文件上传。
* **Paste FASTA Content:** 替代标签页，用于直接输入序列。
* **Uploaded Protein Sequence:** 显示原始氨基酸序列。

**Configure AI Analysis (Optional) (配置AI分析 - 可选):**

* 用户可选择启用 **Enable AI Summary**，在预测完成后获得专业 **AI 生物专家**的文本评估。

### 7.2 执行预测

1.  **Ensure all model configuration parameters are set correctly (确保所有模型配置参数设置正确)**。
2.  Click the **"Start Prediction"** button to start the prediction process (点击“Start Prediction”按钮开始预测流程)。
3.  The system will display **prediction progress and status information** (系统将显示预测进度和状态信息)。
4.  To abort the prediction, click the **"Abort"** button (如需中止预测，点击“Abort”按钮)。

### 7.3 结果展示功能

**Status (状态):** 提供实时反馈。

**Raw Results (原始结果):** 该表格展示**逐残基的预测结果**。

* **表格内容:** **Position (位置), Residue (残基), Predicted Label (预测标签), Probability (概率)**。

**Prediction Heatmap (预测热图):** 以**一维条形图**形式可视化残基预测的概率分布。

* **图表类型:** **一维条形图**，横轴为**残基位置**。
* **信息展示:** **颜色或高度变化**映射了每个残基位置被预测为目标位点的**概率或强度**。**推理基于用户选择的 PLM 模型**。

**AI Expert Analysis (AI 专家分析):** 专业 **AI 生物专家**评估结果，提供**文本形式**的解读和建议。

**Download Results (下载结果):** 允许用户下载所有详细预测数据。
