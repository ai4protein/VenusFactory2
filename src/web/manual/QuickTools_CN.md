# VenusFactory Quick Tools 用户指南

## 1. 简介

Quick Tools (快捷工具) 专为需要快速、高效、无需复杂配置的分析用户设计。它提供了进入 **4 个关键预测任务**的无代码入口：

* **Intelligent Directed Evolution (智能定向进化):** 基于 AI 的突变预测。此工具可快速评估蛋白质突变的效果，提供对单点或多点突变影响的洞察。
* **Protein Function Prediction (蛋白质功能预测):** 利用预训练模型，预测给定氨基酸序列的多种功能属性，如溶解性、亚细胞定位等。
* **Functional Residue Prediction (功能残基预测):** 专注于识别单个氨基酸残基的功能，如活性位点、结合位点和基序，帮助精确定位结构-功能关系。
* **Physicochemical Property Analysis (理化性质分析):** 执行基于氨基酸组成的快速、确定性计算，以确定理论 pI、分子量、不稳定性指数和二级结构含量等基本性质。

---

## 2. Quick Tools 界面概览

### 2.1 模型配置与数据输入区域

该区域用于选择预测任务并提供蛋白质序列。

**Model Configuration (模型配置):**

* **Select Task / Select Protein function / Select Properties of Protein** (具体选项由工具功能决定): 这是一个下拉菜单，用于选择当前运行的预测任务类型。
    * *Intelligent directed evolution*: **Select Protein function**
    * *Protein function prediction*: **Select Task**
    * *Functional residual prediction*: **Select Task**
    * *Physicochemical property analysis*: **Select Properties of Protein**

**Data Input (数据输入):** 该区域允许用户通过文件上传或粘贴提供序列数据，并显示已上传序列的内容。

* **Upload Protein File** (Upload FASTA File): 选中的标签页，用于通过文件上传序列。
* **Paste Protein Content** (Paste FASTA Content): 替代标签页，用于直接粘贴 FASTA 格式的序列内容。
* **Uploaded Protein Sequence**: 此区域将显示从输入文件或粘贴内容中读取的**原始氨基酸序列**。

**Configure AI Analysis (Optional) (配置AI分析 - 可选):**

* **AI Settings**: 用于选择或配置 AI 分析的具体参数。
* **Enable AI Summary**: 激活此选项后，系统将在预测完成后生成**文本总结和分析**。
* **Start Prediction**: 启动预测任务。

### 2.2 结果展示功能

该模块显示任务完成后的预测结果，并提供结果解读和下载选项。

* **Status**: 提供**实时反馈**的预测进度。
* **Result Display Tabs (结果展示标签页):** 结果以表格形式展示，并分为：
    * **Raw Results**: 生成与任务对应的初始结果。
    * **AI Expert Analysis**: 如果 **Enable AI Summary** 被启用，此标签页将显示由 AI 生成的**文本分析报告**。
    * **Prediction heatmap**: **Intelligent directed evolution** 和 **Functional residual prediction** 任务会提供此热图。
* **Download Results**: 允许用户下载表格中的所有预测结果。

---

## 3. Intelligent Directed Evolution (智能定向进化)

### 3.1 序列输入与蛋白质功能选择

**Sequence Input Method (序列输入方法):** 用户可通过两种方式提供蛋白质序列：
* **Upload Protein File:** 支持 **.fasta, .fa, 或 .pdb** 格式的文件。
* **Paste Protein Content:** 直接将序列输入 **FASTA format** 文本框。

**Select Protein Function (选择蛋白质功能):** 用于选择定向进化预测的**目标功能**。
* **Activity (活性):** 预测突变对蛋白质**催化或生物活性**的影响。
* **Binding (结合):** 预测突变对蛋白质**结合配体或相互作用伙伴**能力的影响。
* **Expression (表达):** 预测突变对蛋白质在宿主细胞中**表达水平**的影响。
* **Organismal Fitness (有机体适应性):** 预测突变对整个有机体**生存或生长能力**的影响。
* **Stability (稳定性):** 预测突变对蛋白质**热力学或构象稳定性**的影响。

### 3.2 执行预测

1.  **Ensure all model configuration parameters are set correctly (确保所有模型配置参数设置正确)**。
2.  Click the **"Start Prediction"** button to start the prediction process (点击“Start Prediction”按钮开始预测流程)。
3.  The system will display **prediction progress and status information** (系统将显示预测进度和状态信息)。
4.  To abort the prediction, click the **"Abort"** button (如需中止预测，点击“Abort”按钮)。

### 3.3 结果展示功能

**Raw Results (原始结果):** 清晰列出**最优突变体**及其**性能指标**，包括 **Mutant** 名称、**Prediction Rank** 和 **Prediction Score**。

**Prediction Heatmap (预测热图):** 以**二维矩阵**形式展示每个残基位置突变为任何其他氨基酸的**潜在影响**。
* **Y-axis** 显示**残基位置**；**X-axis** 显示**突变的氨基酸类型**。
* **颜色强度**代表突变对目标功能的**归一化影响**。**深色 (High)** 表示有**更强的正面增强效果**。

**AI Expert Analysis:** 由专业 **AI biology expert** 提供深入解读和实验建议。

**Download Results:** 允许用户下载所有详细预测数据。

---

## 4. Protein Function Prediction (蛋白质功能预测)

### 4.1 任务选择与序列输入

**Model Configuration (模型配置):** **Select Task** 下拉菜单用于选择具体预测目标：
* **Solubility (溶解性):** 预测蛋白质表达后的**可溶性**。
* **Localization (亚细胞定位):** 预测蛋白质在细胞内的**最终位置**。
* **Metal ion binding (金属离子结合):** 预测蛋白质**结合特定金属离子**的能力。
* **Stability (稳定性):** 预测蛋白质的**固有稳定性**。
* **Sorting signal (分选信号):** 预测是否存在**信号肽**。
* **Optimum temperature (最适温度):** 预测蛋白质发挥功能所需的**温度范围**。

**Sequence Input Method:** 用户通过 **Upload FASTA File** 或 **Paste FASTA Content** 提供序列。

### 4.2 执行预测

1.  **Ensure all model configuration parameters are set correctly (确保所有模型配置参数设置正确)**。
2.  Click the **"Start Prediction"** button to start the prediction process (点击“Start Prediction”按钮开始预测流程)。
3.  The system will display **prediction progress and status information** (系统将显示预测进度和状态信息)。
4.  To abort the prediction, click the **"Abort"** button (如需中止预测，点击“Abort”按钮)。

### 4.3 结果展示功能

**Status:** 提供实时反馈 (例如: **"All predictions completed! Results were aggregated using soft voting."**)。

**Raw Results:** 显示单序列预测的核心输出，包括 **Protein Name**、**Sequence**、**Predicted Class** (例如 **Soluble**) 和 **Confidence Score** (0 到 1 之间的值)。

**AI Expert Analysis:** 由专业 **AI biology expert** 提供解读和实验建议。

**Download Results:** 允许用户下载所有详细预测数据。

---

## 5. Functional Residue Prediction (功能残基预测)

### 5.1 任务选择与序列输入

**Model Configuration (模型配置):** **Select Task** 用于选择要预测的功能残基类型：
* **Activity Site (活性位点):** 预测负责**催化或生物功能**的关键残基位置。
* **Binding Site (结合位点):** 预测蛋白质**结合配体、离子**的关键残基位置。
* **Conserved Site (保守位点):** 预测在进化中**高度保留**的残基位置。
* **Motif (基序):** 预测形成特定结构或功能特征的**短氨基酸模式**。

**Sequence Input Method:** 用户通过 **Upload FASTA File** 或 **Paste FASTA Content** 提供序列。

### 5.2 执行预测

1.  **Ensure all model configuration parameters are set correctly (确保所有模型配置参数设置正确)**。
2.  Click the **"Start Prediction"** button to start the prediction process (点击“Start Prediction”按钮开始预测流程)。
3.  The system will display **prediction progress and status information** (系统将显示预测进度和状态信息)。
4.  To abort the prediction, click the **"Abort"** button (如需中止预测，点击“Abort”按钮)。

### 5.3 结果展示功能

**Raw Results:** 显示**逐残基的预测结果**。
* **表格内容:** **Position** (位置)、**Residue** (残基)、**Predicted Label** (预测标签，**1** 为目标残基)、**Probability** (置信度得分)。

**Prediction Heatmap:** 此热图以**线性条形图**形式可视化整个序列的**概率分布**。
* **图表类型:** **一维条形图**，**X-axis** 为 **Residue Position**。
* **信息展示:** **颜色或高度变化**直观映射了残基被预测为目标位点的**概率**。

**AI Expert Analysis:** 由专业 **AI biology expert** 提供解读和实验建议。

**Download Results:** 允许用户下载所有详细预测数据。

---

## 6. Physicochemical Property Analysis (理化性质分析)

### 6.1 任务选择与序列输入

**Task Configuration (任务配置):** **Select Properties of Protein** 用于选择分析类型：
* **Relative solvent accessible surface area (PDB only):** 计算每个残基的相对表面积。**注意:** 需要 **PDB structure file**。
* **SASA value (PDB only):** 计算整个蛋白质的总 **SASA value**。**注意:** 需要 **PDB structure file**。
* **Physical and chemical properties:** 计算**序列基础属性**。**注意:** 接受 **FASTA sequence file**。
* **Secondary structure (PDB only):** 提取二级结构信息。**注意:** 需要 **PDB structure file**。

**Data Input:** 需要 PDB 文件的任务必须上传 **.pdb** 文件。

### 6.2 执行预测

1.  **Ensure all model configuration parameters are set correctly (确保所有模型配置参数设置正确)**。
2.  Click the **"Start Prediction"** button to start the prediction process (点击“Start Prediction”按钮开始预测流程)。
3.  The system will display **prediction progress and status information** (系统将显示预测进度和状态信息)。
4.  To abort the prediction, click the **"Abort"** button (如需中止预测，点击“Abort”按钮)。

### 6.3 结果展示功能

**Status:** 提供实时反馈。

**Raw Results:** 根据任务输出相应数据：
* **Physical and chemical properties:** 显示**Sequence length**、**Molecular weight**、**Theoretical pI**、**Instability index** 等属性。
* **Secondary structure:** 显示每个残基的 **DSSP secondary structure code**。

**Download Results:** 允许用户下载所有详细预测数据。