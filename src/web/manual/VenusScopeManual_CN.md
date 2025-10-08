# VenusScope 功能手册 (VenusScope Function Manual)

### 1. 简介

**VenusScope** 是 VenusFactory 平台提供的一项综合性高级分析工具。它旨在通过**一键式操作**，对单个蛋白质序列执行多达四种关键的预测和分析任务，并将所有结果整合输出为一份结构清晰、内容全面的**《Comprehensive Protein Analysis Report (综合蛋白质分析报告)》**。此工具显著简化了用户的工作流程，提供了从突变优化到理化性质的全方位洞察，是蛋白质工程和基础研究的首选入口。

---

### 2. 界面概览

VenusScope 界面设计直观，主要分为数据输入、分析类型选择和报告输出三个部分。

**Data Input (数据输入)**:
- **Upload Protein File (.fasta, .fa, .pdb) / Paste Protein Content (上传蛋白质文件/粘贴蛋白质内容)**: 用户可以上传 FASTA 或 PDB 文件，或直接粘贴序列，提供待分析的蛋白质。
- **Uploaded Protein Sequence (已上传的蛋白质序列)**: 显示已输入或已上传的氨基酸序列。

**Analysis Type Selection (分析类型选择)**:
- 提供四个核心预测任务的复选框，允许用户根据需要**自由选择**要整合到报告中的分析类型。
- 选择完成后，点击 **Generate Brief Report (生成简要报告)** 按钮启动分析。

**Report Output (报告输出)**:
- 分析完成后，界面会显示报告摘要和实验建议，并提供完整的《Comprehensive Protein Analysis Report》供用户查阅。

---

### 3. 支持的分析类型

VenusScope 允许用户通过勾选复选框，定制报告中包含的预测模块，支持以下四种类型：

- **Protein Mutation Prediction (蛋白质突变预测)**: 预测单点突变对蛋白质功能（例如活性、稳定性）的影响，以指导定向进化优化。
- **Protein Function Prediction (蛋白质功能预测)**: 预测序列的整体功能和性质，例如溶解性、亚细胞定位和稳定性。
- **Functional Residue Prediction (功能残基预测)**: 识别关键功能位点，例如活性位点、结合位点和保守残基。
- **Physical & Chemical Properties (理化性质)**: 计算基于序列的基础属性，例如分子量、理论 pI 和不稳定性指数。

---

### 4. 报告内容详情

生成的《Comprehensive Protein Analysis Report》是一份结构化的文档，详细展示了所有预测结果：

- **Comprehensive Summary (综合摘要)**:
    - 位于报告顶部，提供蛋白质的基本信息（例如分子量、理论 pI），以及对整体预测结果的简要概括和评估。

- **Mutation Prediction Analysis (突变预测分析)**:
    - **Top beneficial Mutations (顶级有益突变)**: 以表格形式列出评分最高的突变体，包括 **Rank (排名)**、**Position (位置)**、**Mutation (突变)**、**Prediction Score (预测分数)** 和 **Potential Function (潜在功能)** 描述。
    - **Secondary Beneficial Mutations (次级有益突变)**: 列出排名较低的有益突变，提供额外的优化选项。
    - **Key Site Optimization (关键位点优化)**: 聚焦于关键功能位点（例如活性位点）的潜在优化突变。

- **Protein Function Analysis (蛋白质功能分析)**:
    - 以表格形式显示整体功能预测结果，包括：**Functional/Property Assessment (功能/性质评估)**、**Predicted Value/Class (预测值/类别)**、**Confidence (置信度)** 和 **Description (描述)**。
    - 报告内容涵盖：溶解性、亚细胞定位、金属离子结合、稳定性、分选信号和最适温度。

- **Functional Residue Prediction (功能残基预测)**:
    - 以列表或表格形式显示残基级别的预测结果，包括 **Binding Site Prediction (结合位点预测)**、**Functional Residues Prediction (功能残基预测)** 和 **Functional Motifs (功能基序)**。
    - 报告提供关键位点的序列位置和概率，帮助用户精确定位功能区域。

- **Physical & Chemical Properties (理化性质)**:
    - **Biophysical Characterization (生物物理特征)**: 列出计算所得的序列基础属性（例如分子量、pI、芳香性）。
    - **Stability Considerations (稳定性考量)**: 提供基于不稳定性指数的**稳定性预测结果**（例如 Predicted as unstable protein/预测为不稳定蛋白质）以及相应的实验建议。

- **Experimental Recommendations (实验建议)**:
    - 基于所有预测结果，系统提供总结性的**功能/稳定性建议**、**技术考量**和**实验方案建议**，直接指导用户的湿实验室工作。

- **Conclusion (结论)**:
    - 提供报告的最终总结，重申蛋白质的关键特性和最重要的实验优化方向。

---

### 5. User Guide (用户指南)

VenusScope 致力于提供最简洁、最高效的蛋白质综合分析体验。

- **Operation Flow (操作流程)**:
    1.  **Input Data (输入数据)**: 通过上传文件或粘贴序列，提供待分析的蛋白质序列/结构。
    2.  **Select Tasks (选择任务)**: 在 **Select Analysis Types** 区域，勾选您希望包含在报告中的一个或多个分析模块（建议全部勾选以获得最全面的洞察）。
    3.  **Generate Report (生成报告)**: 点击 **Generate Brief Report** 按钮启动分析。
    4.  **Review Results (查阅结果)**: 系统将生成《Comprehensive Protein Analysis Report》，您可以滚动查看详细分析，并在 **Experimental Recommendations** 部分找到关键结论。

- **Usage Tips (使用建议)**:
    - **Prioritize