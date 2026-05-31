# Report — 一键综合蛋白质分析

**Report** 标签页能为同一个蛋白同时生成突变、功能、残基、理化四类分析，并在原始结果之上自动写一段 AI 评述，输出一份完整文档。

## 1. 什么时候用

- 拿到新蛋白，想要一份不用挨个配置工具的整体概览。
- 需要把分析结果（HTML + PDF）发给合作者。
- 想在同一份文档里同时看到突变热点和功能上下文。

如果想要 Agent 主导的分步骤分析，去 **Agent / Chat**。

---

## 2. 界面布局

页面分三栏：

| 栏 | 作用 |
| :--- | :--- |
| **左栏 — 输入与控制** | 选择输入模式（Paste / Upload）、选链或序列、勾选要跑的分析、点 **Generate Report**，并显示进度百分比。 |
| **中栏 — AI Expert Analysis** | 任务完成后渲染 AI 评述，下方提供 **Download HTML** / **Download PDF** 链接。 |
| **右栏 — Streaming Logs** | 后端流水线的实时日志。 |

顶部状态胶囊显示当前阶段：*Ready / Processing Input / Generating Report*。

---

## 3. 输入

| 输入模式 | 用法 |
| :--- | :--- |
| **Paste** | 在文本框里粘贴一段序列或 FASTA。 |
| **Upload** | 拖入 `.fasta` / `.fa` / `.pdb` 文件，从 **Workspace** 选一个，或者点 **Use Default Example** 加载示例 FASTA。 |

解析完成后：

- 如果输入包含多条链 / 记录，会出现**链 / 序列选择器**。
- 显示所选序列的短预览。

> **在线模式：** 上传受限，请用 Paste 并控制在 FASTA 字符上限以内。

---

## 4. 选择分析项

至少勾选四项中的一项：

| 图标 | 分析 | 背后技术 |
| :---: | :--- | :--- |
| 🧬 | **Mutation** | 饱和突变打分（ESM-2、ProSST、ProtSSN…） |
| 🔬 | **Function** | 微调预测器（溶解性、定位、稳定性、最适温度、信号肽、金属离子结合） |
| 🎯 | **Residue** | 活性位点 / 结合位点 / 保守位点 / motif |
| ⚗️ | **Properties** | 序列级理化计算（MW、pI、不稳定指数、GRAVY、二级结构组成） |

四项全勾会得到最完整的报告。

---

## 5. 运行 & 观察

点 **Generate Report**。页面会以流式方式推送事件：

- **Progress** — 进度条 + 消息（例如"Predicting solubility…"、"Scoring mutations…"）
- **Logs** — 追加到右栏
- **Done** — 中栏渲染 AI 评述、HTML / PDF 链接生效

若失败，左栏会出现简短错误说明，右栏日志里有详细堆栈。

---

## 6. 报告内容详解

| 章节 | 内容 |
| :--- | :--- |
| **Comprehensive Summary** | 顶部简报：分子量、理论 pI，以及对整体结果的一段评估。 |
| **Mutation Prediction Analysis** | Top beneficial mutations 表格（Rank / Position / Mutation / Score / Notes）、次选突变、关键位点优化建议。 |
| **Protein Function Analysis** | 各预测任务表格：属性、预测值 / 类别、置信度、说明——覆盖溶解性、定位、金属结合、稳定性、信号肽、最适温度。 |
| **Functional Residue** | 结合位点 / 功能残基 / motif 预测，含序列位置和概率。 |
| **Physical & Chemical Properties** | 生物物理表征，并基于不稳定指数给出稳定性判断。 |
| **Experimental Recommendations** | 综合功能 / 稳定性、技术注意事项、实验方案建议。 |
| **Conclusion** | 整篇报告的最终总结，重申蛋白关键特性和最重要的优化方向。 |

---

## 7. 使用建议

- **有 PDB 优先传 PDB。** 结构相关分析（如结构感知的突变打分）只在提供 `.pdb` 时启用。
- **跨章节交叉读。** 落在**预测结合位点**附近的有利突变，通常比远离功能残基的突变更值得实验验证。
- **下载存档。** HTML / PDF 文件存在后端临时目录里，关浏览器前务必下载到本地。
- **长序列会慢。** 突变打分时间正比于"序列长度 × 替换氨基酸数"。如果只关心某个 domain，可先剪掉信号肽 / 无序区。
