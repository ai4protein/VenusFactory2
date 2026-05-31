# Custom Model — Evaluation（自定义评估）

> **位置（v2）：** 侧边栏 → **Custom Model → Evaluation**（`/custom-model/evaluation`）。
> **仅 Local 模式：** Online 模式下整页只读。

Evaluation 页面用来在留出集上测量训练好的模型实际表现。选一个模型 checkpoint、选一个数据集（预定义或自定义），页面会端到端跑模型并报告各项指标。

---

## 1. 输入

| 字段 | 说明 |
| :--- | :--- |
| **Model Folder** | 从 `ckpt/` 列表里选 — 由之前的训练运行填充。 |
| **Model Path** | 所选文件夹下具体的 checkpoint 文件（默认 `best_model.pt`）。选定模型后会从保存的 config 自动填 PLM、Eval Method、Pooling、Problem Type 和 Num Labels。 |
| **PLM** | （选定模型后锁定）必须与训练时一致。 |
| **Eval Method** | （选定模型后锁定）`freeze` · `full` · `plm-lora` · `plm-dora` · `plm-adalora` · `plm-ia3` · `plm-qlora` · `ses-adapter`。必须与训练一致。 |
| **Pooling** | （选定模型后锁定）`mean` · `attention1d` · `light_attention`。必须与训练一致。 |
| **Problem Type** | （Pre-defined 数据集时锁定）`single_label_classification` · `multi_label_classification` · `regression`。 |
| **Num Labels** | （Pre-defined 数据集时锁定）分类任务必填。 |

> 如果 Eval Method 为 `ses-adapter` 或 PLM 为结构感知（ProSST / ProtSSN / SaProt），会多出一个 **Structure Seq** 选择器（foldseek_seq / ss8_seq）和一个 **PDB Dir** 字段。

---

## 2. 数据集

| 模式 | 说明 |
| :--- | :--- |
| **Pre-defined** | 从数据集配置下拉选 — VenusFactory 自动填问题类型、标签数、指标和列映射。 |
| **Custom** | 要么填 Hugging Face `username/dataset_name`，要么上传测试文件（CSV / TSV / XLSX / XLS）。支持 Workspace 选择器。 |

自定义数据集要设置：

- **Problem Type** + **Num Labels**
- **Sequence Column** / **Label Column** — 从文件自动检测，可覆盖
- **Metrics** — 多选

点 **Preview Dataset** 看 train / val / test 切分数量和样例行。

---

## 3. Batch

| 字段 | 默认 |
| :--- | :--- |
| **Batch Mode** | `Batch Size Mode` 或 `Batch Token Mode` |
| **Batch Size / Tokens** | 16 / 10000 |

序列长度差异大时用 **Batch Token Mode**。

---

## 4. 运行 & 观察

- **Preview Command** — 等效 CLI 命令，便于复现 / 脚本化。
- **Start** — 启动评估；页面实时显示进度条和评估日志尾部。
- **Test Results** — 各项指标；点 CSV 下载保存原始输出。

---

## 5. 支持的指标

| 缩写 | 指标 | 适用问题类型 | 方向 |
| :--- | :--- | :--- | :---: |
| **Accuracy** | 正确预测比例 | 单 / 多标签分类 | ↑ |
| **Recall** | 真阳率 | 单 / 多标签分类 | ↑ |
| **Precision** | 阳性预测值 | 单 / 多标签分类 | ↑ |
| **F1** | 精确率 / 召回率的调和均值 | 单 / 多标签分类 | ↑ |
| **MCC** | Matthews 相关系数 | 单 / 多标签分类 | ↑ |
| **AUROC** | ROC 曲线下面积 | 单 / 多标签分类 | ↑ |
| **F1_max** | 多阈值下最大 F1 | 多标签分类 | ↑ |
| **Spearman_corr** | 秩相关 | 回归 | ↑ |
| **MSE** | 均方误差 | 回归 | ↓ |

---

## 6. 完整使用流程

从模型 + 数据集到结果解读的完整一遍。

### 6.1 准备模型与数据集

1. **挑选训好的模型。** 应该已经被 Training 页面保存在 `ckpt/` 下。记下产生它的 **PLM**、**Training Method**、**Pooling** — 选定 checkpoint 后 config 会自动锁这些。
2. **挑选评估数据集。**
   - 用训练时的同一数据集？取 test split — 测留出集表现。
   - 新数据集？用 Custom 路径 / 上传 — 测分布外泛化。
   - 确认列约定一致（序列列、标签列，`ses-adapter` 需要结构序列列）。

### 6.2 配置

1. **Model Folder + Model Path** — 从 `ckpt/` 列表选。PLM / Eval Method / Pooling 自动锁。
2. **Dataset Source** — Pre-defined 或 Custom。
3. **Sequence / Label 列** — 上传时自动检测；列名不标准就手动覆盖。
4. **Metrics** — 多选。分类任务用 `accuracy + mcc + f1 + precision + recall + auroc` 基本够；回归用 `mse + spearman_corr`。
5. **Structure Seq + PDB Dir** — 模型为 `ses-adapter` 或结构感知 PLM 时必填。
6. **Batch Mode + Size** — 长度均匀用 Batch Size Mode，差异大用 Batch Token Mode。
7. **Preview Dataset** — 跑之前先看一眼切分数量和几行样例。

### 6.3 运行

1. **Preview Command** — 看一眼等效 CLI 命令，双重确认每个 flag 都符合预期。
2. **Start** — 看进度条；日志实时显示 batch 级更新和任何数据警告。
3. **Abort** — 不对劲就优雅停止。

### 6.4 解读结果

1. **指标表** — 关注对任务最有判别力的指标（方向看 §5 表）。
2. **Download CSV** — 包含逐样本预测，便于做混淆矩阵、错误分析或喂下游脚本。
3. **决策。** 表现达标就可以拿到 **Custom Model → Predict** 用了。不达标就调方法 / 超参重训，或回头看数据集（参考 QA Q14 / Q15）。

---

## 7. 小贴士

- **必须和训练配置一致。** Checkpoint 的 config 会自动锁住 PLM / 方法 / 池化 — 不要尝试换方法评估，结果是噪声。
- **基准比较用 Pre-defined 数据集。** 让指标配置保持诚实和可复现。
- **自定义测试文件**结构应与训练集一致（序列 + 标签列）。`ses-adapter` 模型用到结构序列时，要包含 `foldseek_seq` / `ss8_seq` 列。
- **测试集很大？** OOM 就降低 batch size；评估不需要梯度，但激活仍占显存。
- **输出 CSV** 包含逐样本预测 — 便于在下游工具里做错误分析和混淆矩阵。
