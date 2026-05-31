# Custom Model — Prediction（自定义预测）

> **位置（v2）：** 侧边栏 → **Custom Model → Predict**（`/custom-model/predict`）。
> **仅 Local 模式：** Online 模式下整页只读。

Predict 页面用你训练好的模型对新序列做预测 — 单条或者批量。

---

## 1. 选模型

| 字段 | 说明 |
| :--- | :--- |
| **Model Folder** | 从 `ckpt/` 列表里选 — 由之前的训练运行填充。 |
| **Model Path** | 所选文件夹下的具体 checkpoint 文件（默认 `best_model.pt`）。选定模型后会自动填充其余模型配置。 |

选模型会**锁定**以下字段为 checkpoint config 中的值（不允许用不匹配的设置预测）：

- **PLM** — 必须与训练时一致
- **Eval Method** — `freeze` · `full` · `plm-lora` · `plm-dora` · `plm-adalora` · `plm-ia3` · `plm-qlora` · `ses-adapter`
- **Pooling** — `mean` · `attention1d` · `light_attention`
- **Problem Type** — `single_label_classification` · `multi_label_classification` · `regression`
- **Num Labels** — 分类任务需要

如果 Eval Method 为 `ses-adapter` 或 PLM 为结构感知（ProSST / ProtSSN / SaProt），会出现额外输入项（Structure Seq 选择器、PDB Dir、以及 Single 模式下的 Foldseek / SS8 序列文本框）。

---

## 2. 选预测模式

在 **Single** 和 **Batch** 之间切换。

### 2.1 Single（单条）

| 字段 | 说明 |
| :--- | :--- |
| **AA Sequence** | 粘贴一条氨基酸序列（单字母代码）。 |
| **Foldseek Sequence** | （仅当 `ses-adapter` + `foldseek_seq` 启用）对应的 foldseek 编码字符串。 |
| **SS8 Sequence** | （仅当 `ses-adapter` + `ss8_seq` 启用）对应的 8 类二级结构字符串。 |

点 **Predict** 即可得到即时结果。

### 2.2 Batch（批量）

| 字段 | 说明 |
| :--- | :--- |
| **Source** | `Upload file`（上传 CSV/TSV/XLSX） · `Paste FASTA`（在文本框粘贴 FASTA 内容） · `Path`（指向服务器上已有的文件路径）。 |
| **Workspace** | upload 源还支持从 Workspace 选。 |
| **Batch Size** | 每次前向传播打分的序列数。长序列 OOM 时降低。 |

文件输入（Upload / Path）期望的列：

| 列 | 必填？ | 说明 |
| :--- | :---: | :--- |
| `aa_seq` | ✓ | 氨基酸序列。 |
| `id` / `name` | 可选 | 样本标识。 |
| `foldseek_seq` | 可选 | `ses-adapter` + `foldseek_seq` 启用时必须。 |
| `ss8_seq` | 可选 | `ses-adapter` + `ss8_seq` 启用时必须。 |

---

## 3. 运行 & 观察

- **Preview Command** — 等效 CLI 命令。
- **Start** — 启动预测；页面实时显示进度条和预测日志尾部。
- **Logs** 面板显示每批进度和数据问题（非法 AA 字符、缺失结构序列等）。

---

## 4. 结果

| 问题类型 | 你能得到 |
| :--- | :--- |
| **Single-label classification** | 预测类别 + 各类概率分布。 |
| **Multi-label classification** | 每个标签 0/1 + 每个标签的概率。 |
| **Regression** | 预测数值。 |

批量模式下，结果保存为 CSV，包含每条输入样本和预测列。从结果面板下载。

---

## 5. 完整使用流程

### 5.1 单条预测

对单个蛋白做快速验证。

1. **选模型** — Model Folder + Model Path；其余模型配置自动锁定。
2. **切换到 Single 模式。**
3. **粘贴 AA 序列** 到文本框（仅单字母代码）。
4. （`ses-adapter` 模型）粘贴对应的 **Foldseek** 和/或 **SS8** 序列。
5. **Predict** — 结果出现在面板：
   - 单标签分类 → 预测类别 + 各类概率。
   - 多标签分类 → 各标签 0/1 + 概率。
   - 回归 → 预测数值。
6. 运行出问题就 **Abort**。

### 5.2 批量预测

扩展到上百 / 上千条序列。

1. **选模型** — 同 Single。
2. **切换到 Batch 模式。**
3. **选 Source：**
   - `Upload file` — 拖入 CSV / TSV / XLSX，或从 Workspace 选。
   - `Paste FASTA` — 在文本框粘贴 multi-record FASTA。
   - `Path` — 指向服务器上已有的文件路径（巨大文件最快）。
4. **看输入行 / 记录的预览**，确认列映射正确。
5. **设置 Batch Size** — 16–32 通常够用；长序列 OOM 就降低。
6. **Start** — 进度条显示总数 / 已处理 / 剩余时间。
7. **下载 CSV** — 每条输入行加上预测列。

---

## 6. 小贴士

- **序列要规范。** 粘贴前去掉终止密码子、换行和非 AA 字符。
- **长序列。** 如果训练时限了序列长度，更长输入预测时可能被静默截断 — 看日志。
- **结构感知模型。** 别忘了结构侧输入（批量用 PDB Dir，单条用 Foldseek / SS8 文本框）。
- **>10 条序列用 Batch。** Single 模式每次调用都重建 GPU 流水线；批量摊销了这个开销。
- **自定义工作流：** 要做海量筛选时，从数据库查询或脚本生成输入 CSV，通过 **Source = Path** 喂进来 — 比 upload UI 快多了。
