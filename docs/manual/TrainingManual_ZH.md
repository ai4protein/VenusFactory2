# Custom Model — Training（自定义训练）

> **位置（v2）：** 侧边栏 → **Custom Model → Training**（`/custom-model/training`）。
> **仅 Local 模式：** Online 模式下整页只读，不能训练。

Training 页面让你在自己的数据集（或 VenusFactory 预定义数据集）上微调一个蛋白语言模型（PLM），完全不用写训练代码。覆盖全生命周期：数据集上传 → 方法 / 超参配置 → 实时曲线训练 → 测试指标 + CSV 导出。

## 1. 两种模式

| 模式 | 什么时候用 |
| :--- | :--- |
| **From Scratch** | 微调一个新 PLM。需要 PLM Model、Output Dir、Output Model Name。 |
| **Continue Training** | 从 checkpoint 续训。需要 Model Folder（`ckpt/` 下）+ Checkpoint 文件。 |

两种模式下，数据集既可以用预定义的，也可以用自定义的。

## 2. 支持的 PLM

| 家族 | 可选规模 | HF 示例 |
| :--- | :--- | :--- |
| **ESM2** | 8M / 35M / 150M / 650M / 3B / 15B | `facebook/esm2_t33_650M_UR50D` |
| **ESM-1v** | 650M × 5 seeds | `facebook/esm1v_t33_650M_UR90S_1` |
| **ProtBert** | 420M（Uniref100 / BFD） | `Rostlab/prot_bert_bfd` |
| **IgBert / IgBert_unpaired** | 420M | `Exscientia/IgBert` |
| **ProtT5** | 3B / 11B（Uniref50 / BFD） | `Rostlab/prot_t5_xl_uniref50` |
| **IgT5 / IgT5_unpaired** | 3B | `Exscientia/IgT5` |
| **Ankh** | 450M / 1.2B | `ElnaggarLab/ankh-base` |
| **ProSST** | 110M × 7 codebook（20/128/512/1024/2048/4096 + AE） | `AI4Protein/ProSST-2048` |
| **ProPrime** | 690M | `AI4Protein/Prime_690M` |
| **VenusPLM** | 300M | `AI4Protein/VenusPLM-300M` |
| **PETA** | 80M（base / bpe / unigram） | `AI4Protein/PETA-base` |

> **按显存选：** <8 GB → ESM2-8M / 35M、ProSST · 8–16 GB → ESM2-150M / 650M、ProtBert · 24 GB+ → ESM2-3B、ProtT5-XL · 多卡 → ESM2-15B、ProtT5-XXL。

## 3. 支持的微调方法

| 方法 | 描述 | 数据类型 |
| :--- | :--- | :--- |
| **freeze** | 冻结 PLM，只训分类头 | 序列 |
| **full** | 训练所有参数 | 序列 |
| **plm-lora** | Low-Rank Adaptation | 序列 |
| **plm-dora** | Weight-Decomposed Low-Rank Adaptation | 序列 |
| **plm-adalora** | Adaptive Low-Rank Adaptation | 序列 |
| **plm-ia3** | Infused Adapter (IA³) | 序列 |
| **plm-qlora** | Quantized LoRA（最省显存） | 序列 |
| **ses-adapter** | 结构增强序列 adapter | 序列 + 结构 |

> **ses-adapter** 需要勾选所用的 structure sequence 类型（`foldseek_seq` / `ss8_seq`）。
> 结构感知 PLM（**ProSST / ProtSSN / SaProt**）需要 **PDB Dir**，让模型能读取每条样本的结构。

## 4. 评价指标

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

## 5. 页面详解

### 5.1 数据集

| 字段 | 说明 |
| :--- | :--- |
| **Dataset Source** | `Pre-defined`（下拉选 — 自动填充问题类型 / 标签数 / 指标 / 序列与标签列）或 `Custom`。 |
| **HF ID 或本地上传**（自定义） | 要么填 `username/dataset_name` 的 HF 路径，要么上传 `train / valid / test` 文件（CSV / TSV / XLSX）。支持 Workspace 选择器。 |
| **Problem Type** | `single_label_classification` · `multi_label_classification` · `regression`。 |
| **Num Labels** | 分类任务必填。 |
| **Sequence Column** | 从上传文件自动检测，可覆盖。默认约定：`aa_seq`。 |
| **Label Column** | 同上。默认：`label`。 |
| **Metrics** | 多选格子；选与问题类型相关的指标。 |
| **Preview Dataset** | 快速看 train / val / test 数量和样例行。 |

### 5.2 训练方法

| 字段 | 说明 |
| :--- | :--- |
| **PLM Model** | 下拉选 — "From Scratch" 必填。 |
| **Training Method** | §3 中之一。 |
| **Pooling** | `mean` · `attention1d` · `light_attention`。 |
| **Structure Seq** | （仅 ses-adapter / 结构 PLM）— `foldseek_seq`、`ss8_seq`。 |
| **PDB Dir** | （仅 ProSST / ProtSSN / SaProt）— 本地目录，每条样本对应一个 PDB。 |
| **LoRA 参数** | （plm-lora / qlora / adalora / dora / ia3）— `lora_r`、`lora_alpha`、`lora_dropout`、`lora_target_modules`（默认 `query,key,value`）。 |
| **Model Folder + Checkpoint** | （仅 Continue Training）— 从 `ckpt/` 列表里选。 |

### 5.3 Batch

| 字段 | 说明 |
| :--- | :--- |
| **Batch Mode** | `Batch Size Mode`（每批固定 N 条样本）或 `Batch Token Mode`（每批约 N 个 token — 序列长度差异大时更好）。 |
| **Batch Size / Tokens** | 默认：16 / 10000。 |

### 5.4 超参数

| 字段 | 默认 | 说明 |
| :--- | :--- | :--- |
| Learning Rate | `5e-4` | 全参微调用更低；LoRA 系列用更高。 |
| Num Epochs | `20` | 早停通常会更早结束。 |
| Patience | `10` | 验证指标 N 个 epoch 不提升就停。 |
| Max Seq Length | `1024` | `-1` 表示不限。 |
| Scheduler | `linear` | `linear` / `cosine` / `step`。 |
| Warmup Steps | `0` | 建议 5–10% 总步数。 |
| Gradient Accumulation Steps | `1` | 显存不够时模拟更大 batch。 |
| Max Grad Norm | `-1` | `-1` = 不裁剪；不稳时设 1.0–5.0。 |
| Num Workers | `4` | 数据加载线程数。 |
| Monitored Metric | （依任务） | 驱动早停和最佳模型选择。 |
| Monitored Strategy | `max` | `max` 或 `min`，看指标方向。 |

### 5.5 输出

| 字段 | 说明 |
| :--- | :--- |
| **Save Directory** | 默认 `ckpt/`。 |
| **Output Model Name** | 默认 `demo/best_model.pt`。 |
| **Enable W&B Logging** | 可选 — 启用后设置 `wandb_project`、`wandb_entity`。 |

### 5.6 运行 & 观察

- **Preview Command** — 显示等效 CLI 命令（复现 / 写脚本时有用）。
- **Start** — 启动训练；右栏实时显示训练 / 验证 loss 曲线、每 epoch 指标、模型统计（总参数 / 可训练参数 + 百分比）和进度日志。
- **Abort** — 优雅停止当前运行。
- **Test Results** — 最终各项指标 + CSV 下载。

---

## 6. 自定义数据集格式

走 Hugging Face / 文件上传路径时，数据应包含这些列：

| 列 | 必填？ | 说明 |
| :--- | :---: | :--- |
| `aa_seq`（或你选的序列列） | ✓ | 氨基酸序列，单字母代码。 |
| `label`（或你选的标签列） | ✓ | 回归用数值；分类用整数 / 列表。 |
| `foldseek_seq` | 可选 | `ses-adapter` 启用 `foldseek_seq` 时需要。 |
| `ss8_seq` | 可选 | `ses-adapter` 启用 `ss8_seq` 时需要。 |
| `name` / `id` | 可选 | 样本标识。 |

需要的切分：`train`、`validation`、`test`。Custom 路径接受 3 个独立文件；Pre-defined 路径期望 HF 数据集结构。

### 6.1 各问题类型的标签格式

**单标签分类（single_label_classification）** — `label` 是整数类别索引（从 0 开始）：

```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,1
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,0
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,2
```

**多标签分类（multi_label_classification）** — `label` 是逗号分隔的类别索引字符串（用引号包起来，避免逗号破坏 CSV）：

```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,"373,449,584,674,780,883,897,911,1048,1073,1130,1234"
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,"15,42,87,103,256"
```

**回归（regression）** — `label` 是浮点数：

```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,0.75
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,-1.2
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,3.45
```

### 6.2 加入结构列（ses-adapter）

用 `ses-adapter` 训练时，加上 `foldseek_seq` 和/或 `ss8_seq` 列。每行字符要与 `aa_seq` 一一对齐：

```csv
name,aa_seq,foldseek_seq,ss8_seq,label
Q9LSD8,MPEEDLVELKFR...,DPPQLWAFAWEA...,LLLLLLEEEEEE...,0
```

### 6.3 上传到 Hugging Face

1. 创建三个文件：`train.csv`、`validation.csv`、`test.csv`。
2. 推送到一个新的 HuggingFace 数据集仓库。
3. 在 Training 页面把数据集引用为 **Custom Path** = `username/dataset_name`。

---

## 7. 工作流总结

1. **Dataset** — 预定义或自定义；训前先预览。
2. **Model + Method** — 选 PLM、微调方法、池化。按需配置 LoRA / 结构序列 / PDB Dir。
3. **Batch + Hyperparams** — 默认起步，需要时再调。
4. **Output** — 选保存路径，按需启用 W&B。
5. **Preview Command** — sanity check。
6. **Start** — 监控曲线和指标；不对劲就 **Abort**。
7. **Test Results** — 下载 CSV；保存的模型现在可在 **Custom Model → Evaluation** 和 **Custom Model → Predict** 里使用。
