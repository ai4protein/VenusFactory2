# VenusFactory 训练模块使用指南

## 1. 简介

VenusFactory 训练模块是一个强大的工具，允许您使用蛋白质序列数据训练自定义模型。这些模型可以预测蛋白质的各种特性，如亚细胞定位、功能、稳定性等。训练模块提供了直观的界面，使生物学研究人员无需编程知识即可训练高性能的蛋白质预测模型。

## 2. 支持的蛋白质语言模型

VenusFactory 支持多种先进的蛋白质语言模型，您可以根据任务需求和计算资源选择合适的模型。

| 模型名称                                                    | 模型参数规模          | 模型数量                | 模型样例                        |
| ------------------------------------------------------------ | ----------------------- | -------- | ------------------------------- |
| [ESM2](https://huggingface.co/facebook/esm2_t33_650M_UR50D)  | 8M/35M/150M/650M/3B/15B | 6        | facebook/esm2_t33_650M_UR50D    |
| [ESM-1b](https://huggingface.co/facebook/esm1b_t33_650M_UR50S) | 650M                    | 1        | facebook/esm1b_t33_650M_UR50S   |
| [ESM-1v](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1) | 650M                    | 5        | facebook/esm1v_t33_650M_UR90S_1 |
| [ProtBert-Uniref100](https://huggingface.co/Rostlab/prot_bert) | 420M                    | 1        | Rostlab/prot_bert_bfd           |
| [ProtBert-BFD100](https://huggingface.co/Rostlab/prot_bert_bfd) | 420M                    | 1        | Rostlab/prot_bert_bfd           |
| [IgBert](https://huggingface.co/Exscientia/IgBert) | 420M                    | 1        | Exscientia/IgBert           |
| [IgBert_unpaired](https://huggingface.co/Exscientia/IgBert_unpaired) | 420M                    | 1        | Exscientia/IgBert_unpaired           |
| [ProtT5-Uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) | 3B/11B                  | 2        | Rostlab/prot_t5_xl_uniref50     |
| [ProtT5-BFD100](https://huggingface.co/Rostlab/prot_t5_xl_bfd) | 3B/11B                  | 2        | Rostlab/prot_t5_xl_bfd          |
| [IgT5](https://huggingface.co/Exscientia/IgT5) | 3B                  | 1        | Exscientia/IgT5          |
| [IgT5_unpaired](https://huggingface.co/Exscientia/IgT5_unpaired) | 3B                  | 1        | Exscientia/IgT5_unpaired          |
| [Ankh](https://huggingface.co/ElnaggarLab/ankh-base)         | 450M/1.2B               | 2        | ElnaggarLab/ankh-base           |
| [ProSST](https://huggingface.co/AI4Protein/ProSST-2048)  |110M                     | 7        | AI4Protein/ProSST-2048     |
| [ProPrime](https://huggingface.co/AI4Protein/Prime_690M)  |690M                     | 1        | AI4Protein/Prime_690M     |

## 3. 支持的微调方法

VenusFactory 提供多种训练方法，每种方法有其特定的优势和适用场景。

| 微调方法 | 描述 | 数据类型 | 内存使用 | 训练速度 | 性能 |
|---------|------|------------|---------|---------|------|
| **Freeze** | 冻结预训练模型，只训练分类器 | 序列信息 | 低 | 快速 | 良好 |
| **Full** | 全参数微调，训练所有参数 | 序列信息 | 高 | 慢速 | 优秀 |
| **LoRA** | 使用LoRA (Low-Rank Adaptation)方法训练，减少参数量 | 序列信息 | 低 | 快速 | 良好 |
| **DoRA** | 使用DoRA (Weight-Decomposed Low-Rank Adaptation)方法训练 | 序列信息 | 低 | 中等 | 较好 |
| **AdaLoRA** | 使用AdaLoRA (Adaptive Low-Rank Adaptation)方法训练 | 序列信息 | 低 | 中等 | 较好 |
| **IA3** | 使用IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)方法训练 | 序列信息 | 极低 | 快速 | 良好 |
| **QLoRA** | 使用QLoRA (Quantized Low-Rank Adaptation)方法训练，降低内存需求 | 序列信息 | 极低 | 较慢 | 良好 |
| **SES-Adapter** | 使用结构增强适配器训练，融合序列和结构信息 | 序列信息 & 结构信息 | 中等 | 中等 | 较好 |

*注：具体的模型表现取决于特定问题和数据集，上表中的性能评估仅供参考。*

## 4. 支持的评估指标

VenusFactory 提供多种评估指标，用于评估模型性能。

| 简称 | 指标名称 | 适用问题类型 | 说明 | 优化方向 |
|---------|------|------------|------|---------|
| **Accuracy** | 准确率 (Accuracy) | 单标签/多标签分类 | 正确预测的样本比例，适用于平衡的数据集 | 越大越好 |
| **Recall** | 召回率 (Recall) | 单标签/多标签分类 | 正确识别出的正类比例，关注减少假阴性 | 越大越好 |
| **Precision** | 精确率 (Precision) | 单标签/多标签分类 | 正确预测为正类的比例，关注减少假阳性 | 越大越好 |
| **F1** | F1分数 (F1Score) | 单标签/多标签分类 | 精确率和召回率的调和平均，适用于不平衡的数据集 | 越大越好 |
| **MCC** | Matthews相关系数 (MatthewsCorrCoef) | 单标签/多标签分类 | 综合考虑所有混淆矩阵元素的指标，对不平衡数据集更公平 | 越大越好 |
| **AUROC** | ROC曲线下面积 (AUROC) | 单标签/多标签分类 | 评估不同阈值下的分类性能 | 越大越好 |
| **F1_max** | 最大F1分数 (F1ScoreMax) | 多标签分类 | 不同阈值下的最大F1值，适用于多标签分类 | 越大越好 |
| **Spearman_corr** | Spearman相关系数 (SpearmanCorrCoef) | 回归 | 评估预测值与真实值的单调关系，范围为[-1,1] | 越大越好 |
| **MSE** | 均方误差 (MeanSquaredError) | 回归 | 评估回归模型的预测误差 | 越小越好 |

## 5. 训练界面详解

训练界面分为几个主要部分，每个部分包含特定的配置选项。

### 5.1 模型和数据集配置

#### 蛋白质语言模型选择
- **Protein Language Model**：从下拉菜单中选择一个预训练模型
  - 选择时考虑您的计算资源和任务复杂度
  - 较大的模型需要更多计算资源

#### 数据集选择
- **Dataset Selection**：选择数据集来源
  - **Use Pre-defined Dataset**：使用系统预定义的数据集
    - **Dataset Configuration**：从下拉菜单中选择一个数据集
    - 系统会自动加载数据集的问题类型、标签数量和评估指标
  - **Use Custom Dataset**：使用自定义数据集
    - **Custom Dataset Path**：输入Hugging Face数据集路径（格式：`用户名/数据集名`）
    - **Problem Type**：选择问题类型
      - `single_label_classification`：单标签分类
      - `multi_label_classification`：多标签分类
      - `regression`：回归
    - **Number of Labels**：设置标签数量（分类问题）
    - **Metrics**：选择评估指标（可多选）
      - `accuracy`：准确率 (Accuracy)
      - `f1`：F1分数 (F1Score)
      - `precision`：精确率 (Precision)
      - `recall`：召回率 (Recall)
      - `mcc`：Matthews相关系数 (MatthewsCorrCoef)
      - `auroc`：ROC曲线下面积 (AUROC)
      - `f1max`：最大F1分数 (F1ScoreMax)
      - `spearman_corr`：Spearman相关系数 (SpearmanCorrCoef)
      - `mse`：均方误差 (MeanSquaredError)

      具体信息参考 [4. 支持的评估指标](#header-4)


#### 数据集预览
- **Preview Dataset**：点击此按钮预览所选数据集
  - 显示数据集统计信息：训练集、验证集和测试集的样本数量
  - 显示数据集样例：包括序列和标签

### 5.2 训练方法配置

- **Training Method**：选择训练方法
  - `freeze`：冻结预训练模型，只训练分类器
  - `full`：全参数微调，训练所有参数
  - `plm-lora`：使用LoRA (Low-Rank Adaptation)方法训练，减少参数量
  - `dora`：使用DoRA (Weight-Decomposed Low-Rank Adaptation)方法训练
  - `adalora`：使用AdaLoRA (Adaptive Low-Rank Adaptation)方法训练
  - `ia3`：使用IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)方法训练
  - `plm-qlora`：使用QLoRA (Quantized Low-Rank Adaptation)方法训练，降低内存需求
  - `ses-adapter`：使用结构增强适配器训练，融合序列和结构信息

  具体信息参考 [3. 支持的微调方法](#header-3)

- **Pooling Method**：选择池化方法
  - `mean`：平均池化
  - `attention1d`：注意力池化
  - `light_attention`：轻量级注意力池化

- **Structure Sequence**（当选择`ses-adapter`时可见）：
  - 选择结构序列类型（可多选），默认为多选`foldseek_seq`和`ss8_seq`

- **LoRA Parameters**（当选择`plm-lora`或`plm-qlora`时可见）：
  - **LoRA Rank**：LoRA的秩，默认为8，影响参数量和性能
  - **LoRA Alpha**：LoRA的alpha值，默认为32，影响缩放因子
  - **LoRA Dropout**：LoRA的dropout率，默认为0.1，影响正则化
  - **LoRA Target Modules**：LoRA应用的目标模块，默认为`query,key,value`

### 5.3 批处理配置

- **Batch Processing Mode**：选择批处理模式
  - **Batch Size Mode**：固定批次大小
    - **Batch Size**：设置每批处理的样本数量，默认为16
  - **Batch Token Mode**：固定Token数量
    - **Tokens per Batch**：设置每批处理的Token数量，默认为10000
    - 适用于序列长度变化大的数据集

### 5.4 训练参数

- **Learning Rate**：学习率，默认为5e-4
  - 影响模型训练的步长，较大值可能导致不收敛，较小值可能导致训练缓慢

- **Number of Epochs**：训练轮数，默认为100
  - 完整数据集的训练次数
  - 实际训练可能因早停而提前结束

- **Early Stopping Patience**：早停轮数 N，默认为10
  - 如果验证性能在连续 N 轮没有提升，训练将提前停止

- **Max Sequence Length**：最大序列长度，默认为None（-1表示无限制）
  - 处理的最大蛋白质序列长度

- **Scheduler Type**：学习率调度器类型
  - `linear`：线性衰减
  - `cosine`：余弦衰减
  - `step`：阶梯式衰减
  - `None`：不使用调度器

- **Warmup Steps**：预热步数，默认为0
  - 学习率从小到设定值逐渐增加的步数
  - 有助于稳定训练初期

- **Gradient Accumulation Steps**：梯度累积步数，默认为1
  - 累积多个批次的梯度后再更新模型
  - 可以模拟更大的批次大小

- **Max Gradient Norm**：梯度裁剪阈值，默认为-1（不裁剪）
  - 限制梯度的最大范数，防止梯度爆炸
  - 推荐范围：1.0到5.0

- **Number of Workers**：数据加载的工作线程数，默认为4
  - 影响数据加载速度
  - 根据CPU核心数调整

### 5.5 输出和日志设置

- **Save Directory**：保存目录，默认为`ckpt`
  - 模型和训练结果的保存路径

- **Output Model Name**：输出模型名称，默认为`model.pt`
  - 保存的模型文件名

- **Enable W&B Logging**：是否启用Weights & Biases日志
  - 勾选后可设置W&B项目名称和实体
  - 用于实验跟踪和可视化

### 5.6 训练控制和输出

- **Preview Command**：预览将要执行的训练命令
  - 点击后显示完整的命令行参数

- **Abort**：中止当前训练过程

- **Start**：开始训练过程

- **Model Statistics**：显示模型参数统计
  - 训练模型、预训练模型和组合模型的参数量
  - 可训练参数的百分比

- **Training Progress**：显示训练进度
  - 当前阶段（训练、验证、测试）
  - 进度百分比
  - 已用时间和预估剩余时间
  - 当前损失值和梯度步数

- **Best Performance**：显示最佳模型信息
  - 最佳轮次和对应的评估指标

- **Training and Validation Loss**：损失曲线图
  - 训练损失和验证损失随时间变化

- **Validation Metrics**：验证集评估指标图
  - 各项评估指标随时间变化

- **Test Results**：测试结果
  - 在测试集上的最终性能指标
  - 可下载CSV格式的评估指标

## 6. 训练流程指南

以下是使用VenusFactory训练模块的完整流程指南，从数据准备到模型评估。

### 6.1 准备数据集

#### 使用预定义数据集
1. 在**Dataset Selection**中选择"Use Pre-defined Dataset"
2. 从**Dataset Configuration**下拉菜单中选择一个数据集
3. 点击**Preview Dataset**按钮查看数据集统计和样例

#### 使用自定义数据集
1. 准备符合要求的数据集并上传到Hugging Face（详见 [自定义数据集格式要求](#header-7)）
2. 在**Dataset Selection**中选择"Use Custom Dataset"
3. 在**Custom Dataset Path**中输入Hugging Face数据集路径（格式：`用户名/数据集名`）
4. 设置**Problem Type**、**Number of Labels**和**Metrics**
5. 点击**Preview Dataset**按钮验证数据集是否正确加载

### 6.2 选择模型和训练方法

1. 从**Protein Language Model**下拉菜单中选择一个预训练模型

2. 选择合适的**Training Method**

3. 选择**Pooling Method**

4. 如果选择`ses-adapter`，确保在**Structure Sequence**中指定结构序列类型
5. 如果选择`plm-lora`或`plm-qlora`，根据需要调整LoRA参数

### 6.3 配置批处理和训练参数

1. 选择**Batch Processing Mode**
   - 序列长度相近时可以使用**Batch Size Mode**
   - 序列长度差异大时推荐使用**Batch Token Mode**

2. 设置批次大小或Token数量
   - 根据GPU内存调整，出现内存错误时减小值

3. 设置**Learning Rate**

4. 设置**Number of Epochs**
   - 使用早停机制，设置**Early Stopping Patience**为10-20，防止过拟合

5. 设置**Max Sequence Length**

6. 根据需要调整高级参数
   - **Scheduler Type**：推荐使用`linear`或`cosine`
   - **Warmup Steps**：推荐设置为总步数的5-10%
   - **Gradient Accumulation Steps**：内存不足时增大
   - **Max Gradient Norm**：训练不稳定时设置为1.0-5.0

### 6.4 设置输出和日志

1. 设置 **Save Directory** 为模型保存路径
2. 设置 **Output Model Name** 为模型文件名
3. 如需跟踪训练情况，勾选 **Enable W&B Logging** 并设置项目信息

### 6.5 开始训练

1. 点击 **Preview Command** 预览训练命令
2. 点击 **Start** 按钮开始训练
3. 观察训练进度和指标变化
4. 训练完成后，查看测试结果
   - 检查各项评估指标
   - 可下载CSV格式的结果
5. 如需中止训练，点击**Abort**按钮

## 7. 自定义数据集格式要求

要使用自定义数据集，您需要将数据集上传到Hugging Face平台，并确保其符合以下格式要求。

### 7.1 基本要求

- 数据集必须包含`train`、`validation`和`test`三个子集
- 每个样本必须包含以下字段：
  - `aa_seq`：氨基酸序列，使用标准单字母代码
  - `label`：标签，格式取决于问题类型

### 7.2 不同问题类型的标签格式

#### 单标签分类（single_label_classification）
- `label`：整数值，表示类别索引（从0开始）
- 例如：0, 1, 2, ...

CSV 格式示例：
```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,1
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,0
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,2
```


#### 多标签分类（multi_label_classification）
- `label`：以逗号分隔的类别索引字符串，表示存在的类别
- 例如："373,449,584,674,780,883,897,911,1048,1073,1130,1234"

CSV 格式示例：
```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,"373,449,584,674,780,883,897,911,1048,1073,1130,1234"
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,"15,42,87,103,256"
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,"7,98,120,256,512,789"
```

#### 回归（regression）
- `label`：浮点数，表示连续值
- 例如：0.75, -1.2, ...

CSV 格式示例：
```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,0.75
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,-1.2
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,3.45
```

### 7.3 结构信息（可选）

如果使用`ses-adapter`训练方法，可以添加以下结构信息字段：

- `foldseek_seq`：FoldSeek结构序列，使用单字母代码表示结构元素
- `ss8_seq`：8类二级结构序列，使用单字母代码表示二级结构

CSV 格式示例：
```csv
name,aa_seq,labelname,aa_seq,foldseek_seq,ss8_seq,label
Q9LSD8,MPEEDLVELKFRLYDGSDVGPFQYSPTATVSMLKERIVSEWPKDKKIVPKSASDIKLINAGKILENGKTVAQCKAPFDDLPKSVITMHVVVQLSPTKARPEKKIEKEEAPQRSFCSCTIM,DPPQLWAFAWEAEPVRDIDDRDTDHQQQFLLVVLQVCLVRPDPPDPDHAPHSVQKWKDDPNDTGDRNDGNNRRDDPPDDDSPDHHYIYIDGRDPPVVPPVPPPPPPPPPPPPPPPPPPPD,LLLLLLEEEEEELTTSLEEEEEEELTTLBHHHHHHHHHHTLLTTLSSLLSSGGGEEEEETTEELLTTLBHHHHLLLLLLLTTLLEEEEEEELLLLLLLLLLLLLLLLLLLLLLLLLLLLL,0
```

### 7.4 上传数据集到Hugging Face

1. 为训练集、验证集和测试集分别创建CSV文件：
  - `train.csv`：训练数据
  - `validation.csv`：验证数据
  - `test.csv`：测试数据

2. 将数据集上传至Hugging Face

- 相关步骤如下列图所示：

![HF1](/img/HuggingFace/HF1.png)
![HF2](/img/HuggingFace/HF2.png)
![HF3](/img/HuggingFace/HF3.png)
![HF4](/img/HuggingFace/HF4.png)

3. 上传后，在VenusFactory中使用`用户名/数据集名`作为Custom Dataset Path