# 快速Demo指南

本文档提供了一个全面的指南，帮助您快速了解VenusFactory的主要功能，并在一个蛋白质可溶性预测的Demo数据集上进行微调训练、评估和预测。

## 1. 环境准备

在开始之前，请确保您已成功安装 **VenusFactory** 并正确配置了相应的环境和 Python 依赖包。如果尚未安装，请参考 [README.md](README_CN.md) 中的 **✈️ Requirements** 章节进行安装。

## 2. 启动 Web 界面

在命令行中输入以下命令，启动 Web UI：
```bash
python src/webui.py
```

## 3. 训练（Training Tab）

### 3.1 选择预训练模型

在 Protein Language Model 选项中选择合适的预训练模型。建议从 ESM2-8M 开始，该模型计算成本较低，便于快速上手。

### 3.2 选择数据集

在 Dataset Configuration 选项中，选择 Demo_Solubility 数据集（默认选项）。点击 Preview Dataset 按钮可预览数据集内容。

### 3.3 设定任务参数

- Problem Type、Number of Labels 和 Metrics 选项会在选择 Pre-defined Dataset 时自动填充。

- Batch Processing Mode 建议选择 Batch Token Mode，以避免蛋白质序列长度方差过大导致批处理不均。

- Batch Token 推荐设为 4000，若出现 CUDA 内存不足错误，可适当减小该值。

### 3.4 选择训练方法

在 Training Parameters 选项中：

- Training Method 为关键选择项。本 Demo 数据集暂不支持 SES-Adapter 方法（因缺乏结构序列信息）。

- 可选择 Freeze 方法，仅微调分类头，或采用 LoRA 方法进行高效参数微调。

### 3.5 开始训练

- 点击 Preview Command 预览命令行脚本。

- 点击 Start 启动训练，Web 界面会显示模型的统计信息和实时训练监控。

- 训练完成后，界面会展示模型在测试集上的 Metrics，用于评估模型效果。

## 4. 评估（Evaluation Tab）

### 4.1 选择模型路径

在 **Model Path** 选项中，输入训练完成的模型路径（`ckpt` 根目录下）。确保选择的 **PLM** 和 **method** 与训练时一致。

### 4.2 评估数据集加载规则

- 评估系统会自动加载相应数据集的测试集。
- 若找不到测试集，则按照 **验证集 → 训练集** 的顺序加载数据。
- 上传到 Hugging Face 的自定义数据集：
  - **若仅上传单个 CSV 文件**，评估系统会自动加载该文件，不受命名影响。
  - **若上传训练集、验证集和测试集**，请确保文件命名准确。

### 4.3 启动评估

点击 **Start Evaluation** 进行评估。

> **示例模型**  
> 本项目提供了一个已经在 **Demo_Solubility** 数据集上使用 **Freeze** 方法训练的模型 **demo_provided.pt**，可直接用于评估。

## 5. 预测（Prediction Tab）

### 5.1 单序列预测（Sequence Prediction）

输入单个氨基酸序列，即可直接进行可溶性预测。

### 5.2 批量预测（Batch Prediction）

- 通过上传 CSV 文件，可批量预测蛋白质的可溶性，并下载结果（CSV 格式）。

## 6. 下载（Download Tab）

有关 **Download Tab** 的详细使用说明和示例，请参考 **Manual Tab** 中的 **Download** 章节。