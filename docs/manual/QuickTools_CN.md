<!-- Compatibility view; edit zh/models.md -->

# 模型广场

## 按功能找模型

打开 Model Plaza，按功能分类浏览，或搜索模型名、蛋白性质和任务。广场是统一目录，不再分 Quick/Advanced。它包含平台预测器，以及当前部署可用的已发布自定义模型或 Scientist 能力。

功能可包括突变评分、蛋白或残基性质、序列设计、结构工具、小分子 ADMET 等。点击卡片查看输入与参数。是否可运行取决于工具开关、权重和依赖；目录中展示不代表每个部署都能运行。

## 直接运行工具

1. 打开与生物问题匹配的模型。
2. 提供所需的序列、结构、表格或小分子输入。
3. 检查可调整参数；链选择、采样等模型特有选项可在高级设置中查看。
4. 提交一次后跟踪作业，检查输出文件，保存作业号便于追踪。

在线版会保留禁用模型的卡片，悬浮提示说明不可用原因。换用另一种 Chat 模式不能绕过管理员的工具开关。

## 自定义模型和工作流

通过训练、预测和评估入口使用自己的模型。完成训练不会自动公开私有模型，共享发布需满足审核和可见性设置。

重复的多步分析可进入[工作流](./zh/workflows.md)，连接输入输出兼容的工具。也可以让 Agent 搜索合适模型并查看适用范围。

## 排行榜

模型广场包含 VenusX 和 ProteinGym 入口或链接。解读排名前应比较任务与数据集，榜单排名不保证模型在你的蛋白或实验条件下同样有效。

## 模型从哪里来

- **预训练方法**：ProSST、ProtSSN、ESM-1v、ProteinMPNN、LigandMPNN 使用上游方法和权重。
- **平台训练模型**：例如 **DeepSol · ESM2-650M** 表示在特定数据集上训练的预测模型及其骨干；相同骨干在不同数据集上会产生不同模型。
- **Scientist 模型**：RSI 开发的新模型或可复用能力。使用前查看适用范围、评估报告和审核状态；私有研究成果通过发布审核后才会公开。

按功能和关键词寻找模型，按类型与来源理解模型的产生方式。需要帮助时，可以让 Agent 搜索当前可用工具并查询同一份手册。

## 安装模型

本地完整安装使用 `python3 install_full.py`；`--components external,factory,rsi,harnesses` 选择四类扩展。数据库/网络搜索和公开蛋白服务是内置能力。`--rsi-library` 可接入已有审核模型库。

`python3 initialize.py install` 默认只准备小型 Python 环境；`--dependencies platform --stages environment,frontend` 准备 Web 服务；`--tools proteinmpnn,ligandmpnn`、`--harnesses deepseek-harness,codex`、`--factory-preset demo` 分别只安装对应组件。`doctor` 搭配同样的组件参数检查安装。下载和环境放在独立忽略目录，保留管理员开关。
