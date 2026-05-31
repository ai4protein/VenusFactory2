# VenusFactory2 常见问题解答 (FAQ)

## 安装与环境

### Q1: 怎么安装 VenusFactory2？

参考 `README.md` 的 **Installation** 章节。支持三条路径：

1. **conda + pip**（多数用户的默认路径）：
   ```bash
   git clone https://github.com/AI4Protein/VenusFactory2.git && cd VenusFactory2
   conda create -n venus python=3.12 && conda activate venus
   pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
       -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
   pip install -r requirements.txt
   ```

2. **uv**（更快，开发用）：`python install.py --type cu128` 后 `source .venv/bin/activate`。

3. **Docker**：`cp .env.example .env && docker compose --profile gpu up -d --build`。

用 `python scripts/check_env.py` 校验。

### Q2: 安装时报 "Could not find a specific dependency"。

按顺序尝试：

1. 单独装报错的包：
   ```bash
   pip install <name>
   ```
2. 如果是 CUDA 相关，确认 torch 和 CUDA 版本匹配。VenusFactory2 默认是 **CUDA 12.8 + torch 2.8.0**：
   ```bash
   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   ```
3. 某些包需要系统库。Ubuntu 24.04 上最常见的是 `pycairo` 构建依赖（通过 `xhtml2pdf` 引入做 PDF 导出）：
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libcairo2-dev libxml2-dev pkg-config
   ```
   或者用 conda：`conda install -c conda-forge pycairo` 直接装预编译二进制，跳过源码编译。

### Q3: 怎么检查 CUDA 装好了？

1. 驱动：`nvidia-smi`（驱动需支持 CUDA 12.8 → 驱动 ≥ 555.x）。
2. PyTorch：
   ```python
   import torch
   print(torch.__version__)            # 应为 2.8.0+cu128
   print(torch.cuda.is_available())    # True
   print(torch.cuda.device_count())    # 可见 GPU 数
   print(torch.cuda.get_device_name(0))
   ```
3. 端到端：`python scripts/check_env.py` 跑上面的所有 import 加一个 CUDA matmul 烟测。

如果 `cuda.is_available()` 是 False，多半是 torch wheel 没带 CUDA。按 Q2 用正确的 `--index-url` 重装。

## 硬件与资源

### Q4: 训练时报 "CUDA out of memory"。

按效果排序：

1. **降低 batch size** — 最直接。`Batch Size`（或 `Tokens per Batch`）减半。
2. **换小模型** — 比如从 `ESM2-650M` 换到 `ESM2-150M` 或 `ESM2-35M`。ProSST / VenusPLM 也是轻量选择。
3. **用 PEFT 方法** — `plm-lora`、`plm-qlora`、`plm-dora`、`plm-ia3` 训练参数少得多。
4. **开梯度累积** — `gradient_accumulation_steps` 设为 2 / 4 / 8，可以在保持有效 batch 的同时省显存。
5. **降低 `Max Seq Length`**（如果数据允许，砍掉信号肽 / 长无序尾段）。

### Q5: batch size 选多大？

1. **从小往大。** 4 或 8 起步，逐步加大直到接近显存上限。
2. **参考范围。** 蛋白 PLM 常用 16–64，但严重依赖显存和序列长度。
3. **取舍。** 大 batch 梯度更稳，但可能需要更高的学习率。
4. **OOM 规则。** 先减半，再想其他调参。

## 数据集

### Q6: 怎么准备自定义数据集？

在 **Custom Model → Training** 页面：

1. **列。** 至少一个序列列（默认 `aa_seq`）和一个标签列（默认 `label`）。用 `ses-adapter` 还要 `foldseek_seq` 和/或 `ss8_seq`。回归用数值标签，分类用整数（单标签）或列表（多标签）。
2. **切分。** 三个文件 — `train`、`validation`、`test` — 分别上传，或者用带这三个 split 的 HuggingFace 数据集。
3. **HF 路径。** 用 `username/dataset_name` 引用。
4. **配置。** 在页面里设置 Problem Type、Num Labels、Metrics；或者选 Pre-defined 数据集自动填充。

### Q7: 数据集上传报格式错误。

常见原因：

1. **列名错。** 确保序列列叫 `aa_seq`（或表单里你设的名字），标签列叫 `label`。
2. **序列字符不合法。** 只能是 20 个标准 AA 字母（`ACDEFGHIKLMNPQRSTVWY`），未知用 `X`。去掉空白、换行和其他字符。
3. **编码。** 存成 UTF-8。
4. **分隔符。** CSV 用逗号，TSV 用 tab — 文件名要对得上。
5. **缺失值。** 删掉缺序列或标签的行。

### Q8: 数据集很大，系统加载慢 / 崩溃。

1. **先小子集验证。** 1–5k 行先跑通流水线。
2. **Batch Token Mode** — 序列长度差异大时比固定 batch 打包更高效。
3. **离线预处理。** 删无用列、去重、切分多个文件。
4. **加内存。** 机器有富余 RAM 就调高 `num_workers`；如果在 swap 就调低。

## 训练

### Q9: 训练中断了怎么续？

训练保存路径 = **Save Directory** + **Output Model Name**（默认 `ckpt/demo/best_model.pt`）。

1. 在 Training 页面，把 **Training Mode** 从 `From Scratch` 切到 `Continue Training`。
2. 选包含上一个 checkpoint 的 **Model Folder**。
3. 在下拉里选 **Checkpoint** 文件。
4. 点 Start — 训练从该 checkpoint 的 epoch 和优化器状态恢复。

> 系统会按你的 monitored metric 保留"目前最优"checkpoint。基于步数的定期快照默认不开。

### Q10: 训练太慢。

1. **用 PEFT 方法**（`plm-lora`、`plm-qlora`）— 可训参数少几个数量级。
2. **降低 `Max Seq Length`**（任务允许的话）。
3. **换小 PLM** — 分类任务上 ESM2-150M 通常只比 ESM2-650M 差 1–2 个点，但快得多。
4. **数据放 SSD** — 带 PDB 的蛋白数据集常常是 I/O 瓶颈。
5. **`Batch Token Mode`** 处理变长数据 — 比固定 batch 更利用 GPU。

### Q11: loss 不降 / 出现 NaN。

loss 不降：

- **学习率过高** — `full` 微调时试 `1e-5` 而不是 `5e-4`。
- **优化器不对** — full 微调通常用 AdamW。
- **数据问题** — 检查标签噪声、错标样本、off-by-one 索引。

NaN：

- **梯度爆炸** — **Max Grad Norm** 设为 1.0–5.0。
- **学习率过高** — 降低 10×。
- **fp16 数值不稳** — 怀疑下溢就换 fp32。
- **数据异常** — 回归标签的极端值可能产生 NaN；截断或归一化。

### Q12: 怎么避免过拟合？

1. **更多数据 / 数据增强。**
2. **正则化** — dropout（0.1–0.3）、weight decay，或用 `Patience` 早停。
3. **更小模型** — 参数少，或用 `freeze` 锁住 PLM。
4. **交叉验证** — 训多折取中位数。

## 评估

### Q13: 该看哪个评估指标？

| 任务 | 默认关注 |
| :--- | :--- |
| 类别均衡的分类 | **Accuracy**、**F1** |
| 类别不均衡的分类 | **F1**、**MCC**、**AUROC** |
| 多标签分类 | **F1_max**、各标签 AUROC |
| 回归 | **Spearman_corr**、**MSE** |

*最重要* 的指标取决于下游用途。药物筛选可能优先真阳率；做候选排序则用 Spearman。

### Q14: 评估结果很差，怎么改进？

1. **数据质量** — 检查标签噪声、训练 / 测试分布偏移。
2. **模型 + 方法** — 换 PLM，或从 `freeze` 换到 `plm-lora`。
3. **更多特征** — 结构感知方法（`ses-adapter`、ProSST、ProtSSN）在结构依赖任务上通常优于仅序列。
4. **集成** — 训 3–5 个种子取平均。

### Q15: 测试集表现远差于验证集。

1. **分布偏移** — 测试集有训练时未见的家族 / 性质。用分层切分。
2. **过拟合到验证** — 反复用 val 选模型相当于在它上训练。留一份只动一次的测试集。
3. **数据泄漏** — train 和 test 有重复。切分前按序列相似度聚类。
4. **测试集太小** — 换种子重跑看方差。

## 预测

### Q16: 怎么加速预测？

1. **用 Batch 模式** — 在 **Custom Model → Predict** 摊销 GPU 初始化开销。
2. **小模型** — 有时 `ESM2-150M` "够用"了，比 `650M` 快 4 倍。
3. **GPU 而不是 CPU** — 确认 `torch.cuda.is_available()` 是 True。
4. **降低 `Max Seq Length`**（输入允许的话）。

### Q17: 预测结果与预期差很多。

可能原因：

1. **模型 / checkpoint 选错** — Predict 会从 checkpoint 锁定 PLM / 方法 / 池化，确认选对了。
2. **out-of-distribution 序列** — 输入比训练数据长得多 / 短得多，或来自不同物种 / 家族。
3. **缺结构输入** — `ses-adapter` / ProSST / ProtSSN / SaProt 模型必须提供结构侧（PDB Dir 或 Foldseek/SS8 文本）。
4. **序列格式问题** — 非 AA 字符、小写、gap、终止密码子。先清洗一下。

### Q18: 怎么高效批量预测大量序列？

去 **Custom Model → Predict** 用 **Batch** 模式：

1. **准备输入文件** — CSV/TSV/XLSX，至少包含：
   - `aa_seq` — 氨基酸序列
   - `id` / `name` — 可选标识
   - `foldseek_seq` / `ss8_seq` — 仅当模型为 `ses-adapter` 且启用对应结构序列时需要
2. **加载模型** — 选 **Model Folder** + **Model Path**（保存的 config 会锁定 PLM、方法、池化）。
3. **切换到 Batch 模式** — 选 `Upload file`（浏览器上传）、`Paste FASTA`、或 `Path`（指向服务器上已有的文件 — 超大列表最快）。
4. **设置 Batch Size** — 16–32 是个好默认。长序列 OOM 就降低；显存有富余就增加。
5. **Start** — 页面实时显示进度条和预测日志尾部。
6. **结果 CSV** — 含每条输入样本和预测列；从结果面板下载。

## 模型与结果

### Q19: 该选哪个预训练模型？

| 场景 | 推荐 |
| :--- | :--- |
| 通用、计算与质量平衡 | **ESM2-650M** |
| 显存受限（<8 GB） | **ESM2-8M / 35M / 150M**、**ProSST**、**VenusPLM**、**PETA** |
| 长上下文 / 生成 | **ProtT5-XL** |
| 结构感知（有 PDB） | **ProSST-2048**、**ProtSSN**、**SaProt**、**VenusREM** |
| 抗体序列 | **IgBert**、**IgT5** |
| 最大、可用多卡 | **ESM2-15B**、**ProtT5-XXL** |

选择时考虑：

- **数据量：** 训练数据有限时，小模型常常泛化更好（过拟合风险小）。
- **序列长度：** 极长蛋白优先用原生支持长上下文的模型。
- **资源：** 小 PLM + PEFT 方法（如 `plm-lora`）通常是资源 / 质量最优解。
- **任务类型：** 结构感知模型对结构依赖任务（结合、稳定性）有帮助；纯序列模型对溶解性 / 定位足够好。

拿不定主意就训 2-3 个候选，看验证集表现选。

### Q20: Training 页面的 loss 曲线怎么读？

Training 页面会实时显示 **Train Loss**、**Val Loss**、**Val Metrics** 曲线。

| 模式 | 可能含义 | 怎么处理 |
| :--- | :--- | :--- |
| 两条 loss 都下降并干净收敛 | 健康训练 — 让它跑完 | — |
| Train ↓ / Val ↑ | 过拟合 | 加大 dropout、weight decay；降低 `Num Epochs`、缩小 `Patience`、换小模型 |
| 两条都停在高位 | 欠拟合 | 提高学习率、换大模型、加 epoch |
| 曲线疯狂震荡 | 学习率太高 | 学习率降 5–10×，`Max Grad Norm` 设 1.0–5.0 |
| Val < Train | 通常正常（dropout / 数据切分效应）；偶尔提示切分污染 | 检查 train 和 val 是否真的不相交 |
| 突然飙升然后 NaN | 梯度爆炸 | 设 `Max Grad Norm`、降学习率、检查极端标签 |

验证指标在 epoch 上限前停止改善时，**Patience** 早停会自动结束训练。

### Q21: 怎么保存和分享训好的模型？

模型保存到 `Save Directory / Output Model Name`（默认 `ckpt/demo/best_model.pt`）。目录里包含：

| 文件 | 用途 |
| :--- | :--- |
| `*.pt` | 模型权重（你训练的那个） |
| `config.json` / `adapter_config.json` | 运行配置：PLM、方法、池化、问题类型、标签数、LoRA 参数 — **Custom Model → Evaluation / Predict 会读回这个** |
| tokenizer 文件 | 继承自基础 PLM |

**分享方式：**

1. **Hugging Face Hub** — 最简单。创建一个模型仓库、上传文件夹、在 model card 里说明训练数据、架构、指标和一个使用示例。
2. **本地导出** — `tar -czf my_model.tar.gz ckpt/demo/`。同时告诉对方基础 PLM 是哪个 + 训练方法，他们就能在 **Custom Model → Predict** 里同样使用。
3. **写清楚文档** — 训练数据来源、超参数、验证 / 测试指标、预期用途、已知局限。

## 界面与操作

### Q22: WebUI 卡 / 页面崩溃。

1. **浏览器：** Chrome / Edge 对 React + Molstar viewer 兼容性最好。清缓存、禁用重插件。
2. **资源：** 确认机器有空闲 RAM。关掉其他吃 GPU 的程序。远程服务器看 `top` / `nvidia-smi`。
3. **网络：** 远程部署时不稳定的 SSH 隧道或反向代理会导致 API 超时。用 `curl http://<host>:7861/health` 测试。
4. **重启：** 杀掉 `python src/webui_v2.py` 再起。Docker：`docker compose --profile gpu restart`。
5. **重 build 前端：** 部分页面渲染、部分卡在 spinner，可能是前端 bundle 过期。重 build（`cd frontend && npm run build`）。

### Q23: 训练过程中途无响应。

最常见原因：

1. **OOM kill：** Linux OOM killer 杀掉了 Python 进程。看 `dmesg | tail -30` 是否有 `Killed process`。修复：降 batch size、换小 PLM、用 `plm-qlora`。
2. **GPU OOM 没崩但卡住：** 少见但可能（非阻塞 CUDA 错）。`nvidia-smi` 会显示 0% 利用但显存不释放。重启进程。
3. **浏览器断开：** UI 显示"stopped"但后端可能还在跑。训练进程不受影响 — 看 `ckpt/` 里最新 checkpoint。
4. **网络 / SSH 隧道断：** 通过 SSH 启动而没用 `tmux` / `nohup` / `screen`，shell 一死进程就被杀。长任务务必在 `tmux` 里跑。
5. **API 限速**（训练用了外部服务如 W&B）：运行可能卡在 API 等待。关掉 W&B 开关或检查 `wandb` 状态。

以上所有情况，最佳护栏都是 **Continue Training** 模式 — 一键从上一个 checkpoint 重启。

## Agent / Chat

### Q24: Agent 在 plan 中途停了。

- 看右栏 **Execution Status**。红色状态通常指向失败的工具。
- 最常见的原因：缺 API key（在 **Settings** 或 `.env` 里设）、provider 限速、或工具内存不足。
- 失败步骤可以重跑：在 iteration 检查点用 **Modify & Re-execute**，或编辑 plan 后继续。

### Q25: Online 模式下 Agent 配额用完了。

线上部署会强制按用户每日对话配额（输入框附近的配额胶囊可看）。用完后：

- 等次日重置，或
- 用 **Local 模式** + 你自己的 API key — 没有配额。

### Q26: 我自定义的 OpenAI 兼容模型不见了。

- 自定义模型**仅 Local 模式**有效，存在浏览器 `localStorage`（`vf2_custom_openai_style_models`）里。
- 清过网站数据的话，在 chat 页模型选择器里重新添加。
- Online 模式下设计上就会过滤掉。

## WebUI / 部署

### Q27: WebUI v1 和 v2 区别？

- **v1**（`python src/webui.py`）：旧版 Gradio 界面，端口 7860。
- **v2**（`python src/webui_v2.py`）：当前的 FastAPI + React 界面，端口 7861 — 所有手册描述的就是这个。

新部署默认用 v2。v1 保留是为了依赖旧 Gradio MCP 集成的用户。

### Q28: v2 起来了但页面空白。

需要先 build React 前端：

```bash
cd frontend && npm install && npm run build && cd ..
python src/webui_v2.py --host 0.0.0.0 --port 7861
```

没有 `frontend/dist/`，v2 没东西可以渲染。

### Q29: v2 起来了但页面调用不到 API。

检查 host 和端口：

- `--host 0.0.0.0` 接受外部连接（默认就是 `0.0.0.0`）。
- 默认端口 `7861`。被占用就用 `--port` 或 `VENUS_PORT`（Docker）覆盖。
- 通过反代外部访问时，在 `.env` 里把代理 URL 加到 `WEBUI_V2_CORS_ORIGINS`。

### Q30: 怎么完全重置环境？

```bash
# conda 路径
conda deactivate
conda env remove -n venus
conda create -n venus python=3.12
# 然后按 Q1 重新装

# uv 路径
rm -rf .venv
python install.py --type cu128
```

Docker：`docker compose --profile gpu down -v && docker compose --profile gpu up -d --build`。
