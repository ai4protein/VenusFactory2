# 初始化、安装与健康检查

统一入口为仓库根目录的 `initialize.py`，实现放在 `scripts/bootstrap/`。
开源工具安装清单是 `configs/bootstrap.json`；Factory 垂类模型使用独立的
`ckpt/manifest.json`，记录文件大小、SHA256 和下载分组。

## 一键完整安装

```bash
python3 install_full.py             # 完整安装并检查
python3 install_full.py plan        # 只查看完整计划
python3 install_full.py doctor      # 检查同一套完整配置
```

完整入口会准备平台环境、内置能力、前端、科学技能、四种 Harness、独立模型工具目录、
Factory 全部预测任务权重及其五种底座，以及 RSI 研究成果库与案例浏览。
仍使用统一的环境检测、进度条、校验和重试机制。需要许可或缺少 GPU 的工具会单独列出，
返回未完成状态；其余独立组件继续安装。安装后通过 `scripts/serve.py` 启动服务。
Factory 任务权重与底座快照分别验收，部分 Hub 缓存不算完整底座；
`doctor --verify-hashes` 还会校验 Hub 提供的底座文件 SHA256。

数据库检索、网络搜索、公开蛋白服务器、会话文件、基础分析、沙箱接口和手册是**平台自带能力**，
不列入可选模型插件。gget 的隔离依赖环境也随核心平台安装。
Tavily/SerpAPI 的 key、InterProScan 联系邮箱在 Administration 配置；安装接口不等于已配置远端服务凭据。

| 可选组件 | 内容 |
| --- | --- |
| `external` | 独立模型/工具运行时：ProteinMPNN、LigandMPNN、ProSST、ProtSSN、对接、ADMET 等 |
| `factory` | Factory 自有训练数据得到的任务模型及所需底座权重 |
| `rsi` | Scientist 产出模型库的发现和调用，保留审核与发布流程 |
| `harnesses` | Kimi Code、DeepSeek Harness、Codex、Claude Code |

```bash
# 只安装核心平台及内置能力
python3 install_full.py --components none
# 核心平台 + 选择的工具和 Harness
python3 install_full.py --components external,harnesses --tools proteinmpnn,ligandmpnn --harnesses codex
# 核心平台 + Factory 模型 + 已有 RSI 研究成果库
python3 install_full.py --components factory,rsi --rsi-library /path/to/reviewed-library
```

组件参数决定安装哪些内容，运行时启停继续由 Administration 管理，安装不会覆盖禁用开关。
首次 RSI 库可为空；模型由 Scientist 产出或接入已有库，仍经过现有独立审核和证据校验后发布。
新部署默认库在 `runtime/local/models/rsi/library`，已有研究库与显式
`VENUS_SCIENTIST_MODEL_LIBRARY` 配置保留。`--rsi-library` 挂接现有库，不复制全部模型权重。

## 推荐用法

准备 Linux（或 WSL2）、Git、Python 3.10+ 和 HTTPS 网络访问。平台环境由脚本创建为
Python 3.12。无需 sudo，不修改系统 Python 或 shell 启动文件。

```bash
python3 initialize.py plan
python3 initialize.py install
.venv/bin/python initialize.py doctor --verify-hashes
```

不带参数等同于 `plan`，只查看计划。默认 **minimal** 只准备 Python 小型环境，
不下载 Torch、模型权重、外部仓库、Harness、Node 或第三方技能。这个基础环境
还不能启动完整 Web 服务，后续按需追加：

```bash
# 安装完整 Web/API/MCP 依赖并构建前端，暂不安装模型集合
python3 initialize.py install --dependencies platform --stages environment,frontend
# 每个选择只安装对应组件，可单独重复执行
python3 initialize.py install --harnesses deepseek-harness,codex
python3 initialize.py install --tools proteinmpnn,ligandmpnn
python3 initialize.py install --factory-preset demo
python3 initialize.py install --stages skills
```

完整平台依赖目前仍包含 Torch/PyG，只有显式选择才安装。大型模型和依赖可能占用数十 GB；
计划中的 Factory 大小只包含所选垂类权重，不虚报所有上游依赖的总大小。
新克隆建议 `git clone --depth 1`，不加 `--recurse-submodules`；第三方技能后续单独安装。
下载内容进入忽略目录，不混进源代码。脚本不会重写已有 Git 历史、提交或推送。

自动选择会优先复用 uv，其次复用 Conda，两者都没有才引导安装 uv；也可显式指定
`--manager uv/conda`。需要安装 uv 时，从官方 PyPI wheel 校验并提取到项目自己的引导目录，
不依赖系统 Python 预装 pip/venv。也可以使用 Conda：

```bash
python3 initialize.py install --manager conda
```

会复用找到的 Conda；没有时下载官方 Miniforge 并校验 SHA256，安装在项目内。
两种方式都会把依赖冲突的工具拆分到各自环境。只有驱动声明支持 CUDA 12.8 以上时，
`--torch-type auto` 才自动选择 cu128，否则使用 CPU；可显式传入 `cpu` 或 `cu128`。
完整服务安装面向 Linux，macOS/原生 Windows 可查看计划，运行服务建议使用 Linux/WSL2。

## 按需安装

| 配置 | 内容 |
| --- | --- |
| `minimal` | 默认：只准备 Python 小型环境 |
| `standard` | 平台、内置能力、前端、技能、RSI 接口 + ProteinMPNN、LigandMPNN、S4PRED + Factory predict-core |
| `extended` | standard + AntiBMPNN、ProSST-2048、ESM-1v、ESM-IF1、ProtSSN、ADMET-AI、Vina |
| `--tools a,b` | 未指定 stages 时只安装这几个工具 |
| `--harnesses deepseek-harness,codex` | 只安装指定 Harness |
| `full` / `install_full.py` | 完整平台及四类可选组件 |
| `--dependencies platform --stages environment,frontend` | 准备完整 Web 环境，不下载模型集合 |
| `--stages ...` | 显式选择 environment、builtins、frontend、harnesses、tools、factory、rsi、skills；完整平台始终带内置能力 |

```bash
# 单独安装两个工具，各自创建新环境
python3 initialize.py install --stages tools --tools proteinmpnn,ligandmpnn --torch-type cpu --factory-preset none

# 复用已有平台环境，不向该环境安装或升级包
python3 initialize.py install --environment /absolute/path/to/env --use-existing-environment

# 下载全部 Factory 垂类模型；按所选模型补充底座
.venv/bin/python initialize.py install --stages factory --factory-preset predict-all --backbone facebook/esm2_t33_650M_UR50D

# GPU 工具环境与权重；完成真实任务验证后由管理员启用
.venv/bin/python initialize.py install --stages tools --tools boltzgen,diffdock --factory-preset none

# 检查正在运行的 API、MCP，不启动新服务
.venv/bin/python initialize.py doctor --service-url http://127.0.0.1:7862/api/runtime-config --service-url http://127.0.0.1:8084/health
```

`--factory-preset` 支持 none、demo、predict-core、predict-all、all，以及兼容旧下载器的
proteinmpnn。`predict-all` 指各类任务头/adapter，不等于所有底座都已下载；底座通过
可重复的 `--backbone HuggingFace模型ID` 指定。`--tools all` 会包含需要许可或人工配置的项，
有未完成项时返回未完成状态，不会宣称全部安装成功。

前端使用 Node.js **24 LTS** / npm 11；Python 兼容版本记录在
`requirements/bootstrap-constraints.txt`，避免 MCP/FastMCP 或 Torch/PyG 版本不匹配。
安装同时检查 `pip check` 和关键包的实际导入。

官方 Hugging Face 无法访问时，可显式增加 `--hf-endpoint https://hf-mirror.com`。
镜像下载仍按同一份 Factory SHA256 清单校验，不会自动换源，也不发送 HF 凭据。

## 验证的含义

Harness 独立放在 `runtime/local/harnesses/<名称>/`，平台自动发现安装回执。
DeepSeek 使用官方 `deepseek-harness-sdk`，不是 PyPI 上另一个同名 `deepseek-harness` 包；
Codex/Claude 使用官方 npm 包的原生执行文件，Kimi 使用官方 release。
版本固定在 `scripts/bootstrap/harnesses.py`。显式 `VENUS_KIMI_BIN`、`VENUS_CODEX_BIN`、
`VENUS_CLAUDE_BIN`、`VENUS_DSH_BIN`、`VENUS_DSH_SITE` 配置优先。
安装只检查 CLI/SDK，不调用 LLM、不配置密钥，也不会改变免费模型路由和管理员禁用策略。

Kimi 适配层支持已验证的 0.19.2 和 2.5.2 协议，可用 `VENUS_KIMI_BIN` 指定管理员安装的
执行文件；未支持的版本会明确拒绝运行。2.5.2 在会话沙箱内启动带鉴权的 Web 服务。
安装其他版本不会替换已经运行的服务。

ProteinMPNN 下载 vanilla、soluble、CA 三类权重；LigandMPNN 下载设计和侧链 packing 权重；
AntiBMPNN 使用官方 Zenodo 权重。三个工具执行小规模序列设计，S4PRED 对 76 个残基做实际预测，
ADMET-AI 执行乙醇预测。ProSST、ESM、ProtSSN 记录模型加载验证，不能当作完整科学评测。

Vina、BoltzGen、DiffDock 的新安装只获得 CLI 验证，不自动启用或写成“推理通过”。
BindCraft 需要单独许可的 PyRosetta 和有界 GPU 验证；SignalP 6 仍需 DTU 安装包及 fast 权重。
InterProScan、BLAST、Foldseek 使用在线服务，不下载大型数据库。具体支持范围和官方来源见
[完整安装说明](installation.md)与[模型运行时说明](model-runtimes.md)。

## 目录、重复运行与排障

- `.venv/`：平台环境；`runtime/local/model-runtimes/<工具>/`：外部源码、独立环境、权重和安装回执。
- `ckpt/`：Factory 垂类权重；底座缓存归对应模型运行时管理。
- `runtime/local/bootstrap/`：初始化器、计划、日志、结果、进程锁。
- `runtime/local/harnesses/`：按需安装的 Harness、SDK 及版本回执。
- `runtime/local/builtin-tools/`：内置服务的独立依赖；`runtime/local/components/`：组件回执及 RSI 库选项。
- `runtime/local/models/rsi/library/`：新安装的 RSI 模型库，已有研究库不自动搬迁。
- `configs/model_runtimes.json`：管理员运行配置，原有路径、限额、禁用开关都保留。
- local 和 online 可共享只读模型资产，会话、队列及产物仍分别放在各自 runtime 目录。

重复执行会复用已完成的安装；校验失败的已有文件、来源不符的 Git 目录、非本初始化器创建的环境，
均不会被直接覆盖。先检查日志，把问题文件移到备份位置，再执行原命令。中断下载不会被当成完整权重。
初始化器不会执行 git reset/clean，也不会删除原有服务环境。

终端显示阶段序号、工具/Harness 数量、包管理器耗时；HTTP 下载显示字节进度，已知总大小时
显示百分比、速度和预计剩余时间。SSH/CI 输出为间隔更新的普通文本。
`--json` 让 stdout 保持 JSON，`--no-progress` 关闭进度输出；完整日志保存在 install.log。
有校验和的中断下载会保留分片，服务器支持 Range 时续传，否则重新下载；校验失败不会成为正式权重。

报告保存在 `runtime/local/bootstrap/<时间>/result.json`，同时有 `plan.json`、`install.log`；
`latest.json` 指向最近一次报告。`doctor --report path.json` 可另存检查结果。
退出码 0 表示所选检查完成，1 表示仍有失败/未完成项，2 表示参数错误或中断。
`doctor` 不下载模型、不启动任务、不调用 LLM；仅指定 `--service-url` 时访问对应健康接口。
它会区分依赖/文件检查、历史推理记录及当前是否启用。

API key 仍通过管理员界面或 `.env` 单独配置。初始化器不生成凭据、不启用付费模型回退。
Bubblewrap 显示可用仅代表找到程序，命名空间、资源限制及隔离能力仍需沙箱运行测试。

## 启动

```bash
.venv/bin/python scripts/serve.py --mode local --port 7862 --mcp-port 8084
.venv/bin/python scripts/serve.py --mode online --port 7863 --mcp-port 8085 --api-workers 2
```

API、MCP、worker 独立启动使用 `--component api/mcp/workers`，详见
[多进程与异步作业说明](architecture/services.md)。初始化和运行服务分别由两个入口负责。
