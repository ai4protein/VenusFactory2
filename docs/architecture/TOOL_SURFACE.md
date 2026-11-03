# 工具边界与精简记录

## 当前调用边界

模型负责决定步骤和编写源代码；MCP 提供受控操作；worker/资源队列负责执行与预算；Bubblewrap 是会话代码的默认隔离后端。沙箱本身不会调用另一个大模型生成代码。

```text
Agent / Scientist / Researcher
  ├─ 已注册科学工具 → 权限检查 → 队列 → 专用 worker / 外部服务
  ├─ 会话文件工具 → 当前会话目录
  └─ 会话代码执行 → CPU 槽位 → 隔离沙箱 → 当前会话产物
```

会话代码只挂载服务器分配的目录。MCP 参数不能覆盖会话、挂载、后端或控制目录。沙箱无网络、无宿主凭据；执行有 CPU、内存、超时和输出限制，无宿主执行回退。搜索与下载使用已注册的联网工具。

## 渐进式工具披露

Web Agent 与 Scientist 的原生 Harness 共用一个发现入口，不将整份科学工具说明和参数写入每次模型请求：

1. `search_tools(query, limit, offset)` 只返回工具名称和简短用途。默认 12 项，最多 20 项，每项说明最多 180 字符；支持分页，不返回参数或长文档。
2. `describe_tool(name)` 按需读取一个工具的完整说明和输入参数。
3. `call_tool(name, arguments)` 调用本轮已经查看过接口的工具。底层继续使用原工具的权限检查、参数验证、会话绑定、队列和执行预算。

Kimi Code、DeepSeek Harness、Codex、Claude Code 使用同一协议。`ask_user`、`request_confirmation` 直接保留；Scientist 还直接保留 `submit_research_checkpoint`，方便保存进度。PI 和规划角色只收到少量相关名称与用途，不再重复注入完整目录。

禁用的工具不进入搜索结果，查看和执行时再次检查权限。包装后的调用仍按原始科学工具名称记录执行状态与用量。Scientist 注册模型通过模型注册表继续二级搜索和按需加载，不把大量模型逐项展开为顶层工具。

实现位于 `src/agent/harness/tool_disclosure.py`。回归测试覆盖一万项模拟目录的固定初始接口、分页、禁用后拒绝执行、四种 Harness 的共享调用协议，以及真实会话沙箱的目录隔离。

## 接口归并

| 能力 | 当前公开接口 | 处理 |
| --- | --- | --- |
| 查看文件 | `list_session_files`、`read_session_file` | Agent、Scientist 共用实现和参数 |
| 编写代码和报告 | `write_session_file(path, content)` | 当前模型直接提供完整内容 |
| 运行 Python | `run_session_python(path, timeout_seconds)` | 运行已经保存的脚本，返回退出码、输出和源文件哈希 |
| Shell 分析 | `run_session_shell(command, timeout_seconds)` | 仅在当前运行明确提供时可调用，仍在会话沙箱内 |
| 旧代码生成工具 | `agent_generated_code` | 从总表、MCP、FastAPI 和训练工具表移除；旧导入返回迁移提示，不生成代码、不执行 |
| 宿主 REPL | `python_repl` / `Python_REPL` | 不注册、不实例化，保留拒绝策略 |
| 训练配置 | `generate_training_config` | 当前模型提供 `parameters`；工具根据数据和默认值确定性生成配置，不再在工具内部请求 LLM；返回解析后的完整参数 |
| 历史研究文件名称 | `*_research_file`、`list_research_files`、`run_research_python` | 内部方法及历史 receipt 保留；新 MCP 使用统一会话名称 |

旧名称下的管理员禁用设置仍然有效。旧、新设置冲突时保持禁用，管理员明确保存后才统一到新名称；不因改名恢复权限。

训练配置的旧 `user_requirements` 文本参数返回明确的迁移错误，请调用者转换为 `parameters` 对象。不会默默忽略模型选择或训练偏好，也不会退回另一供应商的 LLM。配置报告中的类别数现在来自实际分析结果，移除固定的 19 类占位值。

## 科学工具分类与保留理由

| 类别 | 范围 | 为什么保留专用工具 |
| --- | --- | --- |
| 网络与文献 | Web、Tavily、SerpAPI、PubMed、arXiv、Semantic Scholar | 服务端凭据、来源引用、服务限流；供应商专用入口与统一路由保留兼容，普通检索优先 `query_web` |
| 蛋白数据库 | UniProt、NCBI、RCSB、InterPro、STRING、HPA、KEGG、ChEMBL 等 | 标识符解析、格式约定、分页与下载产物；不能假定都能由一段离线代码替代 |
| 远程分析 | BLAST、Foldseek、MSA 等 | 外部任务号、轮询和超时，避免重复提交 |
| 序列计算 | 理化性质、翻译、FASTA 等 | 低成本、已验证的输入输出；序列与文件入口满足不同数据输入方式 |
| 结构计算 | SASA/RSA、二级结构、结构转换和渲染 | 专用依赖和稳定产物格式 |
| 小分子 ADMET | 已注册 ADMET 预测 | 模型权重、适用范围及任务调度 |
| 模型预测与设计 | 平台模型、已审核 Scientist 模型、序列设计 | 模型选择、权重加载、资源限制和可复用接口 |
| 模型训练 | 训练配置、平台 PLM 训练、checkpoint 预测 | 管理平台训练依赖与队列；自定义轻量算法可以在会话沙箱研发 |
| 文件与能力管理 | 技能、模型搜索、接口描述、注册查询 | 面向大量模型动态发现；搜索、描述和执行分别有独立职责 |
| 会话沙箱 | Python、Shell | 仅执行当前会话的分析程序，不承担发布审核 |

Scientist 的策略指南、模型评估、研究检查点、独立 Review 和成果注册继续独立保留。代码成功运行只证明该次执行完成，不等于模型质量通过审核，也不等于可以公开给 Agent 调用。

## 本轮核对

初始静态 hub 有 121 项且无重名；其中 `agent_generated_code` 已被运行策略禁用，但仍出现在注册总表和旧提示中。本轮移除该残留注册，hub 变为 120 项。环境未安装可用的宿主 REPL，因此不将它虚报为额外减少一项在线可用工具。

Agent 和 Scientist 在各自上下文内原本就没有重复的工作区函数；本轮归并的是公开命名、构造代码和管理员条目。领域专用工具没有因为名字接近而被删除。历史研究证据、执行 receipt 和底层离线重放实现不被改写。

验证覆盖真实 MCP HTTP 调用、两个并行会话、Bubblewrap 的目录/网络/凭据隔离、退出清理、旧权限兼容、工作流及 Scientist 模型研发/审核链。测试日志与目录快照保存于 `research_runs/rsi/20261101_tool_simplification/`。
