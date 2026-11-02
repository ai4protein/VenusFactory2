# VenusFactory2 仓库 RSI 改造实施计划

本轮更新：见 [新的 RSI 改造与 GLM 试验方案](/home/tanyang/workspace/VenusFactory2/docs/rsi/RSI_REDESIGN_GLM_20261010.md)。论文题目固定不改；正式比较共用修复后的执行基线；旧 30 题均降为开发诊断材料。新六对 GLM 检查为父策略 3/6、候选 4/6，仅作诊断，候选未晋升。

日期：2026年10月10日。状态：首轮基础代码已实现；已接入 DMXAPI/GLM，六项自主检查完成、30 项工程先导已完成（16/30 通过）；完整 RSI 尚未完成。唯一实施位置为 hl_gpu 的 `/home/tanyang/workspace/VenusFactory2`；代码基线为 `3b891df941335a548e662cc7d5ac129fec94cc4a`。本文规定现有代码怎么改、增加哪些科研功能、各工作包怎样验收。[实验协议](/home/tanyang/workspace/VenusFactory2/docs/rsi/EXPERIMENT_PROTOCOL.md)规定性能验证，[文献与论文改造依据](/home/tanyang/workspace/VenusFactory2/docs/paper/plan/RSI_research_and_paper_revision_20261010.md)解释方法来源。

保留 Kimi Code 执行引擎，新增领域能力层与研究控制层。目标不是让 agent 任意改整个仓库，而是让它能依据科研反馈提出可检验的能力更新，经过独立选择后在下一代使用，并检验改进策略本身是否变好。下列工作包仍按完整目标列出；实际完成范围、代码位置和运行证据见 [首轮实施记录](/home/tanyang/workspace/VenusFactory2/docs/rsi/IMPLEMENTATION_STATUS_20261010.md)，不以文件存在等同于工作包完成。

## 1 已有能力与实际缺口

| 已核查位置 | 可以复用的能力 | 实施时必须解决的缺口 |
| --- | --- | --- |
| [会话路由](/home/tanyang/workspace/VenusFactory2/backend/web_v2/chat_api/_shared.py:205) | Science Agent 走 Kimi；Science Expert 走 graph | 主实验明确使用 Kimi；四角色结构不作为当前执行器的默认描述 |
| [KimiClient](/home/tanyang/workspace/VenusFactory2/src/agent/kimi_client.py:212) | create_session、submit_prompt、事件流、abort_prompt | 抽成无浏览器研究 runner，补完整生命周期、使用量与失败状态 |
| [系统提示构造](/home/tanyang/workspace/VenusFactory2/backend/web_v2/chat_api/_stream_kimi.py:78) | 构建领域提示和注入技能信息 | 提取纯构造函数，增加版本资源输入；不复用整套 UI SSE 状态机 |
| [技能根与缓存](/home/tanyang/workspace/VenusFactory2/src/agent/skills.py:136) | VF 和第三方 skills 的双根加载 | 当前全局根和 maxsize=1 缓存不区分候选版本 |
| [技能挂载](/home/tanyang/workspace/VenusFactory2/src/agent/kimi_skills.py:27) | 建项目级 skills 符号链接 | 实验使用版本快照内的挂载，不能指向另一谱系的可写技能 |
| [领域上下文](/home/tanyang/workspace/VenusFactory2/src/agent/tool_context.py:1) | 序列、结构和生物实体上下文 | 新 RunContext 用于资源根、版本与预算，与现有生物上下文组合，不直接替换 |
| [训练 MCP](/home/tanyang/workspace/VenusFactory2/mcp_tools/tools/train/tools_mcp.py:36) | 配置生成、训练、预测、注册、列模型 | 训练和“允许跨任务使用”需分开；test_csv_file 只允许开发域数据，不能传最终隐藏测试 |
| [训练执行](/home/tanyang/workspace/VenusFactory2/mcp_tools/tools/train/train_operations.py:1793) | 子进程训练，训练后自动注册 | 新研究模式先入 staging；记录并终止派生训练进程；保持既有产品默认行为 |
| [模型目录](/home/tanyang/workspace/VenusFactory2/mcp_tools/tools/train/model_registry.py:29) | 持久保存配置、权重与索引 | 增加私有资源根、原子写、来源和验证状态；同步修路径解析与允许根 |
| [MCP 服务](/home/tanyang/workspace/VenusFactory2/mcp_tools/mcp_server.py:32) | 统一挂载科学工具，训练仅 local 开启 | 每个隔离运行实例绑定对应资源根，不能复用全局可写 registry |
| [Tracing](/home/tanyang/workspace/VenusFactory2/src/agent/tracing.py:480) | trace/span 扩展接口 | 汇总实际事件、科学产物和成本，标注未知 usage，不能将缺失量记为零 |
| [旧 judge](/home/tanyang/workspace/VenusFactory2/src/agent/agent_eval/evaluation.py:24) | 回答 A/B 比较 | 解析失败和异常默认 A；改为显式 invalid/error，主指标改用可执行评价 |

训练工具内部还会额外调用 LLM 生成代码和配置，见 `train_operations.py` 约 394–579 行与 1474–1537 行。因此冻结 Kimi 的主模型仍不等于冻结全系统。基线需同时固定内部生成模型、第三方服务与模型权重，并核算这些调用；研究配置缺失时不得静默落到另一默认模型。

数据核查发现 `data/FLIP_GB1/`、`data/FLIP_AAV/`、`data/TAPE_Fluorescence/`、`data/eSOL/`、`data/DLKcat/`、`data/DeepSol/`、`data/Thermostability/` 顶层现有文件均为 JSON 配置。它们可作为数据接入起点，尚不能证明原始数据已在本机缓存、许可已核查或无泄漏划分已经就绪。

## 2 改造后的运行边界

系统版本定义为 A=(P,S,W,R,U)。P 是执行领域提示，S 是技能与脚本，W 是工具路由和科研工作流策略，R 是具有来源和验证信息的能力目录，U 是产生未来修改的改进策略。另有独立经验库 M，B2 只更新 M，用来分开记忆与机制改进。

冻结部分包括 Kimi 运行时和模型、实验控制器、外部评分程序、隐藏任务、资源上限、候选接受规则及原始工具权限。可演化部分位于独立能力包。第一版使用配置和提示表达 W/U，技能允许有限脚本；第二版可增加可执行插件，但必须仍服从固定控制器。

执行会话使用 E=(P,S,W,R,M) 完成任务，改进会话使用 U 和公开开发证据提出候选。B3 始终装载 U0，B4 下一轮装载已接受的 Ut。U 不是仅保存在日志里的文本，其实际输入策略、失败采样、提案步骤或上下文选择必须发生变化并留下加载记录。

评价器读取任务产物，不能从候选自报的 success 得到结论。外部评价器由单独进程及真正隔离的文件权限或容器控制；仅靠当前 Python sandbox 或 system prompt 不能证明 Kimi 的 shell 看不到答案。共享只读基础权重可以使用内容哈希寻址；版本索引和任务 scratch 必须私有。

## 3 新增功能与使用行为

| 功能 | 触发条件 | 系统应输出或执行什么 | 独立验收 |
| --- | --- | --- | --- |
| 能力缺口诊断 | 用户目标与现有工具能力不匹配，或已有模型适用证据不足 | 结构化说明：可直接调用、需要轻量适配、需要训练、数据不足；保留证据和替代路线 | 在标注好的适用与不适用任务上评价误训率、遗漏率及总成本 |
| 模型适用性检查 | 准备调用持久模型 | 核对任务定义、标签单位、蛋白家族/序列范围、训练数据与验证范围；必要时拒绝复用或回退 | 对域内、近域、远域和不兼容标签任务评价正确复用与拒绝 |
| 验证后的能力注册 | 新模型或脚本训练/生成完成 | 先写 staging，保存训练和验证数据指纹、测量指标、环境和适用范围；通过开发验证才进入可复用目录 | 训练成功但预测 ID 错乱或验证失败的模型不得进入下一代目录 |
| 科研产物检查 | 工具返回结果或提交最终结果 | 检查样本 ID、残基映射、单位、缺失值、约束和数据划分；区分可修复错误与需停止的问题 | 用带已知错误的输出测试检出率、误报和修复结果 |
| 经验提炼 | 开发任务结束或失败 | 生成可追溯经验条目；区别事实、解释与待验证假设；形成带来源的技能候选 | 冻结重放与未见任务表现；不得只用 LLM 给经验打分 |
| 工作流改进 | 多次出现同类失败或高成本路径 | 调整检索、验证、重试、模型选择和训练顺序 | 与固定策略及提示词优化对照，报告成功率与成本 |
| 改进策略演化 | 有候选 U 并通过 meta-probe | 更新下一轮改进的失败采样、证据选择或修改策略 | 同起点新旧改进器产生后代，再由独立留出任务评价 |
| 版本对比与回退 | 版本发布、实验复核或检测退化 | 显示改了什么、证据、成本、来源和回退目标 | 失败发布不污染 incumbent；重启后的 manifest 与文件一致 |

能力缺口诊断和科学检查是拟议的领域方法。其价值必须由实验量化，不因加入功能就声称方法创新。对测不到不确定性的模型标为 unknown，不能伪造置信度或把序列相似性当作校准后的误差概率。

## 4 工作包及代码改动

所有新模块计划放在 `/home/tanyang/workspace/VenusFactory2/src/agent/rsi/`。配置、入口和测试分别位于该仓库的 `configs/rsi/`、`scripts/rsi/`、`tests/rsi/`。P0 表示前置基础，P1 表示完整研究必需，P2 表示后续扩展。

| 工作包 | 优先级及依赖 | 新增或修改 | 交付与完成条件 |
| --- | --- | --- | --- |
| R0 基线与数据清单 | P0，无 | 新 `baseline.py`、`configs/rsi/baseline.yaml`；记录所有内部模型 | manifest 包含 git/runtime/model/tool/skill/data 哈希和非敏感配置；原始数据状态明确 |
| R1 离线执行器 | P0，R0 | 新 `runtime.py`、`specs.py`；复用 KimiClient，抽取 prompt builder | 无浏览器完成任务；事件、产物、失败码、超时和恢复有记录 |
| R2 资源隔离与预算 | P0，R1 | 新 `context.py`、`budget.py`；适配 skills、kimi_skills、registry、MCP 启动 | 两个并行版本互不读写；隐藏标签不可访问；派生训练作业随预算终止 |
| R3 版本与账本 | P0，R1/R2 | 新 `versions.py`、`ledger.py` | 只读快照、候选分支、内容哈希、父子谱系、原子发布和中断恢复 |
| R4 能力诊断与路由 | P1，R0/R1 | 新 `capabilities.py`、`policies/`，增加 W 的机器可读规范 | 可给出调用/适配/训练/停止决策，并记录证据；先用冻结人工规则建立基线 |
| R5 验证与模型复用 | P1，R2/R3/R4 | 适配 train_operations、model_registry；新 `validation.py` | 研究训练不立即晋升；模型卡、适用范围和验证记录可查询；索引写入和解析一致 |
| R6 科学产物和外部评价 | P0/P1，R1/R2 | 新 `artifacts.py`、`evaluation.py`；固定评分实现置于独立评价环境 | schema、ID、单位、残基检查；隐藏指标由外部算；修旧 judge 的错误分支 |
| R7 经验库与技能候选 | P1，R3/R6 | 新 `experience.py`；版本包中的 memory/skills | 失败与成功有来源；可按来源任务、蛋白族和错误类型检索；相同容量的 B2 可运行 |
| R8 固定和递归提案器 | P1，R3/R6/R7 | 新 `proposer.py`、`policies/improver/` | I0/It 只差 U；新 U 真正被装载；记录可检验假设与修改范围 |
| R9 控制器和独立选择 | P1，R3/R6/R8 | 新 `controller.py`、`selection.py` | 预检、开发筛选、选择确认、U probe、原子接受/拒绝；预算从外部生效 |
| R10 复现实验和分析 | P1，R0–R9 | `scripts/rsi/run_experiment.py`、`analyze.py`、`plot_results.py`；任务配置 | 六组主比较、外部方法适配和 E0–E8 可配置；断点恢复不重复计分；导出 source data |
| R11 研究界面 | P2，R3/R9/R10 | 后续在 web_v2 增加版本、证据和成本查看 | 离线结果可导入；生产对话不默认触发演化；CLI 完成前不作为前置 |

R4–R7 是面向科研能力的功能，R0–R3/R8–R10 是使其可被可信比较的研究基础。优先完成两类功能共同形成的最小闭环，不能只做展示界面或只累积工具数量。

## 5 接口与数据合同

`RuntimeAdapter.run_task(task_spec, version, context, budget)` 返回 `RunResult`。底层实际调用沿用 KimiClient 的 `create_session`、`submit_prompt`、`stream_session_events` 和 `abort_prompt`。终止还须经资源管理器停止本次运行派生的训练进程；不以 HTTP abort 成功代替资源停止验证。

| 合同 | 至少包含 | 不应包含 |
| --- | --- | --- |
| RunContext | experiment/arm/lineage/version/run ID；workspace、skill、registry、scratch 根；预算句柄 | 全局可写根、其他组历史、认证材料 |
| TaskSpec | task/family/category；可见输入；产物合同；公开反馈规则；资源上限 | 隐藏标签、测试标准答案、可回写评分器路径 |
| RunResult | 外部汇总的状态、事件索引、产物哈希、工具错误、成本与 unknown 字段 | 将 agent 自报成功当独立评分 |
| CapabilityCard | 输入/输出规范、任务/单位、来源模型、数据指纹、适用范围、验证证据、失败边界 | 无依据的适用范围或伪造不确定性 |
| CandidateProposal | 父版本、修改对象、假设、预期改善任务、diff、开发验证、依赖与预算 | 测试任务专属答案、评价规则改动 |
| VersionManifest | P/S/W/R/U/M 哈希；原始基线；父版本；接受记录；依赖与工具清单 | 将路径存在等同于内容未变 |
| EvalResult | 评价器版本、有效性、主要指标、次要指标、失败类别、可公开反馈 | 在演化期间返回逐样本隐藏答案 |

证据 provenance 至少连接数据快照→训练配置→checkpoint→预测文件→开发验证→能力包→后续调用→最终产物。大文件保存哈希和对象位置，避免反复拷贝权重；不得将绝对机器路径当作跨环境可复现标识。

## 6 修改和接受的具体流程

1. 在干净会话和版本私有 MCP 中执行开发任务，采集原始轨迹、产物、成本和失败类型。
2. 改进器读取同一公开证据库，输出结构化假设和候选 patch。证据摘要须能追溯原始记录，不能只保留模型的解释。
3. 控制器检查路径范围、语法、依赖和内容；对任务 ID、标准答案、固定数据值等泄漏信号预检。可疑候选不得通过修改评分器获得收益。
4. 开发筛选通过后，在选择集对候选与 incumbent 配对确认。评价采用固定质量规则、噪声校准和资源限制。
5. 若候选包含 U 改动，额外做同父版本的 meta-probe。U 的验收依赖其生成后代的表现；一次执行侧提升不能替代。
6. 如果 E/U 互相依赖而只接受其中一部分，必须重新组合并验证实际待发布 manifest，不能直接拆分一份联合通过的修改。
7. 接受时原子写入完整能力包与指针；下一轮从新会话加载。拒绝时 incumbent 不变，但失败候选和花费保留。
8. 到达外部预算即停止。最终测试使用冻结版本，测试产物不回流到经验库或下一任务。

首版控制器按单一 incumbent 实现，但完整记录候选树。将多分支 archive 搜索作为后续独立扩展和消融，避免首版同时改变搜索空间、父选择和递归机制导致无法归因。本文借鉴 DGM 的谱系记录，不把首版称为完整 DGM 复现。

## 7 兼容策略与测试范围

Kimi 版本目前静态基线为 0.19.2；实际 provider/model snapshot 需 R0 运行核实。保留现有产品默认路径；研究入口必须显式提供 RunContext，不能通过全局环境变量临时切换生产服务状态。若采用进程级环境配置，应在独立子进程启动前注入并固定到进程结束。

`run_train_tool` 拟增加显式研究输出策略：产品调用沿用原有行为，研究调用写 staging 且不自动晋升为可复用能力。注册元数据区分 checkpoint 已生成、开发验证状态和是否可复用，不把已有模型默认标成通过新标准。历史模型可列出，但研究可复用目录只包含有对应证据的条目。

registry 的根目录、允许输入根、相对路径序列化、预测解析必须一起适配。索引写入采用原子替换并配合锁或单写者；只改 `registry_root()` 会留下读写不一致。skills 第一版采用进程与版本根隔离；后续同进程缓存必须带版本内容指纹，不能只清一次全局缓存。

有意义的回归测试范围：

- 两组同名 skill 和同名 model ID 返回不同且正确的版本；不能跨组读取 scratch 或模型索引。
- 候选发布中断、索引写入中断、重复事件与重复任务恢复后，版本和成本账本一致。
- 超时后本次 run 的训练进程及其子进程停止；其他用户或实验任务不受影响。
- 已知错误的样本 ID、残基编号、标签单位与训练测试重叠被检出；有效数据不误报。
- 错误 schema 或缺文件不能以正常退出码通过评价；旧 judge 的 invalid/error 不计成 A 赢。
- 用示例 U_A/U_B 验证装载机制：每轮记录不同哈希并改变下一次提案行为。这只证明接线，不能当作 RSI 效果。
- 同一冻结模型能够由 model_id 与版本 manifest 正确解析；旧产品路径仍能预测。

受影响的既有测试包括 `/home/tanyang/workspace/VenusFactory2/tests/agent/test_skills_loader.py`、`test_dual_mode_skills.py`、`tests/test_tool_context.py`、`tests/test_train_agent_sandbox.py`、`tests/test_sandbox.py` 和 `tests/test_tracing.py`。这些已有训练沙箱测试不自动覆盖 Kimi shell、远端 MCP 或 GPU 进程树。

## 8 里程碑与投入

以下是实现和整理的人日估计，不含 API/GPU 排队、原始数据恢复、专家任务审核和湿实验等待，也不是承诺的日历工期。本轮增加领域功能和强基线后，完整范围较原先 21–34 人日的初始估计扩大。

| 里程碑 | 预计投入 | 范围 | 验收物 |
| --- | --- | --- | --- |
| M0 基线和可用性审计 | 2–3 人日 | R0、旧 judge 问题复现、数据许可和来源清单 | baseline manifest、数据状态、费用计量可用性 |
| M1 可重复运行的实验底座 | 6–9 人日 | R1/R2/R3、R6 基本合同 | 6 个 smoke task；隔离、恢复、终止和错误产物测试 |
| M2 科研能力闭环 | 7–10 人日 | R4/R5/R6/R7 | 一条训练→验证→复用链和一条不适用→拒绝复用链；只要求接线正确 |
| M3 可递归改进和强基线 | 6–9 人日 | R8/R9、GEPA 适配、固定全历史提案器 | B3/B4/B5 可执行，U 真实加载且 meta-probe 独立 |
| M4 先导和正式协议冻结 | 4–6 人日 | R10、30 任务先导、成本与功效设计 | 继续/调整决定、固定任务和预算、无污染证明 |
| M5 正式实验分析与稿件 | 5–8 人日加计算 | E1–E8 的预定子集、source data、主文图件 | 可追溯比较与负结果；论文主张和证据对应 |

核心工程与先导 M0–M4 合计 25–37 人日；包含 M5 共 30–45 人日。可执行 U 插件、多分支 archive 和研究 UI 另列扩展，不悄悄计入此范围。生物实验工作量在确定数据与测量条件后单列。

最小可行版本由 M0/M1 和 M2 的一个轻量模型案例组成，先验证真实闭环与成本。它不满足 RSI 论文全部证据；M3 的递归机制和独立 H2 测试仍是必要条件。

## 9 首批实施顺序

第一次代码改动建议分为小范围、可单独审核的变更集，均在 hl_gpu 完成。

1. 基线导出和旧 judge 修复：枚举所有实际模型调用，新增显式 error/invalid 结果，不重写旧答案。
2. 研究 specs 与 runner：先用 mock Kimi 事件验证生命周期，再在 1 个轻量真实任务验证实际字段；缺失 usage 保留 unknown。
3. 独立资源根：串通 RunContext→MCP→skills→训练 staging→registry→预测解析，完成双版本同时运行测试。
4. 六个 smoke task 和外部评价：确认执行成功、科学正确和成本停止是不同检查。
5. 能力卡、适用性判断及模型复用：先用有来源的小数据和冻结特征模型跑通，暂不做大规模 GPU 搜索。
6. 版本控制器与固定提案器，再增加 U 版本化；不要将手工写好的 U_A/U_B 接线测试当自主改进结果。
7. GEPA 与增强固定改进器适配，随后运行 30 任务先导。

在 M4 之前不把正式测试分数用于调试。若数据或算力不足，应先减少同时推进的任务类型，保留 B3/B4 及同起点 H2 比较，不能通过删除强基线制造结论。

## 10 交付目录与完成定义

拟输出结构：

```text
/home/tanyang/workspace/VenusFactory2/
  src/agent/rsi/              运行适配和研究模块
  configs/rsi/                基线、各组配置、任务和预算规范
  scripts/rsi/                执行、分析、绘图和预算展开入口
  tests/rsi/                  隔离、事务、评价、预算与接线测试
  docs/rsi/                  实施计划和实验协议
  research_runs/rsi/          谱系、版本、公开开发轨迹和汇总结果
```

隐藏标签和最终评价器存储在 agent 无权读取的外部评价环境，不能因上面列出 `research_runs/` 就把答案放在同用户可读目录。凭据、私有输入和模型大文件不写进版本包或论文 source data。

工程完成定义：可从 manifest 重建运行配置；可追踪每个新能力的来源；候选拒绝/中断不会污染基线；新 U 的实际作用可验证；预算和评价独立生效。研究完成定义：完成预注册比较、失败与成本记录以及独立 H2 验证。前者不能代替后者。
