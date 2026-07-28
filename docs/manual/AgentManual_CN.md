# Agent：你的蛋白质工程 AI 助手

**Agent** 是 VenusFactory2 的对话入口。用中文（或英文）描述你想做什么，Agent 会自动规划多步骤工作流、调用合适的工具、与你确认中间结果，最后整合成一份完整报告。

---

## 1. 什么时候用 Agent

| 场景 | 为什么用 Agent |
| :--- | :--- |
| 需要多步分析，不想手动串工具 | Agent 自动规划流水线 |
| 需要围绕同一个蛋白做追问 | 整段对话保留上下文 |
| 想在执行前先看一眼计划 | 可在 Plan Editor 里重排、改写或删除步骤（Science Expert） |
| Expert 长任务、偶尔需要人工闸门 | 仍要在 Plan Editor 确认计划；步骤与小报告默认自动推进（只有选「确认计划」才会逐步暂停） |

如果只是想跑**单个**预测、不需要编排，用 **Quick Tools** 更快。
想完全掌控模型 + 数据集的选择，用 **Advanced Tools**。

---

## 2. 界面布局

聊天工作区分三栏：

| 栏 | 内容 |
| :--- | :--- |
| **左栏 — 会话列表** | 新建会话、历史会话、删除、复制 session ID。每个会话独立保留历史和上传的文件。 |
| **中栏 — 对话区** | 消息时间线、**Science Agent / Science Expert** 模式切换、模型选择器（**仅 Local Expert**；Online Expert 无选择器，使用平台固定后端模型）、文件上传 / Workspace 选择、Regenerate / Export / Stop / Send。Expert 在需要时内嵌澄清 / 计划 / 迭代等检查点卡片。 |
| **右栏 — 执行状态** | 实时状态、最近 12 次工具调用、对话日志尾部。长任务时用来观察进度。 |

**在线模式**下右上有一个配额胶囊提示今日剩余对话次数；用完后 Send / Regenerate / 文件上传都会被禁用。

---

## 3. 两种对话模式

在输入框上方用分段开关选择模式。选择会保存在浏览器；同一会话中途切换可能导致行为不一致，如有异常请新建会话。

### 3.1 Science Agent（kimi-code）

- **引擎：** 本地 **kimi-code** 守护进程（不是 LangGraph PI/CB/MLS/SC 图）。
- **行为：** 流式工具调用（VenusFactory MCP、允许的 shell）、可折叠 **Thinking** 块、时间线内工具执行卡片。
- **上下文（对齐 [Kimi Code sessions](https://www.kimi.com/code/docs/en/kimi-code-cli/guides/sessions.html)）：**
  - 占用 ≥ ~90% 时自动压缩；时间线展示压缩摘要。
  - 输入区控件：**压缩**（`/compact`）、**计划**（`/plan`）、**分叉**（`/fork`）、**清空上下文**（`/new`），以及上下文占用指示。
  - 停止时尽量 abort 正在运行的 kimi prompt。
- **模型：** 固定为 kimi-code；在 Settings / `.env` 配置 provider（见 WebUI wiki）。Online 模式可能走 `bwrap` 沙箱。
- **适用：** 希望单一 agent 流畅调工具、不走 Expert 规划流水线的开放式任务。

### 3.2 Science Expert（LangGraph）

- **引擎：** LangGraph **`graph`** 流水线，四角色 **PI**（规划 / 调研）→ **CB**（具体步骤）→ **MLS**（执行校验）→ **SC**（审查）。
- **先 PI：** 新请求一律先走 PI 分析 / 澄清（含显式「跳过 Research」），不再按关键字静默直跳 CB。
- **计划闸门：** 执行前仍要在 **Plan Editor** 审计划（改描述、排序、删步骤）。
- **默认更顺（少停顿）：**
  - **计划确认后：** 推荐 **确认并自动执行**（后端默认）；不在每步工具后暂停。只有点 **确认计划** 才会逐步检查点。
  - **小报告：** 若走文献调研，默认自动推进，不在每个小节都停；若仍出现检查点，可继续 / 重写 / 跳过。
  - **最终 SC 报告：** paper 级稿件（约 5000–8000 词），不是 800–1500 词短总结。
- **模型：** **Local** 可在模型选择器切换任意 graph 引擎 LLM（内置或自定义 OpenAI 兼容）；**Online** 无模型选择器，使用平台固定后端模型。

Expert 仍可能经历下列阶段；并非每次任务都会走全链路。

```
你 ─▶ 澄清 ─▶ 规划 ─▶ 执行 ─▶ 子报告 ─▶ 迭代决策 ─▶ 最终报告
```

#### 澄清（Clarification）

Agent 可能会先问几个简短问题——单选、多选或文本输入。选择最符合你意图的选项（或在"其他"里填自由文本）。提交后进入下一步。

> **小技巧：** 没有 `/skip-research`、`/loop` 等 slash 命令。请在澄清表单里选 **跳过 Research**，或直接发执行型请求（只要跑工具、无文献调研意图）——流水线可能自动跳过调研。

#### 规划（Plan）

Agent 给出一个有序步骤列表，每步包含工具名和简短任务描述。在 **Plan Editor** 里你可以：

- 修改任一步骤的描述
- 用 ↑ / ↓ 重排
- 删除某一步（至少保留 1 步）
- **确认并自动执行**（推荐）或 **确认计划**（逐步闸门）

#### 执行（Execution）

步骤按顺序执行。自动执行时右栏滚动工具日志，无步骤检查点。若未选自动执行，每一步可能出现 **继续 / 中止**。

#### 子报告（仅文献调研）

有调研小节时会生成短子报告；默认自动推进。若出现检查点，可选 **Continue Research**、**Comment & Rewrite** 或 **Skip to Report**。

#### 迭代决策（Iteration）

主流水线跑完后会出现一个最终选择：

- **Satisfied** — 直接出最终报告
- **Modify & Re-execute** — 回到规划阶段做修改后再跑
- **Continue Analysis** — 在已有结果之上继续追问

最终报告以结构化文档形式导出，**HTML** 和 **PDF** 都可以从 Export 按钮下载。

### 3.3 示例提示词

Agent 围绕"自然语言、目标驱动"设计。Planner 处理得好的几类典型模式：

| 工作流 | 示例提示词 | Planner 常用工具 |
| :--- | :--- | :--- |
| **功能 / 结构定位** | *"这个蛋白进入细胞核的可能性多大？"* — *"这个锌指蛋白的 DNA 结合位点在哪里？"* | Protein Function（Localization、Sorting Signal）；Functional Residue（Binding Site） |
| **理性设计与优化** | *"找出 5 个能同时提升活性和稳定性的突变。"* — *"这条序列的 Instability Index > 40，推荐 3 个稳定化突变。"* | Directed Evolution（Activity、Stability）；Physical & Chemical Properties |
| **序列与结构检索** | *"我的实验用到 UniProt ID P05798，把它的序列和结构拉下来。"* — *"用 InterPro 查这个序列的 domain 结构。"* | InterPro / UniProt 下载工具 |
| **集成分析** | *"分析这条序列的溶解性和热稳定性，基于结果给实验方案建议。"* | 多个预测工具串联，经 Analysis 模块整合 |
| **复杂任务拆解** | *"分析这条序列 (P60002.fasta) 的稳定性，再找出 5 个能提升稳定性的突变。"* | Planner 拆分为：Function Prediction (Stability) → Directed Evolution → Analysis |
| **条件逻辑** | *"如果 Instability Index 大于 40，推荐 5 个稳定化突变。"* | 先跑 Physical & Chemical Properties；只有阈值满足才触发 Mutation Prediction |
| **报告合成** | *"总结突变扫描的 top 3 结果并解释为什么预测它们稳定。"* | Analysis 模块把原始数据合成为自然语言 |

> **小技巧：** 目标越具体，Plan 越准。*"找突变"* 太泛；*"找 5 个能提升热稳定性又不损害活性的突变"* 才可执行。

---

## 4. 上传数据

输入框旁边的回形针 / Workspace 图标对应两种方式：

| 方式 | 适用场景 |
| :--- | :--- |
| **Upload（上传）** | 从本地传一个新文件（FASTA / PDB / CSV / TXT）。会根据扩展名自动归类。 |
| **Workspace** | 选用之前上传过、或工具产出的文件；可搜索。 |

Workspace **只在 Local 模式可用**。Online 模式下选择器被禁用。

---

## 5. 模型选择

**Science Expert（Local）：** 输入框上方切换驱动 LangGraph 的 LLM：

- **内置：** Gemini 2.5 Pro、GPT-4o、Claude 3.7、DeepSeek-R1
- **自定义 OpenAI 兼容：** 填入你自己的端点（显示名、模型名、API Key、Base URL）。保存在浏览器里，**仅 Local 模式可用**。

**Science Expert（Online）：** 无模型选择器——平台固定后端模型，客户端不可切换。

**Science Agent：** 仅使用 kimi-code；选择器显示固定的 Agent 标识，不切换 graph 模型。

中途换模型（Local Expert）会有提示——模型行为可能不一致。

---

## 6. 使用建议

- **说目标，不说按钮。** "为这个酶找 5 个稳定性提升的突变" 比 "跑突变工具然后排序" 更好。
- **复用上下文。** 在同一个会话里直接说"这个序列"，不用反复重传。
- **信任但复核。** Plan Editor 让你在烧 token / 算力之前修掉糟糕的计划。
- **重要的存下来。** Agent 不会在服务端长期保留会话——关闭浏览器前下载最终报告或把文件存到 Workspace。
- **看右栏。** 步骤看起来卡住时，实时工具日志通常能告诉你它是在下数据、调模型，还是真的挂了。

---

## 7. Local 模式 vs Online 模式

| 能力 | Local | Online |
| :--- | :---: | :---: |
| 内置模型 | ✓ | ✓ |
| 自定义 OpenAI 兼容模型 | ✓ | — |
| Workspace 上传 / 替换 / 删除 | ✓ | — （只读） |
| 每日对话配额 | 不限 | 受限（看配额胶囊） |

侧边栏始终显示当前运行模式的小徽章。
