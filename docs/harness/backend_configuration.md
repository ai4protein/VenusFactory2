# 后台模型、路由与 Scientist 配置

Web 的 Agent 和 Scientist 共用 `configs/harness/free_execution_catalog.json`。
也可以通过 `VENUS_EXECUTION_CATALOG` 指定服务端配置文件。修改后刷新页面的模型选项；已有任务保留创建时的模型、路由和 Harness，下一次请求仍检查最新的免费授权。

每个模型配置包含：

- `id`、`label`：页面选择项的稳定 ID 和名称。
- `binding`：`provider`、`model`、`base_url`、`protocol`、`credential_env`。凭据通过环境变量引用；不要把 API Key 写进此文件。
- `credential_toml`：可选的既有服务端凭据文件；其中端点必须与路由一致。
- `enabled`：是否开放该模型。
- `pricing`：已核实的输入和输出单价，字段为 `input_per_million` 和 `output_per_million`。仅两者均为数值 0 的路由能够调用；未知或付费模型保留灰色选项。
- `protocols`、`max_output_tokens`：允许的协议及输出上限。
- `inference_controls`：可选的 `temperature`、`thinking`、`reasoning_effort`。

`defaults` 分别设置 `science_agent` 和 `science_expert` 的默认模型与 Harness。`harnesses` 设置两种模式开放的执行器。Web 默认开放 Kimi Code、DeepSeek Harness、Codex、Claude Code；Scientist 使用对应的 `expert-` ID。LangGraph 保留历史实验和会话兼容，不出现在新会话选项中。

`configs/rsi/model_access.json` 的 schema 2 指向上述模型目录，实现页面可选项与真实请求的统一授权。OpenRouter 仍要求明确的 `:free` 模型及零费用响应记录，不进行付费回退。

## Progressive tool discovery

Local and online chat use the same native harness dispatcher and discovery protocol.
Deployment policy filters visibility and checks permission again before execution; it
does not select a different conversation implementation. Existing online quotas,
session isolation and resource limits still apply.

1. The initial prompt contains only enabled category names and brief summaries:
   biology databases, web/literature, computational tools, Factory-trained models,
   RSI methods, model development, session files/code, and manuals/skills.
2. `search_tools(category="compute", query="ProSST-2048")` returns a bounded page
   of capability names and short descriptions. Omit both category and query to
   inspect categories. Broad cross-category searches return category choices.
   Exact queries that resolve to one category remain supported for older clients.
   If only part of a long query matches, the best summary matches are labeled as
   partial. A topic with no direct match in guides or model registries returns
   their search entry points, without claiming any content was read.
3. `describe_tool(name)` loads the selected input contract and matching model
   selectors; `call_tool(name, arguments)` executes it through normal admission,
   quota and queue checks. A provider that emits a discovered MCP name directly
   is routed back through this broker; it still requires prior inspection and
   current permission. Other namespaces remain rejected.
4. For guidance, search the `guides` category, then find a relevant manual section
   or skill. Only `read_skill(skill_id)` loads that skill's contents.

Tool search indexes names and short summaries, not full schemas or skill bodies.
Plugins should provide a meaningful first sentence and may set
`metadata.discovery_category` to one of these IDs:
`databases`, `literature`, `compute`, `factory_models`, `rsi`, `training`,
`files`, `guides`. Otherwise the shared scheduler taxonomy and model provenance
determine the category. Platform model aliases come from the same enabled registry
used by Model Plaza; model variants are checked again before execution.

No tool or model becomes public merely by appearing in a skill or an RSI result.
Scientist artifacts continue to require the platform's publication review.

## Harness 与模型兼容性

安装了 Harness、配置了 Key，并不代表所有组合均已通过在线验证。选择接口的 `readiness` 标记的是静态准入；实际工具调用还取决于供应商当前端点、工具调用能力及限流状态。

| Harness | 请求协议 | 当前免费路由 |
| --- | --- | --- |
| Kimi Code | Chat Completions | DMX、OpenRouter |
| DeepSeek Harness | Chat Completions | DMX、OpenRouter |
| Claude Code | Messages 转 Chat Completions | DMX、OpenRouter |
| Codex | Responses；可显式转换为 Chat Completions | DMX、OpenRouter；取决于模型声明的协议 |

Codex 原生客户端仍使用 Responses。对于 Chat Completions 路由，平台网关转换文本、MCP 工具声明及调用历史；供应商、模型和免费授权保持原配置。只有完整且通过校验的工具调用才能执行，角色请求和原生客户端请求共用预算。网关记录供应商返回的 token 用量。

这个转换层支持文本与 MCP 工具结果，不接受图片、服务端历史引用、加密压缩项或结构化输出格式；这些能力需要对应协议的单独验证。原生 Responses 路由继续直接使用 Responses 网关。`protocols` 控制允许的协议，`max_output_tokens` 会约束实际请求输出上限。

工具搜索首先只返回简短名称与描述，读取接口后才可调用。普通 Web 工具允许返回 Markdown、纯文本、JSON 或多个文本块；科学实验的专用结果凭据仍保留严格校验。Claude 的 OpenRouter 适配只接受同一模型省略 `:free` 的返回名，费用和工具权限检查保持独立。

Web 遇到供应商 429 后终止本轮自动重试；内部网关向原生 SDK 返回不可重试错误，日志仍保留真实的 429。页面区分限流、模型不可用和超时，用户可以稍后重试。离线实验的既定重试策略不受此规则影响。

2026-11-03 检查中，OpenRouter 公共目录已不再包含 `poolside/laguna-s-2.1:free` 和 `poolside/laguna-xs-2.1:free`，默认配置将它们设为停用。免费供应商的可用性可能变化，维护时应核对官方模型目录和真实请求，不能只检查 Key 是否存在。

兼容性回归：

```bash
python -m pytest -q tests/harness/test_native_compatibility.py \
  tests/harness/test_web_live_text.py tests/harness/test_messages_bridge.py \
  tests/harness/test_responses_bridge.py \
  tests/harness/test_codex_adapter.py tests/harness/test_claude_events.py \
  tests/harness/test_claude_host.py tests/harness/test_responses_gateway.py \
  tests/harness/test_configured_free_routes.py tests/harness/test_user_interactions.py
```

`scripts/harness/check_compatibility.py` 使用与 Web 相同的隔离原生客户端和渐进式工具发现。默认只检查配置；添加 `--execute` 才会向明确选定的免费模型发送公开合成序列，每组合最多 6 次请求，顺序执行，无付费回退。报告分别记录静态准入和实际成功，不把服务商临时限流判定为永久不兼容。

```bash
python scripts/harness/check_compatibility.py --output research_runs/compatibility/preflight
python scripts/harness/check_compatibility.py --models dmx-glm4-free \
  --harnesses kimi-code deepseek-harness codex claude-code --execute \
  --output research_runs/compatibility/glm-live
```

## 自主研究

Scientist 的默认入口为 `autonomous_research.v1`。提交研究目标和运行预算即可，不需要选择任务模板或蛋白案例。已有模板目标可继续查看和恢复。

每轮按 PI 定义方向、CB 更新计划、MLS 使用原生 Harness 执行、SC 审查结果推进。计划和检查点持久化；暂停后可以恢复，缺失输入时可补充文本继续研究。只有计划中的工作有实际产物且通过审查和完整性检查，目标才完成。运行预算耗尽单独记录，不等同于完成。

生成的 Python 代码运行在 Bubblewrap 隔离环境中，只挂载目标工作目录和 Python 环境，不开放网络或宿主凭据。已有科学工具负责公开数据查询和下载。当前隔离 Python 步骤上限为 300 秒，长时间计算应交由已有训练/作业工具管理。

前端轮询显示专家报告、计划状态、当前工具、程序输出和中间文件。训练脚本可输出 JSON 行 `{"event":"progress","step":5,"total":20,"metrics":{"loss":0.3}}`；常见 `Epoch 5/20` 日志也会显示进度条。没有真实计数时只显示日志和耗时。

可复用程序提供 `run(inputs: dict)` 并通过隔离重放与独立审查。研究产物先保存在所属会话；公开发布还需平台审核和选择，并记录作者邮箱。进入平台注册表后，Agent 通过统一搜索、`describe_generated_capability` 和 `run_generated_capability` 调用。执行重放验证与科学预测准确率分别记录；准确率需要相应的数据划分和实验产物。

## Researcher 报告预算与恢复

Researcher 为八个专题分别检索和撰写，随后逐段核对原始来源，再修订和综合。审阅每批最多八段；超时或格式不完整的批次拆小重试，最多两轮，并继续处理其他章节。每段独立通过同一套证据校验后才缓存；缺失或不合格的段落继续审阅，不拆掉段内的错误论断来换取通过。未核对的内容保持未完成，作者修订后仍须重新审阅。

Administration 的 Environment 中可设置 VENUS_REPORT_MODEL_CALL_LIMIT：新报告默认最多 48 次模型请求，范围为 1–96，包含专题撰写、来源审阅、修订及综合。原有报告保持创建时的限额；降低后台设置也会限制恢复执行，调高设置不会给旧报告增加预算。供应商限流或不可用时停止当前执行，不自动改用付费模型。

Source-review policy v2 invalidates earlier review receipts while preserving them in checkpoint history. It additionally checks that explicitly measured values occur in the quoted evidence. These structural checks do not establish scientific entailment; a model review can still misinterpret a source.
