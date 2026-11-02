# 后台模型、路由与 Scientist 配置

Web 的 Agent 和 Scientist 共用 `configs/harness/free_execution_catalog.json`。
也可以通过 `VENUS_EXECUTION_CATALOG` 指定服务端配置文件。修改后刷新页面的连接状态；已有任务保留创建时的模型、路由和 Harness，下一次请求仍检查最新的免费授权。

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

## 自主研究

Scientist 的默认入口为 `autonomous_research.v1`。提交研究目标和运行预算即可，不需要选择任务模板或蛋白案例。已有模板目标可继续查看和恢复。

每轮按 PI 定义方向、CB 更新计划、MLS 使用原生 Harness 执行、SC 审查结果推进。计划和检查点持久化；暂停后可以恢复，缺失输入时可补充文本继续研究。只有计划中的工作有实际产物且通过审查和完整性检查，目标才完成。运行预算耗尽单独记录，不等同于完成。

生成的 Python 代码运行在 Bubblewrap 隔离环境中，只挂载目标工作目录和 Python 环境，不开放网络或宿主凭据。已有科学工具负责公开数据查询和下载。当前隔离 Python 步骤上限为 300 秒，长时间计算应交由已有训练/作业工具管理。

前端轮询显示专家报告、计划状态、当前工具、程序输出和中间文件。训练脚本可输出 JSON 行 `{"event":"progress","step":5,"total":20,"metrics":{"loss":0.3}}`；常见 `Epoch 5/20` 日志也会显示进度条。没有真实计数时只显示日志和耗时。

可复用程序提供 `run(inputs: dict)` 并通过隔离重放检查，审查通过后注册到 `research_runs/rsi/research_capabilities`。Agent 可通过统一搜索、`describe_generated_capability` 和 `run_generated_capability` 使用它们。执行重放验证与科学预测准确率分别记录；准确率需要相应的数据划分和实验产物。
