export const chatMessages = {
  en: {
    docTitle: "Self-Evolving Agents — Science Chat — VenusFactory2",
    docDescription: "Talk to self-evolving AI agents for protein discovery and directed evolution. Science Agent and Science Expert plan, run, critique and iterate.",
    // Error hints (friendlyErrorHint)
    errHintQuota: "Please try again later.",
    errHintTimeout: "Request timed out. Please retry.",
    errHintNetwork: "Connection lost. Please reconnect and retry.",
    errHintAuth: "Authentication failed. Check your sign-in or API key.",
    errHintForbidden: "You do not have access to this action.",
    errHintRate: "Too many requests. Please try again shortly.",
    errHintServer: "Service unavailable. Please try again shortly.",
    errHintModel: "Model unavailable. Choose another model.",
    errHintSession: "Conversation not found. Start a new conversation.",
    errHintGeneric: "Could not complete this action. Please retry.",
    // ErrorAlert
    dismiss: "Dismiss",
    details: "Details",
    // Error fallbacks
    errCreateSession: "Failed to create session.",
    errDeleteSession: "Failed to delete session.",
    errOnlineLimit: (limit: number) => `Online mode limit reached: up to ${limit} chats per user per day.`,
    errNoUserMsg: "No previous user message in current session.",
    errStreamMsg: "Failed to stream message.",
    errRetryMsg: "Failed to retry message.",
    errStopRun: "Could not stop this task. Please retry.",
    errClarification: "Failed to submit clarification.",
    errAskUser: "Failed to submit answers to the agent.",
    errApproval: "Failed to submit approval decision.",
    errConfirmPlan: "Failed to confirm plan.",
    errIteration: "Failed to process iteration decision.",
    errStepDecision: "Failed to process step decision.",
    errSubReport: "Failed to process sub-report decision.",
    errCopySession: "Failed to copy session id.",
    errDownloadReport: "Failed to download report.",
    errSaveKey: "Failed to save API key.",
    errSwitchGateway: "Failed to switch gateway.",
    errExportSession: "Failed to export session bundle.",
    errRequiredCustomModel: "Display name, model name, API key and base URL are required.",
    errLabelConflictBuiltIn: "Display name conflicts with built-in model name. Please choose another name.",
    errLabelConflict: "Model name already exists. Please use another display name.",
    errEndpointConflict: "A model with the same model name and base URL already exists.",
    errRemoveCustomModel: "Failed to remove custom model.",
    // Notices
    noticeModelSwitched: "",
    noticeModeSwitched: "",
    noticeCustomRemoved: "Model removed.",
    noticeAgentDisabled: "Choose another executor.",
    noticeAskUserSubmitted: "Answers submitted — agent continuing…",
    noticeApprovalApproved: "Approved — agent continuing…",
    noticeApprovalRejected: "Rejected — agent continuing…",
    // Tooltips for model selector
    titleGatewayRequired: "Requires a configured gateway",
    titleMissingKey: (provider: string) => `Missing API key for ${provider}. Click to add.`,
    titleModelInfo: (label: string, provider: string) => `${label} (${provider})`,
    titleProviderWithKey: (provider: string) => `Provider: ${provider} (key configured)`,
    titleProviderNoKey: (provider: string) => `Provider: ${provider} (no API key)`,
    titleKeyConfigured: (provider: string) => `Manage ${provider} API key`,
    // Tooltips
    sendMessage: "Send message",
    regenerateLast: "Regenerate last message",
    quotaRemaining: (remaining: number, limit: number) => `Online mode quota: ${remaining}/${limit} chats remaining for this IP today.`,
    quotaRegenerate: (remaining: number, limit: number) => `Retries stay in the same conversation. New conversations remaining: ${remaining}/${limit}.`,
    // Header
    chat: "Chat",
    running: "Running",
    stopping: "Stopping",
    stopped: "Stopped",
    refresh: "Refresh",
    report: "Report",
    reportTooltip: "Download structured experiment report",
    // Sessions
    sessions: "Sessions",
    newSession: "+ New Session",
    noSessions: "No sessions yet",
    deleteSession: "Delete session",
    newChat: "New chat",
    idle: "idle",
    msgs: "msgs",
    copyFullId: "Copy full session id",
    copied: "✓ Copied",
    copySessionId: "Copy Session ID",
    // Roles
    rolePI: "Principal Investigator",
    roleCompBio: "Computational Biologist",
    roleMLSpec: "Machine Learning Specialist",
    // Composer
    composerWaiting: "Please finish the action card above…",
    composerWaitingClarification: "Please answer the clarification questions above…",
    composerWaitingAskUser: "Agent is waiting for your answer…",
    composerWaitingApproval: "Agent is waiting for your approval…",
    composerWaitingPlan: "Please confirm the execution plan above…",
    composerWaitingSubReport: "Please choose continue or revise above…",
    composerWaitingStep: "Please confirm the step result above…",
    composerWaitingIteration: "Please choose the next action above…",
    checkpointClarify: "Clarification",
    checkpointAskUser: "Agent question",
    checkpointApproval: "Approval required",
    checkpointPlan: "Execution plan",
    checkpointSubReport: "Sub-report checkpoint",
    checkpointStep: "Step checkpoint",
    checkpointIteration: "Next step",
    composerPlaceholder: "Ask anything about AI protein engineering...",
    quotaPillTitle: (remaining: number, limit: number) =>
      `${remaining} of ${limit} new conversations left today, shared by Agent, Scientist and Researcher. Follow-ups and retries do not use another conversation.`,
    quotaPillLabel: (remaining: number, limit: number) => `${remaining} left / ${limit}`,
    quotaPillExhausted: (used: number, limit: number) => `${used}/${limit} used`,
    chatModeAria: "Chat mode",
    modeScienceAgent: "Agent · Questions and tasks with tools as needed",
    modeScienceExpert: "Science · Expert team pursuing a persistent goal",
    modeAgentShort: "Agent",
    modeExpertShort: "Scientist",
    agentContextUsage: (pct: number, used: number, max: number) =>
      max > 0 ? `Context ${pct}% (${used}/${max} tok)` : `Context ${pct}%`,
    modelAria: "Model",
    modelLabel: "Model",
    modelViaKimi: "",
    gatewayRequiredSuffix: " (gateway required)",
    otherModel: "Other Model...",
    noKeySuffix: " (no key)",
    customSuffix: " (Custom)",
    keyOk: "Key ready",
    setKey: "Set key",
    gatewayAria: "Gateway",
    gatewayLabel: "Gateway",
    activeGateway: "Active gateway",
    noGateway: "No gateway",
    uploadFiles: "Attach files",
    fileSourceAria: "Choose file source",
    fileSourceLocal: "Local files",
    fileSourceLocalHint: "Upload files from this computer",
    fromWorkspace: "From Workspace",
    fromWorkspaceHint: "Pick files already in the local Workspace",
    fromWorkspaceOnlineHint: "Available only with local deployment. Online mode cannot browse Workspace.",
    export: "Export",
    exportTooltip: "Export session",
    stop: "Stop",
    stopTooltip: "Stop",
    send: "Send",
    sendTooltipShort: "Send",
    pipelineDismiss: "Dismiss",
    // Key panel
    keyPanelAria: (provider: string) => `Set API key for ${provider}`,
    keyPanelLabelPre: "API key for ",
    keyPanelLabelPost: ":",
    keyPanelHint: "",
    keyPanelSaving: "Saving...",
    save: "Save",
    clearKey: "Clear",
    cancel: "Cancel",
    // File preview
    workspaceSuffix: "(workspace)",
    // Execution status (right panel)
    // Custom model modal
    addCustomModel: "Add OpenAI-Style Model",
    displayName: "Display Name",
    displayNamePlaceholder: "My Model",
    modelName: "Model Name",
    modelNamePlaceholder: "gpt-4.1-mini",
    apiKey: "API Key",
    baseUrl: "Base URL",
    confirm: "Confirm",
    addedModels: "Added Models",
    noCustomModels: "No custom models yet.",
    deleteBtn: "Delete",
  },
  zh: {
    docTitle: "自进化智能体 — 科学对话 — VenusFactory2",
    docDescription: "与面向蛋白质发现与定向进化的自进化智能体对话。Science Agent 与 Science Expert 会规划、执行、批评并迭代。",
    // Error hints (friendlyErrorHint)
    errHintQuota: "请稍后重试。",
    errHintTimeout: "请求超时，请重试。",
    errHintNetwork: "连接中断，请检查网络后重试。",
    errHintAuth: "认证失败，请检查登录状态或 API Key。",
    errHintForbidden: "当前账号没有此操作权限。",
    errHintRate: "请求过于频繁，请稍后重试。",
    errHintServer: "服务暂不可用，请稍后重试。",
    errHintModel: "模型暂不可用，请选择其他模型。",
    errHintSession: "未找到会话，请新建会话。",
    errHintGeneric: "操作未完成，请重试。",
    // ErrorAlert
    dismiss: "关闭",
    details: "详情",
    // Error fallbacks
    errCreateSession: "创建会话失败。",
    errDeleteSession: "删除会话失败。",
    errOnlineLimit: (limit: number) => `已达到在线模式上限：每位用户每日最多 ${limit} 次对话。`,
    errNoUserMsg: "当前会话中没有以往的用户消息。",
    errStreamMsg: "消息流式传输失败。",
    errRetryMsg: "重试消息失败。",
    errStopRun: "暂时无法停止任务，请重试。",
    errClarification: "提交澄清答复失败。",
    errAskUser: "提交回答给 Agent 失败。",
    errApproval: "提交批准决定失败。",
    errConfirmPlan: "确认计划失败。",
    errIteration: "处理迭代决策失败。",
    errStepDecision: "处理步骤决策失败。",
    errSubReport: "处理子报告决策失败。",
    errCopySession: "复制会话 ID 失败。",
    errDownloadReport: "下载报告失败。",
    errSaveKey: "保存 API 密钥失败。",
    errSwitchGateway: "切换网关失败。",
    errExportSession: "导出会话包失败。",
    errRequiredCustomModel: "显示名称、模型名称、API 密钥和 Base URL 均为必填项。",
    errLabelConflictBuiltIn: "显示名称与内置模型名称冲突。请更换名称。",
    errLabelConflict: "模型名称已存在。请使用其他显示名称。",
    errEndpointConflict: "已存在相同模型名称和 Base URL 的模型。",
    errRemoveCustomModel: "删除自定义模型失败。",
    // Notices
    noticeModelSwitched: "",
    noticeModeSwitched: "",
    noticeCustomRemoved: "模型已删除。",
    noticeAgentDisabled: "请选择其他执行器。",
    noticeAskUserSubmitted: "已提交回答 — Agent 继续执行…",
    noticeApprovalApproved: "已批准 — Agent 继续执行…",
    noticeApprovalRejected: "已拒绝 — Agent 继续执行…",
    // Tooltips for model selector
    titleGatewayRequired: "需要先配置网关",
    titleMissingKey: (provider: string) => `缺少 ${provider} 的 API 密钥，点击添加。`,
    titleModelInfo: (label: string, provider: string) => `${label}（${provider}）`,
    titleProviderWithKey: (provider: string) => `服务商：${provider}（密钥已配置）`,
    titleProviderNoKey: (provider: string) => `服务商：${provider}（无 API 密钥）`,
    titleKeyConfigured: (provider: string) => `管理 ${provider} API Key`,
    // Tooltips
    sendMessage: "发送消息",
    regenerateLast: "重新生成上一条消息",
    quotaRemaining: (remaining: number, limit: number) => `在线模式配额：本 IP 今日剩余 ${remaining}/${limit} 次对话。`,
    quotaRegenerate: (remaining: number, limit: number) => `重新生成不占新对话额度。今日剩余：${remaining}/${limit}。`,
    // Header
    chat: "对话",
    running: "运行中",
    stopping: "停止中",
    stopped: "已停止",
    refresh: "刷新",
    report: "报告",
    reportTooltip: "下载结构化实验报告",
    // Sessions
    sessions: "会话",
    newSession: "+ 新建会话",
    noSessions: "暂无会话",
    deleteSession: "删除会话",
    newChat: "新对话",
    idle: "空闲",
    msgs: "条消息",
    copyFullId: "复制完整会话 ID",
    copied: "✓ 已复制",
    copySessionId: "复制会话 ID",
    // Roles
    rolePI: "首席研究员",
    roleCompBio: "计算生物学家",
    roleMLSpec: "机器学习专家",
    // Composer
    composerWaiting: "请先完成上方的操作卡片……",
    composerWaitingClarification: "请先回答上方的澄清问题……",
    composerWaitingAskUser: "Agent 正在等待你的回答…",
    composerWaitingApproval: "Agent 正在等待你的批准…",
    composerWaitingPlan: "请先确认上方的执行计划……",
    composerWaitingSubReport: "请先在上方选择继续或修改……",
    composerWaitingStep: "请先确认上方的步骤结果……",
    composerWaitingIteration: "请先选择下一步操作……",
    checkpointClarify: "需求澄清",
    checkpointAskUser: "Agent 提问",
    checkpointApproval: "需要批准",
    checkpointPlan: "执行计划",
    checkpointSubReport: "小报告检查点",
    checkpointStep: "步骤检查点",
    checkpointIteration: "下一步",
    composerPlaceholder: "随时提问 AI 蛋白质工程相关的任何问题……",
    quotaPillTitle: (remaining: number, limit: number) =>
      `今日还可开启 ${remaining}/${limit} 个新对话，三个模式共享；补充消息和重试不重复占用。`,
    quotaPillLabel: (remaining: number, limit: number) => `剩余 ${remaining}/${limit}`,
    quotaPillExhausted: (used: number, limit: number) => `已用 ${used}/${limit}`,
    chatModeAria: "对话模式",
    modeScienceAgent: "Agent · 自由问答与按需调用工具",
    modeScienceExpert: "Science · 专家团围绕目标持续执行",
    modeAgentShort: "智能体",
    modeExpertShort: "Scientist",
    agentContextUsage: (pct: number, used: number, max: number) =>
      max > 0 ? `上下文 ${pct}%（${used}/${max} tok）` : `上下文 ${pct}%`,
    modelAria: "模型",
    modelLabel: "模型",
    modelViaKimi: "",
    gatewayRequiredSuffix: "（需要网关）",
    otherModel: "其他模型……",
    noKeySuffix: "（无密钥）",
    customSuffix: "（自定义）",
    keyOk: "密钥就绪",
    setKey: "设置密钥",
    gatewayAria: "网关",
    gatewayLabel: "网关",
    activeGateway: "当前网关",
    noGateway: "无网关",
    uploadFiles: "添加文件",
    fileSourceAria: "选择文件来源",
    fileSourceLocal: "本地文件",
    fileSourceLocalHint: "从本机上传文件",
    fromWorkspace: "从工作区选择",
    fromWorkspaceHint: "选择本地 Workspace 中已有的文件",
    fromWorkspaceOnlineHint: "仅本地部署可用；在线模式无法浏览 Workspace。",
    export: "导出",
    exportTooltip: "导出会话",
    stop: "停止",
    stopTooltip: "停止",
    send: "发送",
    sendTooltipShort: "发送",
    pipelineDismiss: "关闭",
    // Key panel
    keyPanelAria: (provider: string) => `为 ${provider} 设置 API 密钥`,
    keyPanelLabelPre: "为 ",
    keyPanelLabelPost: " 设置 API 密钥：",
    keyPanelHint: "",
    keyPanelSaving: "保存中……",
    save: "保存",
    clearKey: "清空",
    cancel: "取消",
    // File preview
    workspaceSuffix: "（工作区）",
    // Execution status (right panel)
    // Custom model modal
    addCustomModel: "添加 OpenAI 风格模型",
    displayName: "显示名称",
    displayNamePlaceholder: "我的模型",
    modelName: "模型名称",
    modelNamePlaceholder: "gpt-4.1-mini",
    apiKey: "API 密钥",
    baseUrl: "Base URL",
    confirm: "确认",
    addedModels: "已添加的模型",
    noCustomModels: "暂无自定义模型。",
    deleteBtn: "删除",
  }
};

export type ChatMessages = typeof chatMessages.en;

type ErrorHintStrings = {
  errHintQuota: string;
  errHintTimeout: string;
  errHintNetwork: string;
  errHintAuth: string;
  errHintForbidden: string;
  errHintRate: string;
  errHintServer: string;
  errHintModel: string;
  errHintSession: string;
  errHintGeneric: string;
};

export function friendlyErrorHint(msg: string, t: ErrorHintStrings): string {
  const m = msg.toLowerCase();
  if (m.includes("quota") || m.includes("limit reached"))
    return t.errHintQuota;
  if (m.includes("timeout") || m.includes("timed out"))
    return t.errHintTimeout;
  if (m.includes("network") || m.includes("fetch") || /connection (lost|closed|reset|interrupted)/.test(m))
    return t.errHintNetwork;
  if (m.includes("401") || m.includes("unauthorized") || m.includes("auth"))
    return t.errHintAuth;
  if (m.includes("403") || m.includes("forbidden") || m.includes("access denied"))
    return t.errHintForbidden;
  if (m.includes("429") || m.includes("rate") || m.includes("too many"))
    return t.errHintRate;
  if (m.includes("500") || m.includes("internal server"))
    return t.errHintServer;
  if (m.includes("model") || m.includes("llm") || m.includes("api key"))
    return t.errHintModel;
  if (m.includes("session") || m.includes("not found"))
    return t.errHintSession;
  return t.errHintGeneric;
}
