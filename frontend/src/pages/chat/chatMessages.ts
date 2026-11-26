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
    // Error fallbacks
    errCreateSession: "Failed to create session.",
    errDeleteSession: "Failed to delete session.",
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
    errExportSession: "Failed to export session bundle.",
    // Notices
    noticeModelSwitched: "",
    noticeAskUserSubmitted: "Answers submitted — agent continuing…",
    noticeApprovalApproved: "Approved — agent continuing…",
    noticeApprovalRejected: "Rejected — agent continuing…",
    // Tooltips for model selector
    titleGatewayRequired: "Requires a configured gateway",
    titleMissingKey: (provider: string) => `Missing API key for ${provider}. Click to add.`,
    titleModelInfo: (label: string, provider: string) => `${label} (${provider})`,
    titleProviderWithKey: (provider: string) => `Provider: ${provider} (key configured)`,
    titleProviderNoKey: (provider: string) => `Provider: ${provider} (no API key)`,
    // Tooltips
    sendMessage: "Send message",
    quotaRemaining: (remaining: number, limit: number) => `Online mode quota: ${remaining}/${limit} chats remaining for this IP today.`,
    // Sessions
    newChat: "New chat",
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
    checkpointPlan: "Execution plan",
    checkpointSubReport: "Sub-report checkpoint",
    checkpointStep: "Step checkpoint",
    checkpointIteration: "Next step",
    composerPlaceholder: "Ask anything about AI protein engineering...",
    quotaPillTitle: (remaining: number, limit: number) =>
      `${remaining} of ${limit} new conversations left today, shared by Agent and Researcher. Follow-ups and retries do not use another conversation.`,
    quotaPillLabel: (remaining: number, limit: number) => `${remaining} left / ${limit}`,
    quotaPillExhausted: (used: number, limit: number) => `${used}/${limit} used`,
    modelAria: "Model",
    modelLabel: "Model",
    modelViaKimi: "",
    gatewayRequiredSuffix: " (gateway required)",
    noKeySuffix: " (no key)",
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
    sendTooltipShort: "Send",
    // File preview
    workspaceSuffix: "(workspace)",
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
    // Error fallbacks
    errCreateSession: "创建会话失败。",
    errDeleteSession: "删除会话失败。",
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
    errExportSession: "导出会话包失败。",
    // Notices
    noticeModelSwitched: "",
    noticeAskUserSubmitted: "已提交回答 — Agent 继续执行…",
    noticeApprovalApproved: "已批准 — Agent 继续执行…",
    noticeApprovalRejected: "已拒绝 — Agent 继续执行…",
    // Tooltips for model selector
    titleGatewayRequired: "需要先配置网关",
    titleMissingKey: (provider: string) => `缺少 ${provider} 的 API 密钥，点击添加。`,
    titleModelInfo: (label: string, provider: string) => `${label}（${provider}）`,
    titleProviderWithKey: (provider: string) => `服务商：${provider}（密钥已配置）`,
    titleProviderNoKey: (provider: string) => `服务商：${provider}（无 API 密钥）`,
    // Tooltips
    sendMessage: "发送消息",
    quotaRemaining: (remaining: number, limit: number) => `在线模式配额：本 IP 今日剩余 ${remaining}/${limit} 次对话。`,
    // Sessions
    newChat: "新对话",
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
    checkpointPlan: "执行计划",
    checkpointSubReport: "小报告检查点",
    checkpointStep: "步骤检查点",
    checkpointIteration: "下一步",
    composerPlaceholder: "随时提问 AI 蛋白质工程相关的任何问题……",
    quotaPillTitle: (remaining: number, limit: number) =>
      `今日还可开启 ${remaining}/${limit} 个新对话，Agent 和 Researcher 共享；补充消息和重试不重复占用。`,
    quotaPillLabel: (remaining: number, limit: number) => `剩余 ${remaining}/${limit}`,
    quotaPillExhausted: (used: number, limit: number) => `已用 ${used}/${limit}`,
    modelAria: "模型",
    modelLabel: "模型",
    modelViaKimi: "",
    gatewayRequiredSuffix: "（需要网关）",
    noKeySuffix: "（无密钥）",
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
    sendTooltipShort: "发送",
    // File preview
    workspaceSuffix: "（工作区）",
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
