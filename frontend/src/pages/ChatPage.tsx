import { useEffect, useMemo, useRef, useState } from "react";
import { ChatTimeline } from "../components/ChatTimeline";
import { SessionFilesPanel } from "../components/SessionFilesPanel";
import { PipelineProgress } from "../components/PipelineProgress";
import { AskUserCard } from "../components/AskUserCard";
import { ApprovalCard, type ApprovalDecision } from "../components/ApprovalCard";
import { ClarificationForm } from "../components/ClarificationForm";
import { IterationDecision } from "../components/IterationDecision";
import { PlanEditor } from "../components/PlanEditor";
import { StepCheckpoint } from "../components/StepCheckpoint";
import { SubReportCheckpoint } from "../components/SubReportCheckpoint";
import { isResearchNoiseTool, type ToolExecution } from "../components/ToolExecutionCard";
import {
  cancelChatSession,
  createChatSession,
  exportChatSessionBundle,
  deleteCustomModelCache,
  deleteChatSession,
  downloadExperimentReport,
  getChatQuota,
  getChatSession,
  getChatSessionAuthHeaders,
  getAskUserRespondUrl,
  getApprovalDecideUrl,
  getClarificationRespondUrl,
  getPlanConfirmUrl,
  getStepDecideUrl,
  getSubReportDecideUrl,
  iterationDecide,
  listChatSessions,
  type ChatQuota,
  type ChatSnapshot,
  type ClarificationAnswer,
  type PlanStep,
  uploadFiles
} from "../lib/api";
import { streamSSEFromPost } from "../lib/sse";
import { PageFooter } from "../components/PageFooter";
import { WorkspaceFilePicker } from "../components/WorkspaceFilePicker";
import { type WorkspaceFile } from "../lib/workspaceApi";
import {
  useModelRegistry,
  setProviderKey,
  setActiveGateway,
  persistChatMode,
  pickExpertModelId,
  chatModeFromSnapshot,
  isKimiEngineModel,
  isGraphEngineModel,
  SCIENCE_AGENT_MODEL_ID,
  ONLINE_FIXED_EXPERT_MODEL_ID,
  type ChatMode,
  type ModelSpec,
} from "../lib/useModelRegistry";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const STRINGS = {
  en: {
    docTitle: "Science Chat — VenusFactory2",
    docDescription: "Science Agent and Science Expert chat for protein engineering predictions, training and analysis.",
    // Error hints (friendlyErrorHint)
    errHintQuota: "You've reached the daily usage limit for online mode. Try again tomorrow, or deploy locally for unlimited access.",
    errHintTimeout: "The request took too long. This can happen with complex tasks or heavy server load. Please try again.",
    errHintNetwork: "A network issue occurred. Please check your connection and try again.",
    errHintAuth: "Authentication failed. Your session may have expired — try refreshing the page.",
    errHintForbidden: "Access was denied. You may not have permission for this action.",
    errHintRate: "Too many requests in a short time. Please wait a moment and try again.",
    errHintServer: "The server encountered an internal error. This is usually temporary — please retry shortly.",
    errHintModel: "There was an issue with the AI model service. The model may be temporarily unavailable.",
    errHintSession: "The session could not be found. It may have expired — try creating a new session.",
    errHintGeneric: "Something went wrong. This is usually temporary — please try again or start a new session.",
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
    noticeModelSwitched: "Model/provider switched. Existing context may not be consistent across providers. Start a new session if results look off.",
    noticeModeSwitched: "Chat mode switched. Existing context may not be consistent across modes. Start a new session if results look off.",
    noticeCustomRemoved: "Custom model removed. Active session switched back to default model context.",
    noticeAgentDisabled: "Science Agent is currently unavailable. Check kimi-code setup, or switch to Science Expert.",
    noticeAskUserSubmitted: "Answers submitted — agent continuing…",
    noticeApprovalApproved: "Approved — agent continuing…",
    noticeApprovalRejected: "Rejected — agent continuing…",
    // Tooltips for model selector
    titleGatewayRequired: "Requires a configured gateway",
    titleMissingKey: (provider: string) => `Missing API key for ${provider}. Click to add.`,
    titleModelInfo: (label: string, provider: string) => `${label} (${provider})`,
    titleProviderWithKey: (provider: string) => `Provider: ${provider} (key configured)`,
    titleProviderNoKey: (provider: string) => `Provider: ${provider} (no API key)`,
    titleKeyConfigured: (provider: string) => `API key saved for ${provider} in ~/.venusfactory/keys.json. Click to update or clear.`,
    // Tooltips
    sendMessage: "Send message",
    regenerateLast: "Regenerate last message",
    quotaReached: (used: number, limit: number) => `Daily quota reached (${used}/${limit}).`,
    quotaRemaining: (remaining: number, limit: number) => `Online mode quota: ${remaining}/${limit} chats remaining for this IP today.`,
    quotaRegenerate: (remaining: number, limit: number) => `Regenerate also consumes quota. Remaining: ${remaining}/${limit}.`,
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
      `Online daily quota: ${remaining} sends left of ${limit} per IP. Counts each Send/Regenerate, not tool calls or streaming.`,
    quotaPillLabel: (remaining: number, limit: number) => `${remaining} left / ${limit}`,
    quotaPillExhausted: (used: number, limit: number) => `${used}/${limit} used`,
    chatModeAria: "Chat mode",
    modeScienceAgent: "Science Agent",
    modeScienceExpert: "Science Expert",
    modeAgentShort: "Agent",
    modeExpertShort: "Expert",
    modelAria: "Model",
    modelLabel: "Model",
    modelViaKimi: "via kimi",
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
    keyPanelHint: "Paste a new key to update, or click Clear to remove the saved key.",
    keyPanelSaving: "Saving...",
    save: "Save",
    clearKey: "Clear",
    cancel: "Cancel",
    // File preview
    workspaceSuffix: "(workspace)",
    // Execution status (right panel)
    executionStatus: "Execution Status",
    termWaiting: "$ waiting for session...",
    termMessagesTools: (msgs: number, tools: number) => `${msgs} messages, ${tools} tool runs`,
    termEmpty: "(empty)",
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
    docTitle: "科学对话 — VenusFactory2",
    docDescription: "通过 Science Agent / Science Expert 双模式对话，运行蛋白质工程预测、训练与分析任务。",
    // Error hints (friendlyErrorHint)
    errHintQuota: "您已达到在线模式的每日使用上限。请明天再试，或本地部署以获得无限制访问。",
    errHintTimeout: "请求耗时过长。复杂任务或服务器负载较高时可能出现此问题，请重试。",
    errHintNetwork: "出现网络问题。请检查您的网络连接后重试。",
    errHintAuth: "身份验证失败。您的会话可能已过期 — 请尝试刷新页面。",
    errHintForbidden: "访问被拒绝。您可能没有执行此操作的权限。",
    errHintRate: "短时间内请求过多。请稍候再试。",
    errHintServer: "服务器遇到内部错误。通常为临时问题 — 请稍后重试。",
    errHintModel: "AI 模型服务出现问题。该模型可能暂时不可用。",
    errHintSession: "未找到该会话。可能已过期 — 请尝试创建新会话。",
    errHintGeneric: "出现错误。通常为临时问题 — 请重试或开启新会话。",
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
    noticeModelSwitched: "已切换模型 / 服务商。已有上下文在不同服务商之间可能不一致，如结果异常请新建会话。",
    noticeModeSwitched: "已切换对话模式。已有上下文在不同模式之间可能不一致，如结果异常请新建会话。",
    noticeCustomRemoved: "自定义模型已删除。当前会话已切换回默认模型上下文。",
    noticeAgentDisabled: "Science Agent 当前不可用。请检查 kimi-code 配置，或切换到 Science Expert。",
    noticeAskUserSubmitted: "已提交回答 — Agent 继续执行…",
    noticeApprovalApproved: "已批准 — Agent 继续执行…",
    noticeApprovalRejected: "已拒绝 — Agent 继续执行…",
    // Tooltips for model selector
    titleGatewayRequired: "需要先配置网关",
    titleMissingKey: (provider: string) => `缺少 ${provider} 的 API 密钥，点击添加。`,
    titleModelInfo: (label: string, provider: string) => `${label}（${provider}）`,
    titleProviderWithKey: (provider: string) => `服务商：${provider}（密钥已配置）`,
    titleProviderNoKey: (provider: string) => `服务商：${provider}（无 API 密钥）`,
    titleKeyConfigured: (provider: string) => `已在 ~/.venusfactory/keys.json 保存 ${provider} 的密钥。点击可更新或清空。`,
    // Tooltips
    sendMessage: "发送消息",
    regenerateLast: "重新生成上一条消息",
    quotaReached: (used: number, limit: number) => `每日配额已用尽（${used}/${limit}）。`,
    quotaRemaining: (remaining: number, limit: number) => `在线模式配额：本 IP 今日剩余 ${remaining}/${limit} 次对话。`,
    quotaRegenerate: (remaining: number, limit: number) => `重新生成同样会消耗配额。剩余：${remaining}/${limit}。`,
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
      `在线日配额：本 IP 今日还剩 ${remaining}/${limit} 次发送。按每次点「发送」计数，工具调用/流式输出不另计。`,
    quotaPillLabel: (remaining: number, limit: number) => `剩余 ${remaining}/${limit}`,
    quotaPillExhausted: (used: number, limit: number) => `已用 ${used}/${limit}`,
    chatModeAria: "对话模式",
    modeScienceAgent: "科学智能体",
    modeScienceExpert: "科学专家",
    modeAgentShort: "智能体",
    modeExpertShort: "专家",
    modelAria: "模型",
    modelLabel: "模型",
    modelViaKimi: "经 kimi",
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
    keyPanelHint: "留空后点「清空」可删除已保存的密钥。",
    keyPanelSaving: "保存中……",
    save: "保存",
    clearKey: "清空",
    cancel: "取消",
    // File preview
    workspaceSuffix: "（工作区）",
    // Execution status (right panel)
    executionStatus: "执行状态",
    termWaiting: "$ 等待会话……",
    termMessagesTools: (msgs: number, tools: number) => `${msgs} 条消息，${tools} 次工具调用`,
    termEmpty: "（空）",
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

type SessionMeta = {
  session_id: string;
  created_at: string;
  model_name: string;
  history_size: number;
  status: string;
  title?: string;
};

const TITLE_SKIP_PREFIXES = [
  "📝",
  "clarification details",
  "**clarification",
  "澄清详情",
  "澄清细节",
];

/** Compress a user message into a short sidebar label (not a full copy). */
function abbreviateSessionTitle(raw: string, maxLen = 22): string {
  const lines = raw
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith(">"));
  let text = (lines.length ? lines.join(" ") : raw).replace(/\s+/g, " ").trim();
  if (!text) return "";
  const lower = text.toLowerCase();
  if (TITLE_SKIP_PREFIXES.some((p) => lower.startsWith(p) || text.startsWith(p))) {
    return "";
  }

  // Prefer the first clause / sentence.
  const hardSeps = ["。", "！", "？", "；", ". ", "! ", "? ", "; "];
  for (const sep of hardSeps) {
    if (text.includes(sep)) {
      const head = text.split(sep)[0]?.trim() || "";
      if (head.length >= 4) {
        text = head;
        break;
      }
    }
  }
  if (text.length > maxLen) {
    for (const sep of ["，", ", "]) {
      if (text.includes(sep)) {
        const head = text.split(sep)[0]?.trim() || "";
        if (head.length >= 4) {
          text = head;
          break;
        }
      }
    }
  }

  // English-heavy prompts → keep a few leading words.
  const asciiRatio = [...text].filter((ch) => ch.charCodeAt(0) < 128).length / Math.max(text.length, 1);
  if (asciiRatio > 0.7) {
    const words = text.split(/\s+/);
    if (words.length > 3) text = words.slice(0, 3).join(" ");
  }

  text = text.replace(/^[\s·\-–—:：,，;；]+|[\s·\-–—:：,，;；]+$/g, "");
  if (!text) return "";
  if (text.length <= maxLen) return text;
  let cut = text.slice(0, maxLen - 1);
  if (asciiRatio > 0.7) {
    const sp = cut.lastIndexOf(" ");
    if (sp >= 6) cut = cut.slice(0, sp);
  }
  return `${cut.trimEnd()}…`;
}

function titleFromHistory(history: Array<{ role?: string; content?: unknown }> | undefined): string {
  if (!history?.length) return "";
  for (const item of history) {
    if (item?.role !== "user") continue;
    const content = item.content;
    let raw = "";
    if (typeof content === "string") raw = content;
    else if (Array.isArray(content)) {
      raw = content
        .map((part) => {
          if (typeof part === "string") return part;
          if (part && typeof part === "object") {
            const obj = part as { text?: string; content?: string };
            return obj.text || obj.content || "";
          }
          return "";
        })
        .filter(Boolean)
        .join(" ");
    }
    const title = abbreviateSessionTitle(raw);
    if (title) return title;
  }
  return "";
}

// Last-resort fallback identifier used only when the model registry has not
// loaded yet AND we cannot read any model id from the active session. The real
// model list comes from GET /api/models via useModelRegistry().
const FALLBACK_MODEL_ID = SCIENCE_AGENT_MODEL_ID;
const OTHER_MODEL_OPTION = "__other_model__";
type RunStatus = "running" | "stopping" | "stopped";

type OpenAIStyleModel = {
  id: string;
  label: string;
  modelName: string;
  apiKey: string;
  baseUrl: string;
};

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

function friendlyErrorHint(msg: string, t: ErrorHintStrings): string {
  const m = msg.toLowerCase();
  if (m.includes("quota") || m.includes("limit reached"))
    return t.errHintQuota;
  if (m.includes("timeout") || m.includes("timed out"))
    return t.errHintTimeout;
  if (m.includes("network") || m.includes("fetch") || m.includes("failed to fetch"))
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

type ErrorAlertStrings = ErrorHintStrings & { dismiss: string; details: string };

function ErrorAlert({ message, onDismiss, t }: { message: string; onDismiss: () => void; t: ErrorAlertStrings }) {
  return (
    <div className="error-box">
      <div className="error-box-header">
        <span className="error-box-icon">!</span>
        <span className="error-box-hint">{friendlyErrorHint(message, t)}</span>
        <button className="error-box-dismiss" onClick={onDismiss} title={t.dismiss}>&times;</button>
      </div>
      <details className="error-box-details">
        <summary>{t.details}</summary>
        <pre className="error-box-raw">{message}</pre>
      </details>
    </div>
  );
}

type ChatPageProps = {
  workspaceEnabled?: boolean;
};

export function ChatPage({ workspaceEnabled = false }: ChatPageProps) {
  const { lang, t: translate } = useLang();
  const t = translate(STRINGS);
  useDocumentMeta({ title: t.docTitle, description: t.docDescription });
  const [sessionId, setSessionId] = useState<string>("");
  const [snapshot, setSnapshot] = useState<ChatSnapshot | null>(null);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [message, setMessage] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [workspaceFiles, setWorkspaceFiles] = useState<WorkspaceFile[]>([]);
  const [running, setRunning] = useState(false);
  const [runStatus, setRunStatus] = useState<RunStatus>("stopped");
  const [streamingIdx, setStreamingIdx] = useState(-1);
  const [error, setError] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>(FALLBACK_MODEL_ID);
  const [chatMode, setChatMode] = useState<ChatMode>("science_agent");
  const registry = useModelRegistry();
  const [keyPanelProvider, setKeyPanelProvider] = useState<string>("");
  const [keyPanelValue, setKeyPanelValue] = useState<string>("");
  const [keyPanelSaving, setKeyPanelSaving] = useState(false);
  const [customModels, setCustomModels] = useState<OpenAIStyleModel[]>([]);
  const [showCustomModelModal, setShowCustomModelModal] = useState(false);
  const [customModelLabel, setCustomModelLabel] = useState("");
  const [customModelName, setCustomModelName] = useState("");
  const [customModelApiKey, setCustomModelApiKey] = useState("");
  const [customModelBaseUrl, setCustomModelBaseUrl] = useState("https://api.openai.com/v1");
  const [modelSwitchNotice, setModelSwitchNotice] = useState("");
  const [chatQuota, setChatQuota] = useState<ChatQuota | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  /** Expert token pump: apply SSE tokens across animation frames so a
   * buffered burst still paints as progressive streaming. */
  const tokenQueueRef = useRef<Array<{ content: string; role_id?: string }>>([]);
  const tokenPumpScheduledRef = useRef(false);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const fileSourceMenuRef = useRef<HTMLDivElement | null>(null);
  const [fileSourceMenuOpen, setFileSourceMenuOpen] = useState(false);
  const [workspacePickerOpen, setWorkspacePickerOpen] = useState(false);
  const SESSION_STORAGE_KEY = "vf2_active_session_id";
  const SESSION_CACHE_KEY = "vf2_session_list_cache";
  const SESSION_OWNED_KEY = "vf2_owned_session_ids";
  const COPY_HINT_MS = 1200;
  const [copiedSessionId, setCopiedSessionId] = useState("");
  const [sessionsCollapsed, setSessionsCollapsed] = useState(false);
  const [logsCollapsed, setLogsCollapsed] = useState(false);
  const [pipelineDismissed, setPipelineDismissed] = useState(false);
  const CUSTOM_MODELS_STORAGE_KEY = "vf2_custom_openai_style_models";

  useEffect(() => {
    try {
      const raw = localStorage.getItem(CUSTOM_MODELS_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as OpenAIStyleModel[];
      if (Array.isArray(parsed)) {
        setCustomModels(parsed.filter((item) => item && item.id && item.modelName && item.baseUrl));
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(CUSTOM_MODELS_STORAGE_KEY, JSON.stringify(customModels));
  }, [customModels]);

  // Cleanup pass: if any customModel id collides with a built-in registry id,
  // drop it — those are stale entries from the older race-condition bug where
  // rememberModelFromSession ran before the registry loaded.
  useEffect(() => {
    if (!registry.data || !Array.isArray(registry.data.models)) return;
    const builtinIds = new Set(registry.data.models.map((m) => m.id));
    setCustomModels((prev) => {
      const cleaned = prev.filter((m) => !builtinIds.has(m.id));
      return cleaned.length === prev.length ? prev : cleaned;
    });
  }, [registry.data]);

  useEffect(() => {
    void bootstrapSession();
  }, []);

  useEffect(() => {
    if (!fileSourceMenuOpen) return;
    function onDocPointerDown(ev: MouseEvent) {
      const root = fileSourceMenuRef.current;
      if (!root) return;
      if (ev.target instanceof Node && !root.contains(ev.target)) {
        setFileSourceMenuOpen(false);
      }
    }
    function onKeyDown(ev: KeyboardEvent) {
      if (ev.key === "Escape") setFileSourceMenuOpen(false);
    }
    document.addEventListener("mousedown", onDocPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onDocPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [fileSourceMenuOpen]);

  const lastContentLen = snapshot?.history?.[snapshot.history.length - 1]?.content?.length ?? 0;
  useEffect(() => {
    const el = timelineRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 150) {
      el.scrollTop = el.scrollHeight;
    }
  }, [snapshot?.history.length, lastContentLen]);

  useEffect(() => {
    const s = (snapshot?.status || "").toLowerCase();
    if (!s) return;
    if (
      s === "stopped" ||
      s === "completed" ||
      s === "error" ||
      s === "planning_failed" ||
      s === "execution_failed" ||
      s === "waiting_for_clarification" ||
      s === "waiting_for_kimi_question" ||
      s === "waiting_for_kimi_approval" ||
      s === "waiting_for_plan_confirmation" ||
      s === "waiting_for_iteration" ||
      s === "waiting_for_step_review" ||
      s === "waiting_for_sub_report_review"
    ) {
      setRunStatus("stopped");
      return;
    }
    setRunStatus((prev) => {
      if (prev !== "running" && prev !== "stopping") setPipelineDismissed(false);
      return prev === "stopping" ? prev : "running";
    });
  }, [snapshot?.status]);

  // Keep sidebar label in sync with the first user message of the active chat.
  const activeSessionTitle = titleFromHistory(snapshot?.history);
  useEffect(() => {
    if (!sessionId) return;
    const historySize = snapshot?.history?.length ?? 0;
    // Never blank out an existing title with "" while history is still loading.
    if (!activeSessionTitle && historySize === 0) return;
    setSessions((prev) => {
      let changed = false;
      const next = prev.map((item) => {
        if (item.session_id !== sessionId) return item;
        const nextTitle = activeSessionTitle || item.title || "";
        if ((item.title || "") === nextTitle && item.history_size === historySize) return item;
        changed = true;
        return { ...item, title: nextTitle, history_size: historySize };
      });
      if (!changed) return prev;
      try {
        localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  }, [sessionId, activeSessionTitle, snapshot?.history?.length]);

  const terminalData = useMemo(() => {
    if (!snapshot) return null;
    return {
      status: snapshot.status || "idle",
      messages: snapshot.history.length,
      toolRuns: snapshot.tool_executions.length,
      conv: snapshot.conversation_log.slice(-6).map((e) => ({
        role: (e.role as string) || "unknown",
        content: ((e.content as string) || "").slice(0, 200),
      })),
      tools: snapshot.tool_executions.slice(-12).map((e) => ({
        name: String(e.tool_name || "tool"),
        ts: String(e.timestamp || ""),
        status: String(e.status || ""),
      })),
    };
  }, [snapshot]);

  async function fetchSessions() {
    const data = await listChatSessions();
    const allServer = data.sessions || [];
    const list = filterOwnedSessions(allServer);
    // Preserve locally derived titles when the API omits them (older server
    // process, or race before the first message is persisted).
    let merged: SessionMeta[] = [];
    setSessions((prev) => {
      const prevById = new Map(prev.map((item) => [item.session_id, item]));
      merged = list.map((item) => {
        const prior = prevById.get(item.session_id);
        const serverTitle = (item.title || "").trim();
        const priorTitle = (prior?.title || "").trim();
        const activeTitle =
          item.session_id === sessionId ? titleFromHistory(snapshot?.history) : "";
        const raw = serverTitle || activeTitle || priorTitle || "";
        return {
          ...item,
          // Always compress — older clients may have stored near-full questions.
          title: abbreviateSessionTitle(raw) || raw,
        };
      });
      try {
        localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(merged));
      } catch {
        /* ignore */
      }
      return merged;
    });
    return merged;
  }

  function readOwnedSessionIds(): string[] {
    try {
      const parsed = JSON.parse(localStorage.getItem(SESSION_OWNED_KEY) || "[]") as unknown;
      return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string" && Boolean(item)) : [];
    } catch {
      return [];
    }
  }

  function writeOwnedSessionIds(ids: string[]) {
    localStorage.setItem(SESSION_OWNED_KEY, JSON.stringify(Array.from(new Set(ids.filter(Boolean)))));
  }

  function rememberOwnedSession(nextSessionId: string) {
    if (!nextSessionId) return;
    writeOwnedSessionIds([...readOwnedSessionIds(), nextSessionId]);
  }

  function forgetOwnedSession(targetSessionId: string) {
    writeOwnedSessionIds(readOwnedSessionIds().filter((item) => item !== targetSessionId));
  }

  function filterOwnedSessions(list: SessionMeta[]) {
    const owned = new Set(readOwnedSessionIds());
    if (sessionId) {
      owned.add(sessionId);
    }
    return list.filter((item) => owned.has(item.session_id));
  }

  async function refreshChatQuota(): Promise<ChatQuota | null> {
    try {
      const quota = await getChatQuota();
      setChatQuota(quota);
      return quota;
    } catch {
      setChatQuota(null);
      return null;
    }
  }

  /** Optimistic local decrement so the pill moves as soon as Send starts. */
  function noteQuotaSendStarted() {
    setChatQuota((prev) => {
      if (!prev?.enforced) return prev;
      const used = (prev.used ?? 0) + 1;
      const limit = prev.limit ?? 0;
      return {
        ...prev,
        used,
        remaining: Math.max(0, limit - used),
      };
    });
  }

  async function createAndActivateSession() {
    if (running) return;
    setError("");
    try {
      const created = await createChatSession();
      rememberOwnedSession(created.session_id);
      setSessionId(created.session_id);
      setModelSwitchNotice("");
      sessionStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
      let runtimeMode = chatQuota?.mode;
      if (!runtimeMode) {
        try {
          runtimeMode = (await getChatQuota()).mode;
        } catch {
          runtimeMode = undefined;
        }
      }
      if (runtimeMode !== "local") {
        // Online: Agent → kimi sentinel; Expert → fixed DeepSeek.
        setSelectedModel(
          chatMode === "science_agent" ? SCIENCE_AGENT_MODEL_ID : ONLINE_FIXED_EXPERT_MODEL_ID
        );
        if (chatMode !== "science_agent") {
          rememberModelFromSession(ONLINE_FIXED_EXPERT_MODEL_ID);
        }
      } else {
        // Local: both modes pick a concrete registry/custom model.
        const createdId = modelLabelFromInternal(created.model_name);
        const nextId =
          isKimiEngineModel({ id: createdId, engine: undefined }) || !createdId
            ? pickExpertModelId(registry.data?.models || [], selectedModel)
            : createdId;
        setSelectedModel(nextId);
        rememberModelFromSession(nextId);
      }
      const newMeta: SessionMeta = {
        session_id: created.session_id,
        created_at: created.created_at,
        model_name: created.model_name,
        history_size: 0,
        status: "",
        title: "",
      };
      setSessions((prev) => {
        const exists = prev.some((s) => s.session_id === created.session_id);
        return exists ? prev : [newMeta, ...prev];
      });
      setSnapshot({
        session_id: created.session_id,
        model_name: created.model_name,
        created_at: created.created_at,
        history: [],
        conversation_log: [],
        tool_executions: [],
        status: "",
        clarification_questions: [],
        plan: [],
        waiting_for: "",
      });
      try { await fetchSessions(); } catch { /* optimistic update above is sufficient */ }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errCreateSession);
    }
  }

  async function deleteAndSelectNextSession(targetId: string) {
    if (running && targetId === sessionId) return;
    setError("");
    try {
      await deleteChatSession(targetId);
      forgetOwnedSession(targetId);
      if (targetId === sessionId) {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
        setSessionId("");
        setSnapshot(null);
      }
      const list = await fetchSessions();
      if (targetId !== sessionId) return;

      const next = list.find((item) => item.session_id !== targetId);
      if (next) {
        await refreshCurrentSession(next.session_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errDeleteSession);
    }
  }

  async function bootstrapSession() {
    setError("");
    const quota = await refreshChatQuota();
    const localMode = quota?.mode === "local";
    let list: SessionMeta[] = [];
    try {
      const raw = localStorage.getItem(SESSION_CACHE_KEY);
      if (raw) {
        const cached = JSON.parse(raw) as SessionMeta[];
        if (Array.isArray(cached) && cached.length > 0) {
          setSessions(cached);
          list = cached;
        }
      }
    } catch {
      // best effort cache read
    }
    try {
      list = await fetchSessions();
    } catch {
      // keep cached list if server refresh fails
    }
    const remembered = sessionStorage.getItem(SESSION_STORAGE_KEY);
    const candidates = remembered
      ? [remembered, ...list.filter((s) => s.session_id !== remembered).map((s) => s.session_id)]
      : list.map((s) => s.session_id);

    for (const sid of candidates) {
      if (!list.find((s) => s.session_id === sid)) continue;
      try {
        await refreshCurrentSession(sid, { localMode });
        return;
      } catch {
        // inaccessible session — already cleaned up by refreshCurrentSession, try next
      }
    }

    await createAndActivateSession();
  }

  // Translate a backend model identifier to the value used in the selector.
  // With the registry-driven UI the selector value IS the backend id, so this
  // just normalizes empties / falls back to the registry default.
  function modelLabelFromInternal(modelName: string) {
    const trimmed = (modelName || "").trim();
    if (trimmed) return trimmed;
    return registry.data?.default_model || FALLBACK_MODEL_ID;
  }

  function rememberModelFromSession(modelName: string) {
    const normalized = (modelName || "").trim();
    if (!normalized) return;
    // Skip if the registry hasn't loaded yet: otherwise every built-in model id
    // would be misclassified as "custom" (race with bootstrapSession), which
    // makes the next send payload carry custom_model_id and get 403'd in online mode.
    if (!registry.data || !Array.isArray(registry.data.models)) return;
    const registryIds = new Set(registry.data.models.map((m) => m.id));
    if (registryIds.has(normalized)) return;
    setCustomModels((prev) => {
      if (prev.some((item) => item.modelName === normalized || item.label === normalized)) return prev;
      return [
        ...prev,
        {
          id: normalized,
          label: normalized,
          modelName: normalized,
          apiKey: "",
          baseUrl: "https://api.openai.com/v1",
        },
      ];
    });
  }

  async function refreshCurrentSession(targetId?: string, opts?: { localMode?: boolean }) {
    const sid = targetId || sessionId;
    if (!sid) return;
    try {
      const s = await getChatSession(sid);
      setSnapshot(s);
      setSessionId(sid);
      setModelSwitchNotice("");
      sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
      // Empty sessions (no turns yet) always open as Science Agent on refresh.
      // Sessions with history keep their persisted/inferred mode.
      const hasHistory = (s.history?.length ?? 0) > 0;
      const nextMode: ChatMode = hasHistory
        ? chatModeFromSnapshot({
            chat_mode: s.chat_mode,
            engine: s.engine,
            model_name: s.model_name,
            models: registry.data?.models,
          })
        : "science_agent";
      setChatMode(nextMode);
      persistChatMode(nextMode);
      const modelId = modelLabelFromInternal(s.model_name);
      const isLocal = opts?.localMode ?? chatQuota?.mode === "local";
      if (nextMode === "science_agent" && !isLocal) {
        setSelectedModel(SCIENCE_AGENT_MODEL_ID);
      } else {
        const graphId = isKimiEngineModel({ id: modelId, engine: s.engine })
          ? pickExpertModelId(registry.data?.models || [], null)
          : modelId;
        setSelectedModel(graphId);
        rememberModelFromSession(s.model_name);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "";
      if (msg.includes("404") || msg.includes("not found") || msg.includes("Not Found")) {
        forgetOwnedSession(sid);
        if (sid === sessionId) {
          setSessionId("");
          setSnapshot(null);
          sessionStorage.removeItem(SESSION_STORAGE_KEY);
        }
        setSessions((prev) => prev.filter((s) => s.session_id !== sid));
      }
      throw err;
    }
  }

  function appendTokenToSnapshot(content: string, roleId?: string) {
    if (!content) return;
    setSnapshot((prev) => {
      if (!prev) return prev;
      const history = [...prev.history];
      const last = history[history.length - 1];
      if (last && last.role === "assistant" && last.kind !== "thinking") {
        history[history.length - 1] = { ...last, content: (last.content || "") + content };
        setStreamingIdx(history.length - 1);
      } else {
        history.push({
          role: "assistant",
          content,
          role_id: roleId,
          kind: "text",
        });
        setStreamingIdx(history.length - 1);
      }
      return { ...prev, history };
    });
  }

  function scheduleTokenPump() {
    if (tokenPumpScheduledRef.current) return;
    tokenPumpScheduledRef.current = true;
    const pump = () => {
      const next = tokenQueueRef.current.shift();
      if (!next) {
        tokenPumpScheduledRef.current = false;
        return;
      }
      appendTokenToSnapshot(next.content, next.role_id);
      // Keep yielding frames while the queue has more Expert tokens.
      if (tokenQueueRef.current.length > 0) {
        window.requestAnimationFrame(pump);
      } else {
        tokenPumpScheduledRef.current = false;
      }
    };
    window.requestAnimationFrame(pump);
  }

  function handleStreamEvent({ event, data }: { event: string; data: string }) {
    if (event === "state" && data) {
      const payload = JSON.parse(data) as ChatSnapshot;
      // Merge carefully: graph token events accumulate on the client, while
      // node `updates` may still carry a shorter placeholder. Prefer the
      // longer in-progress assistant tail so Expert streaming isn't wiped.
      setSnapshot((prev) => {
        if (!prev?.history?.length) return payload;
        const prevHist = prev.history;
        const nextHist = Array.isArray(payload.history) ? [...payload.history] : [];
        if (!nextHist.length) return payload;
        const prevLast = prevHist[prevHist.length - 1];
        const nextLast = nextHist[nextHist.length - 1];
        const prevText = (prevLast?.content || "").trim();
        const nextText = (nextLast?.content || "").trim();
        const nextLooksPlaceholder =
          Boolean(nextLast?.phase) ||
          /思考中|Thinking|正在设计|designing the pipeline|准备澄清|preparing clarification|撰写|Summarizing|正在总结/i.test(
            nextText
          );
        if (
          prevLast?.role === "assistant" &&
          nextLast?.role === "assistant" &&
          prevLast.kind !== "thinking" &&
          prevText.length > nextText.length &&
          (nextLooksPlaceholder || nextText.length === 0 || prevText.startsWith(nextText))
        ) {
          nextHist[nextHist.length - 1] = {
            ...nextLast,
            content: prevLast.content,
            role_id: prevLast.role_id || nextLast.role_id,
            kind: prevLast.kind || "text",
            phase: undefined,
          };
          return { ...payload, history: nextHist };
        }
        return payload;
      });
      // Don't kill the streaming caret on every node update — only clear when
      // the run reached a waiting/terminal status (forms need to mount).
      const status = String(payload.status || "").toLowerCase();
      if (
        status.startsWith("waiting_") ||
        status === "completed" ||
        status === "error" ||
        status === "planning_failed" ||
        status === "execution_failed" ||
        status === "stopped"
      ) {
        tokenQueueRef.current = [];
        tokenPumpScheduledRef.current = false;
        setStreamingIdx(-1);
      }
    } else if (event === "stream_start" && data) {
      const info = JSON.parse(data) as { role_id?: string; turn_id?: string; kind?: string };
      setSnapshot(prev => {
        if (!prev) return prev;
        const history = [...prev.history];
        const last = history[history.length - 1];
        // Prefer the structured `phase` marker (set by backend on placeholder
        // messages). Fall back to legacy substring matching while older backends
        // still in the wild may not emit `phase` yet.
        const isGraphPlaceholder =
          last && last.role === "assistant" && last.kind !== "thinking" && (
            Boolean(last.phase) ||
            last.content.includes("Thinking") || last.content.includes("思考中") ||
            last.content.includes("Summarizing") || last.content.includes("正在总结") ||
            last.content.includes("汇总") || last.content.includes("撰写小报告") ||
            last.content.includes("writing sub-report") ||
            last.content.includes("撰写研究草案") ||
            last.content.includes("writing the draft report")
          );
        if (isGraphPlaceholder && last) {
          // Graph engine: replace status placeholder with an empty answer bubble.
          history[history.length - 1] = {
            role: "assistant",
            content: "",
            role_id: info.role_id || last.role_id,
            kind: "text",
          };
          setStreamingIdx(history.length - 1);
          tokenQueueRef.current = [];
          tokenPumpScheduledRef.current = false;
        } else if (info.kind === "thinking") {
          // kimi-code reasoning stream only (explicit kind).
          if (last?.role === "assistant" && last.kind === "thinking") {
            history[history.length - 1] = {
              ...last,
              turn_id: info.turn_id || last.turn_id,
              role_id: info.role_id || last.role_id,
            };
            setStreamingIdx(history.length - 1);
          } else {
            history.push({
              role: "assistant",
              content: "",
              kind: "thinking",
              phase: "thinking",
              turn_id: info.turn_id,
              role_id: info.role_id,
            });
            setStreamingIdx(history.length - 1);
          }
        } else if (last?.role === "assistant" && last.kind === "thinking") {
          // Answer tokens after a Thinking block: open a fresh text bubble.
          history.push({ role: "assistant", content: "", role_id: info.role_id, kind: "text" });
          setStreamingIdx(history.length - 1);
        } else if (last?.role === "assistant" && !(last.content || "").trim()) {
          // Reuse blank assistant bubble (graph tokens may arrive before placeholder state).
          history[history.length - 1] = {
            ...last,
            content: "",
            role_id: info.role_id || last.role_id,
            kind: last.kind === "thinking" ? "text" : last.kind,
          };
          setStreamingIdx(history.length - 1);
        } else {
          // Science Expert / graph: never invent a thinking bubble on stream_start.
          history.push({ role: "assistant", content: "", role_id: info.role_id, kind: "text" });
          setStreamingIdx(history.length - 1);
        }
        return { ...prev, history };
      });
    } else if (event === "token" && data) {
      const token = JSON.parse(data) as { content?: string; role_id?: string };
      if (token.content) {
        // Queue + rAF pump: even if the proxy delivers many Expert frames in
        // one TCP read, the UI still paints them progressively.
        tokenQueueRef.current.push({
          content: token.content,
          role_id: token.role_id,
        });
        scheduleTokenPump();
      }
    } else if (event === "thinking" && data) {
      // kimi-code reasoning stream: accumulate into a dedicated
      // `kind:"thinking"` assistant message so the timeline can render it
      // as a collapsible block above the final answer.
      const evt = JSON.parse(data) as { content?: string; turn_id?: string };
      if (evt.content) {
        setSnapshot(prev => {
          if (!prev) return prev;
          const history = [...prev.history];
          const last = history[history.length - 1];
          if (last && last.role === "assistant" && last.kind === "thinking") {
            history[history.length - 1] = { ...last, content: last.content + evt.content };
          } else {
            history.push({ role: "assistant", content: evt.content || "", kind: "thinking", turn_id: evt.turn_id });
          }
          return { ...prev, history };
        });
      }
    } else if (event === "error" && data) {
      // SSE error frames (e.g. AskUser/Approve ACK failed) do not reject
      // streamSSEFromPost — surface them so the user can retry the gate.
      try {
        const payload = JSON.parse(data) as { message?: string; detail?: string };
        const msg = String(payload.message || payload.detail || "").trim();
        if (msg) setError(msg);
      } catch {
        const raw = String(data || "").trim();
        if (raw) setError(raw);
      }
      setStreamingIdx(-1);
      setRunStatus("stopped");
    }
  }

  async function sendMessage() {
    const composedText = message;
    const selectedCustomModel = customModels.find((item) => item.id === selectedModel);
    if (running) return;
    if (!message.trim() && files.length === 0 && workspaceFiles.length === 0) return;
    if (chatQuota?.enforced && (chatQuota.remaining ?? 0) <= 0) {
      const limit = chatQuota.limit ?? 3;
      setError(t.errOnlineLimit(limit));
      return;
    }
    setError("");
    setRunning(true);
    setRunStatus("running");
    setMessage("");
    abortRef.current = new AbortController();
    // Backend consumes one quota unit when the stream request is accepted;
    // update the pill immediately so a long Agent turn doesn't look "stuck".
    if (chatQuota?.enforced) noteQuotaSendStarted();
    // Optimistic sidebar title from the first user turn.
    const optimisticTitle = abbreviateSessionTitle(composedText);
    if (optimisticTitle && sessionId) {
      setSessions((prev) =>
        prev.map((item) =>
          item.session_id === sessionId && !(item.title || "").trim()
            ? { ...item, title: optimisticTitle }
            : item
        )
      );
    }

    try {
      let activeSessionId = sessionId;
      if (!activeSessionId) {
        const created = await createChatSession();
        activeSessionId = created.session_id;
        rememberOwnedSession(activeSessionId);
        setSessionId(activeSessionId);
        sessionStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
        if (chatMode === "science_agent") {
          setSelectedModel(SCIENCE_AGENT_MODEL_ID);
        } else {
          const createdId = modelLabelFromInternal(created.model_name);
          const nextId = isKimiEngineModel({ id: createdId, engine: undefined })
            ? pickExpertModelId(registry.data?.models || [], null)
            : createdId;
          setSelectedModel(nextId);
          rememberModelFromSession(created.model_name);
        }
        setSessions((prev) => {
          const exists = prev.some((s) => s.session_id === activeSessionId);
          if (exists) {
            return prev.map((item) =>
              item.session_id === activeSessionId && !(item.title || "").trim() && optimisticTitle
                ? { ...item, title: optimisticTitle }
                : item
            );
          }
          return [{
            session_id: activeSessionId,
            created_at: created.created_at,
            model_name: created.model_name,
            history_size: 0,
            status: "",
            title: optimisticTitle || "",
          }, ...prev];
        });
      } else if (optimisticTitle) {
        setSessions((prev) =>
          prev.map((item) =>
            item.session_id === activeSessionId && !(item.title || "").trim()
              ? { ...item, title: optimisticTitle }
              : item
          )
        );
      }

      let attachmentPaths: string[] = [];
      if (files.length > 0) {
        const uploaded = await uploadFiles(activeSessionId, files);
        attachmentPaths = uploaded.files.map((f) => f.path);
      }
      if (workspaceFiles.length > 0) {
        attachmentPaths = [...attachmentPaths, ...workspaceFiles.map((item) => item.storage_path)];
      }

      // Online Agent always sends the kimi sentinel; local Agent sends the
      // picker value so the backend can map it into kimi agent_config.model.
      // Custom OpenAI-style models stay Expert/local-only (not wired into kimi).
      const streamModel = selectedCustomModel
        ? selectedCustomModel.modelName
        : !isLocalMode
          ? chatMode === "science_agent"
            ? SCIENCE_AGENT_MODEL_ID
            : ONLINE_FIXED_EXPERT_MODEL_ID
          : chatMode === "science_agent" && isKimiEngineModel({ id: selectedModel, engine: "kimi-code" })
            ? SCIENCE_AGENT_MODEL_ID
            : selectedModel;
      await streamSSEFromPost(
        `/api/chat/sessions/${encodeURIComponent(activeSessionId)}/messages/stream`,
        {
          text: composedText,
          model: streamModel,
          chat_mode: chatMode,
          engine: chatMode === "science_agent" ? "kimi-code" : "graph",
          custom_model_config:
            selectedCustomModel && chatMode === "science_expert"
              ? {
                  model_name: selectedCustomModel.modelName,
                  api_key: selectedCustomModel.apiKey,
                  base_url: selectedCustomModel.baseUrl,
                }
              : undefined,
          custom_model_id:
            selectedCustomModel && chatMode === "science_expert" ? selectedCustomModel.id : "",
          attachment_paths: attachmentPaths,
          // Pin response language to the UI locale. Backend stores this on
          // session state, so retries and follow-up actions inherit it.
          lang,
        },
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(activeSessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
      } else {
        setMessage(composedText);
        setError(err instanceof Error ? err.message : t.errStreamMsg);
        setRunStatus("stopped");
      }
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      setMessage("");
      setFiles([]);
      setWorkspaceFiles([]);
      abortRef.current = null;
      await refreshChatQuota();
    }
  }

  async function retryLastMessage() {
    if (!sessionId || running) return;
    if (chatQuota?.enforced && (chatQuota.remaining ?? 0) <= 0) {
      const limit = chatQuota.limit ?? 3;
      setError(t.errOnlineLimit(limit));
      return;
    }
    if (!snapshot?.history?.some((h) => h.role === "user")) {
      setError(t.errNoUserMsg);
      return;
    }
    setError("");
    setRunning(true);
    setRunStatus("running");
    abortRef.current = new AbortController();
    if (chatQuota?.enforced) noteQuotaSendStarted();
    try {
      await streamSSEFromPost(
        `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages/retry/stream`,
        {},
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
      } else {
        setError(err instanceof Error ? err.message : t.errRetryMsg);
        setRunStatus("stopped");
      }
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      abortRef.current = null;
      await refreshChatQuota();
    }
  }

  async function abortRun() {
    setRunStatus("stopping");
    if (sessionId) {
      try {
        await cancelChatSession(sessionId);
      } catch {
        // best effort cancellation
      }
    }
    abortRef.current?.abort();
    setRunning(false);
    // Keep "Stopping" until backend status confirms stopped/completed.
    setTimeout(() => {
      void refreshCurrentSession();
    }, 600);
  }

  async function submitClarification(answers: ClarificationAnswer[]) {
    if (!sessionId || running) return;
    setError("");
    setRunning(true);
    setRunStatus("running");
    abortRef.current = new AbortController();
    try {
      await streamSSEFromPost(
        getClarificationRespondUrl(sessionId),
        { answers },
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errClarification);
      setRunStatus("stopped");
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      abortRef.current = null;
    }
  }

  async function submitAskUser(answers: ClarificationAnswer[]) {
    if (!sessionId || running) return;
    setError("");
    setModelSwitchNotice(t.noticeAskUserSubmitted);
    setRunning(true);
    setRunStatus("running");
    abortRef.current = new AbortController();
    try {
      await streamSSEFromPost(
        getAskUserRespondUrl(sessionId),
        { answers },
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errAskUser);
      setRunStatus("stopped");
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      abortRef.current = null;
    }
  }

  async function submitApproval(decision: ApprovalDecision) {
    if (!sessionId || running) return;
    setError("");
    setModelSwitchNotice(
      decision.decision === "approved" ? t.noticeApprovalApproved : t.noticeApprovalRejected
    );
    setRunning(true);
    setRunStatus("running");
    abortRef.current = new AbortController();
    try {
      await streamSSEFromPost(
        getApprovalDecideUrl(sessionId),
        {
          decision: decision.decision,
          selected_label: decision.selected_label || "",
          feedback: decision.feedback || "",
        },
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errApproval);
      setRunStatus("stopped");
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      abortRef.current = null;
    }
  }

  async function confirmPlan(plan: PlanStep[], autoExecute: boolean) {
    if (!sessionId || running) return;
    setError("");
    setRunning(true);
    setRunStatus("running");
    abortRef.current = new AbortController();
    try {
      await streamSSEFromPost(
        getPlanConfirmUrl(sessionId),
        { plan, auto_execute: autoExecute },
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errConfirmPlan);
      setRunStatus("stopped");
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      abortRef.current = null;
    }
  }

  async function handleIterationDecision(action: "satisfied" | "modify_plan" | "continue") {
    if (!sessionId || running) return;
    setError("");
    try {
      const result = await iterationDecide(sessionId, action);
      if (result.status === "waiting_for_plan_confirmation" && result.plan) {
        setSnapshot(prev => prev ? {
          ...prev,
          status: "waiting_for_plan_confirmation",
          waiting_for: "plan_confirmation",
          plan: result.plan!,
        } : prev);
      } else {
        setSnapshot(prev => prev ? {
          ...prev,
          status: result.status,
          waiting_for: "",
        } : prev);
      }
      await refreshCurrentSession();
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errIteration);
    }
  }

  async function handleStepDecision(action: "continue" | "abort") {
    if (!sessionId || running) return;
    setError("");
    setRunning(true);
    setRunStatus("running");
    abortRef.current = new AbortController();
    try {
      await streamSSEFromPost(
        getStepDecideUrl(sessionId),
        { action },
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errStepDecision);
      setRunStatus("stopped");
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      abortRef.current = null;
    }
  }

  async function handleSubReportDecision(
    action: "continue" | "skip" | "rewrite",
    comment?: string
  ): Promise<boolean> {
    if (!sessionId || running) return false;
    setError("");
    setRunning(true);
    setRunStatus("running");
    abortRef.current = new AbortController();
    try {
      await streamSSEFromPost(
        getSubReportDecideUrl(sessionId),
        { action, comment: comment || "" },
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
      return true;
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return false;
      }
      setError(err instanceof Error ? err.message : t.errSubReport);
      setRunStatus("stopped");
      return false;
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      abortRef.current = null;
    }
  }

  async function copySessionId(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedSessionId(value);
      window.setTimeout(() => setCopiedSessionId(""), COPY_HINT_MS);
    } catch {
      setError(t.errCopySession);
    }
  }

  async function handleDownloadReport() {
    if (!sessionId) return;
    setError("");
    try {
      await downloadExperimentReport(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errDownloadReport);
    }
  }

  function onComposerKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter") return;
    // Keep Shift+Enter for newline; ignore IME composition Enter.
    if (e.shiftKey || e.nativeEvent.isComposing) return;
    e.preventDefault();
    void sendMessage();
  }

  const waitingFor = snapshot?.waiting_for || "";
  const snapStatus = snapshot?.status || "";
  // Hard split: gate UI follows the *session* mode (snapshot), not the composer
  // toggle. Expert = LangGraph checkpoints only; Agent = kimi Ask/Approve only.
  const sessionMode: ChatMode =
    snapshot?.chat_mode === "science_agent" ||
    snapshot?.engine === "kimi-code" ||
    waitingFor === "kimi_question" ||
    waitingFor === "kimi_approval" ||
    snapStatus === "waiting_for_kimi_question" ||
    snapStatus === "waiting_for_kimi_approval"
      ? "science_agent"
      : snapshot?.chat_mode === "science_expert" ||
          snapshot?.engine === "graph" ||
          snapStatus === "waiting_for_clarification" ||
          snapStatus === "waiting_for_plan_confirmation" ||
          snapStatus === "waiting_for_sub_report_review" ||
          snapStatus === "waiting_for_step_review" ||
          snapStatus === "waiting_for_iteration"
        ? "science_expert"
        : chatMode;
  const isAgentSession = sessionMode === "science_agent";
  const isExpertSession = sessionMode === "science_expert";

  const isKimiAskUser =
    isAgentSession &&
    (waitingFor === "kimi_question" ||
      snapStatus === "waiting_for_kimi_question" ||
      Boolean(snapshot?.kimi_pending_question));
  const isKimiApproval =
    isAgentSession &&
    !isKimiAskUser &&
    (waitingFor === "kimi_approval" ||
      snapStatus === "waiting_for_kimi_approval" ||
      Boolean(snapshot?.kimi_pending_approval));
  const isExpertClarification =
    isExpertSession &&
    snapStatus === "waiting_for_clarification" &&
    (waitingFor === "clarification" || waitingFor === "");
  // If the graph paused on a sub-report, always show Continue/Rewrite.
  // (review_sub_reports only controls whether the graph pauses — not whether
  // the already-paused UI may render. Hiding the card left users stuck.)
  const isSubReportGate =
    isExpertSession && snapStatus === "waiting_for_sub_report_review";
  const isExpertPlanGate =
    isExpertSession && snapStatus === "waiting_for_plan_confirmation";
  const isExpertStepGate =
    isExpertSession && snapStatus === "waiting_for_step_review";
  const isExpertIterationGate =
    isExpertSession && snapStatus === "waiting_for_iteration";
  const isWaitingForInteraction =
    isKimiAskUser ||
    isKimiApproval ||
    isExpertClarification ||
    isExpertPlanGate ||
    isExpertStepGate ||
    isExpertIterationGate ||
    isSubReportGate;
  const composerWaitingText = (() => {
    if (isKimiApproval) return t.composerWaitingApproval;
    if (isKimiAskUser) return t.composerWaitingAskUser;
    switch (snapStatus) {
      case "waiting_for_clarification":
        return t.composerWaitingClarification;
      case "waiting_for_kimi_question":
        return t.composerWaitingAskUser;
      case "waiting_for_kimi_approval":
        return t.composerWaitingApproval;
      case "waiting_for_plan_confirmation":
        return t.composerWaitingPlan;
      case "waiting_for_sub_report_review":
        return t.composerWaitingSubReport;
      case "waiting_for_step_review":
        return t.composerWaitingStep;
      case "waiting_for_iteration":
        return t.composerWaitingIteration;
      default:
        return t.composerWaiting;
    }
  })();
  const hasReportData = Boolean(snapshot && (snapshot.tool_executions.length > 0 || snapshot.plan.length > 0));
  const quotaExhausted = Boolean(chatQuota?.enforced && (chatQuota.remaining ?? 0) <= 0);
  const sendTooltip = chatQuota?.enforced
    ? quotaExhausted
      ? t.quotaReached(chatQuota.limit ?? 3, chatQuota.limit ?? 3)
      : t.quotaRemaining(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)
    : t.sendMessage;
  const isLocalMode = chatQuota?.mode === "local";
  const registryModels: ModelSpec[] = registry.data?.models || [];
  const keyStatus: Record<string, boolean> = registry.data?.key_status || {};
  const defaultModelId = registry.data?.default_model || FALLBACK_MODEL_ID;
  type ModelOption = {
    value: string;
    label: string;
    disabled?: boolean;
    title?: string;
    provider?: string;
    requiresAdapter?: boolean;
    hasKey?: boolean;
  };
  // Local picker: graph models (+ custom in Expert). Agent still uses these ids
  // as the underlying kimi LLM (engine stays kimi-code).
  const expertRegistryModels = useMemo(
    () => registryModels.filter((m) => isGraphEngineModel(m)),
    [registryModels]
  );
  const modelOptions: ModelOption[] = [
    ...expertRegistryModels.map((m) => {
      const hasKey = keyStatus[m.provider] === true;
      const requiresAdapter = m.requires_adapter === true;
      const adapterReady = requiresAdapter ? Boolean(registry.data?.active_gateway) : true;
      const disabled = (requiresAdapter && !adapterReady) || m.disabled === true;
      const title = m.disabled
        ? (m.disabled_reason || t.titleGatewayRequired)
        : requiresAdapter && !adapterReady
        ? t.titleGatewayRequired
        : !hasKey
        ? t.titleMissingKey(m.provider)
        : t.titleModelInfo(m.label, m.provider);
      return {
        value: m.id,
        label: hasKey ? m.label : `${m.label}${t.noKeySuffix}`,
        disabled,
        title,
        provider: m.provider,
        requiresAdapter,
        hasKey,
      };
    }),
    // Custom OpenAI-style endpoints are Expert-only (not registered in kimi).
    ...(isLocalMode && chatMode === "science_expert"
      ? customModels.map((m) => ({ value: m.id, label: `${m.label}${t.customSuffix}` }))
      : []),
  ];

  const selectedModelSpec = registryModels.find((m) => m.id === selectedModel);
  const selectedProvider = selectedModelSpec?.provider || "";
  const selectedProviderHasKey = selectedProvider ? keyStatus[selectedProvider] === true : true;
  // Local: model picker for both Agent (→ kimi) and Expert (→ graph).
  const showLocalModelControls = isLocalMode;

  // Keep selected model aligned with chat mode once registry is available.
  useEffect(() => {
    if (registry.loading || !registry.data) return;
    // Online locks: Agent → kimi sentinel, Expert → fixed DeepSeek.
    if (!isLocalMode) {
      const locked =
        chatMode === "science_agent" ? SCIENCE_AGENT_MODEL_ID : ONLINE_FIXED_EXPERT_MODEL_ID;
      if (selectedModel !== locked) setSelectedModel(locked);
      return;
    }
    // Local Agent/Expert: keep a concrete graph (or Expert custom) model.
    const isCustomId =
      chatMode === "science_expert" && customModels.some((m) => m.id === selectedModel);
    const isExpertRegistryId = expertRegistryModels.some((m) => m.id === selectedModel);
    if (!isExpertRegistryId && !isCustomId) {
      setSelectedModel(pickExpertModelId(registryModels, selectedModel));
    }
  }, [
    registry.loading,
    registry.data,
    chatMode,
    selectedModel,
    registryModels,
    expertRegistryModels,
    customModels,
    isLocalMode,
  ]);

  useEffect(() => {
    if (isLocalMode) return;
    if (customModels.some((m) => m.id === selectedModel)) {
      setSelectedModel(
        chatMode === "science_agent"
          ? SCIENCE_AGENT_MODEL_ID
          : ONLINE_FIXED_EXPERT_MODEL_ID
      );
    }
  }, [isLocalMode, selectedModel, customModels, chatMode]);

  function openKeyPanelForProvider(provider: string) {
    setKeyPanelProvider(provider);
    setKeyPanelValue("");
  }

  function handleChatModeChange(next: ChatMode) {
    if (running || next === chatMode) return;
    if (next === "science_agent") {
      const agent = registryModels.find((m) => isKimiEngineModel(m));
      if (agent?.disabled) {
        setModelSwitchNotice(agent.disabled_reason || t.noticeAgentDisabled);
        return;
      }
      if (!agent && registry.data) {
        setModelSwitchNotice(t.noticeAgentDisabled);
        return;
      }
      setChatMode("science_agent");
      persistChatMode("science_agent");
      if (!isLocalMode) {
        setSelectedModel(SCIENCE_AGENT_MODEL_ID);
        setKeyPanelProvider("");
      } else {
        // Keep current graph model as the kimi underlying LLM when possible.
        const currentSpec = registryModels.find((m) => m.id === selectedModel);
        const keepCurrent = Boolean(
          currentSpec && isGraphEngineModel(currentSpec) && !currentSpec.disabled
        );
        const nextModel = keepCurrent
          ? selectedModel
          : pickExpertModelId(registryModels, selectedModel);
        setSelectedModel(nextModel);
        setKeyPanelProvider("");
      }
    } else {
      setChatMode("science_expert");
      persistChatMode("science_expert");
      if (!isLocalMode) {
        setSelectedModel(ONLINE_FIXED_EXPERT_MODEL_ID);
        setKeyPanelProvider("");
      } else {
        const currentSpec = registryModels.find((m) => m.id === selectedModel);
        const keepCurrent =
          Boolean(currentSpec && isGraphEngineModel(currentSpec) && !currentSpec.disabled) ||
          customModels.some((m) => m.id === selectedModel);
        const nextModel = keepCurrent
          ? selectedModel
          : pickExpertModelId(registryModels, selectedModel);
        setSelectedModel(nextModel);
        const spec = registryModels.find((m) => m.id === nextModel);
        if (spec && keyStatus[spec.provider] !== true && !isKimiEngineModel(spec)) {
          openKeyPanelForProvider(spec.provider);
        } else {
          setKeyPanelProvider("");
        }
      }
    }
    if ((snapshot?.history?.length || 0) > 0) {
      setModelSwitchNotice(t.noticeModeSwitched);
    }
  }

  function handleModelChange(next: string) {
    // Online: model is server-fixed (no client picker).
    if (!isLocalMode) return;
    if (next === OTHER_MODEL_OPTION) {
      // Custom endpoints are Expert-only.
      if (chatMode !== "science_expert") {
        setChatMode("science_expert");
        persistChatMode("science_expert");
      }
      setShowCustomModelModal(true);
      return;
    }
    const prev = selectedModel;
    setSelectedModel(next);
    // Stay in the current mode: Agent → kimi with this model; Expert → graph.
    if (prev !== next && (snapshot?.history?.length || 0) > 0) {
      setModelSwitchNotice(t.noticeModelSwitched);
    }
    // Expert/graph keys live in VF; Agent uses kimi's catalog — no VF key panel.
    if (chatMode === "science_expert") {
      const spec = registryModels.find((m) => m.id === next);
      if (spec && keyStatus[spec.provider] !== true) {
        openKeyPanelForProvider(spec.provider);
      } else {
        setKeyPanelProvider("");
      }
    } else {
      setKeyPanelProvider("");
    }
  }

  async function submitProviderKey() {
    if (!keyPanelProvider) return;
    const value = keyPanelValue.trim();
    setKeyPanelSaving(true);
    setError("");
    try {
      await setProviderKey(keyPanelProvider, value);
      setKeyPanelProvider("");
      setKeyPanelValue("");
      registry.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errSaveKey);
    } finally {
      setKeyPanelSaving(false);
    }
  }

  async function handleGatewayChange(next: string) {
    const gatewayId = next === "" ? null : next;
    setError("");
    try {
      await setActiveGateway(gatewayId);
      registry.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errSwitchGateway);
    }
  }

  async function exportCurrentSession() {
    if (!sessionId || running) return;
    setError("");
    setRunning(true);
    try {
      await exportChatSessionBundle(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errExportSession);
    } finally {
      setRunning(false);
    }
  }

  function saveCustomModel() {
    const label = customModelLabel.trim() || customModelName.trim();
    const modelName = customModelName.trim();
    const apiKey = customModelApiKey.trim();
    const baseUrl = customModelBaseUrl.trim();
    if (!label || !modelName || !apiKey || !baseUrl) {
      setError(t.errRequiredCustomModel);
      return;
    }
    const normalizedLabel = label.toLowerCase();
    const normalizedKey = `${baseUrl.toLowerCase()}::${modelName.toLowerCase()}`;
    const builtInNameConflict = registryModels.some(
      (m) => m.label.trim().toLowerCase() === normalizedLabel || m.id.trim().toLowerCase() === normalizedLabel
    );
    if (builtInNameConflict) {
      setError(t.errLabelConflictBuiltIn);
      return;
    }
    const labelConflict = customModels.some((m) => m.label.trim().toLowerCase() === normalizedLabel);
    if (labelConflict) {
      setError(t.errLabelConflict);
      return;
    }
    const endpointConflict = customModels.some(
      (m) => `${m.baseUrl.trim().toLowerCase()}::${m.modelName.trim().toLowerCase()}` === normalizedKey
    );
    if (endpointConflict) {
      setError(t.errEndpointConflict);
      return;
    }
    const item: OpenAIStyleModel = {
      id: `custom-${Date.now()}`,
      label,
      modelName,
      apiKey,
      baseUrl,
    };
    setCustomModels((prev) => [...prev, item]);
    setSelectedModel(item.id);
    setShowCustomModelModal(false);
    setCustomModelLabel("");
    setCustomModelName("");
    setCustomModelApiKey("");
    setCustomModelBaseUrl("https://api.openai.com/v1");
  }

  async function removeCustomModel(modelId: string) {
    setError("");
    try {
      await deleteCustomModelCache(modelId);
      setCustomModels((prev) => prev.filter((m) => m.id !== modelId));
      if (selectedModel === modelId) {
        setSelectedModel(
          chatMode === "science_agent"
            ? SCIENCE_AGENT_MODEL_ID
            : pickExpertModelId(registryModels, defaultModelId)
        );
        setModelSwitchNotice(t.noticeCustomRemoved);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errRemoveCustomModel);
    }
  }

  return (
    <div className="chat-page">
      <section className={`chat-grid${sessionsCollapsed ? " left-collapsed" : ""}${logsCollapsed ? " right-collapsed" : ""}`}>
        <aside
          className={`chat-panel left${sessionsCollapsed ? " collapsed" : ""}`}
          data-collapsed-label={t.sessions}
          onClick={sessionsCollapsed ? () => setSessionsCollapsed(false) : undefined}
        >
          <div className="session-panel-head" onClick={() => setSessionsCollapsed(!sessionsCollapsed)}>
            <h3>{t.sessions} <span className="panel-toggle-icon">{sessionsCollapsed ? "›" : "‹"}</span></h3>
          </div>
          {!sessionsCollapsed && (
            <>
              <button
                type="button"
                className="session-new-btn"
                onClick={() => void createAndActivateSession()}
                disabled={running}
              >
                {t.newSession}
              </button>
              <div className="session-list">
                {sessions.map((s) => {
                  const raw =
                    (s.session_id === sessionId ? activeSessionTitle : "") ||
                    s.title?.trim() ||
                    "";
                  const label = abbreviateSessionTitle(raw) || raw || t.newChat;
                  return (
                  <div
                    key={s.session_id}
                    className={s.session_id === sessionId ? "session-item active" : "session-item"}
                  >
                    <button
                      className="session-select-btn"
                      onClick={() => void refreshCurrentSession(s.session_id)}
                      disabled={running && s.session_id === sessionId}
                      title={label}
                    >
                      <span className="session-title-label">{label}</span>
                      <small className="session-time-label">{new Date(s.created_at).toLocaleString()}</small>
                      <small className="session-meta-label">{s.status || t.idle} · {s.history_size} {t.msgs}</small>
                    </button>
                    <button
                      type="button"
                      className="session-delete-btn"
                      onClick={(e) => { e.stopPropagation(); void deleteAndSelectNextSession(s.session_id); }}
                      disabled={running}
                      title={t.deleteSession}
                    >
                      ✕
                    </button>
                  </div>
                  );
                })}
                {sessions.length === 0 && <div className="session-empty">{t.noSessions}</div>}
              </div>
              {sessionId && (
                <div className="session-sidebar-footer">
                  <button
                    type="button"
                    className="session-copy-btn"
                    onClick={() => void copySessionId(sessionId)}
                    title={t.copyFullId}
                  >
                    {copiedSessionId === sessionId ? t.copied : t.copySessionId}
                  </button>
                </div>
              )}
            </>
          )}
        </aside>

        <section className="chat-panel center">
          <div className="timeline-column">
            <div className="timeline-wrap" ref={timelineRef}>
            {/* Expert 模式：时间线上方轻量流水线进度（不跟 Agent/kimi 耦合） */}
            {isExpertSession && snapshot?.status && (
              <PipelineProgress
                status={snapshot.status}
                plan={snapshot.plan || []}
                toolExecutions={(snapshot.tool_executions || []).filter(
                  (e) => !isResearchNoiseTool(e as ToolExecution)
                )}
              />
            )}
            <ChatTimeline
              items={snapshot?.history || []}
              streamingIndex={streamingIdx}
              onSuggestedPrompt={(text) => setMessage(text)}
              sessionId={sessionId}
              toolExecutions={(snapshot?.tool_executions || []).filter(
                (e) => !isResearchNoiseTool(e as ToolExecution)
              )}
              securityEvents={snapshot?.security_events || []}
              onRetry={() => void retryLastMessage()}
              retryDisabled={running || quotaExhausted}
              onQuoteReply={(text) => {
                const lines = text.split("\n").slice(0, 3);
                const quoted = lines.map((l) => `> ${l}`).join("\n");
                setMessage((prev) => `${quoted}\n\n${prev}`);
              }}
            />
            {/* Science Agent (kimi-code) gates — never mix with Expert forms */}
            {isAgentSession &&
              streamingIdx < 0 &&
              isKimiAskUser &&
              (snapshot?.clarification_questions?.length ?? 0) > 0 && (
                <div className="agent-gate-wrap">
                  <AskUserCard
                    key={`ask-${waitingFor}-${snapshot?.kimi_pending_question?.question_id || ""}-${(snapshot?.clarification_questions || []).map((q) => q.question).join("|").slice(0, 80)}`}
                    questions={snapshot!.clarification_questions}
                    onSubmit={submitAskUser}
                    disabled={running}
                  />
                </div>
              )}
            {isAgentSession && streamingIdx < 0 && isKimiApproval && (
                <div className="agent-gate-wrap">
                  <ApprovalCard
                    key={`appr-${waitingFor}-${snapshot?.kimi_pending_approval?.approval_id || ""}`}
                    toolName={snapshot?.kimi_pending_approval?.tool_name || ""}
                    prompt={
                      snapshot?.kimi_pending_approval?.approval_prompt ||
                      snapshot?.approval_prompt ||
                      ""
                    }
                    planMarkdown={
                      snapshot?.kimi_pending_approval?.plan_markdown ||
                      snapshot?.plan_markdown ||
                      ""
                    }
                    optionLabels={
                      snapshot?.kimi_pending_approval?.option_labels ||
                      snapshot?.clarification_questions?.[0]?.options ||
                      []
                    }
                    onDecide={submitApproval}
                    disabled={running}
                  />
                </div>
              )}
            {/* Science Expert (LangGraph PI→CB→MLS→SC) checkpoints only */}
            {isExpertSession &&
              streamingIdx < 0 &&
              isExpertClarification &&
              (snapshot?.clarification_questions?.length ?? 0) > 0 && (
                <div className="expert-checkpoint-card">
                  <div className="expert-checkpoint-title">{t.checkpointClarify}</div>
                  <ClarificationForm
                    key={`${snapshot?.waiting_for}-${(snapshot?.clarification_questions || []).map((q) => q.question).join("|").slice(0, 120)}`}
                    questions={snapshot!.clarification_questions}
                    onSubmit={submitClarification}
                    disabled={running}
                  />
                </div>
              )}
            {isExpertSession &&
              isExpertPlanGate &&
              (snapshot?.plan?.length ?? 0) > 0 &&
              streamingIdx < 0 && (
                <div className="expert-checkpoint-card">
                  <div className="expert-checkpoint-title">{t.checkpointPlan}</div>
                  <PlanEditor
                    plan={snapshot!.plan}
                    onConfirm={confirmPlan}
                    disabled={running}
                  />
                </div>
              )}
            {isExpertSession && isSubReportGate && (
              <div className="expert-checkpoint-card">
                <div className="expert-checkpoint-title">{t.checkpointSubReport}</div>
                <SubReportCheckpoint
                  onDecide={handleSubReportDecision}
                  disabled={running}
                />
              </div>
            )}
            {isExpertSession && isExpertStepGate && (
              <div className="expert-checkpoint-card">
                <div className="expert-checkpoint-title">{t.checkpointStep}</div>
                <StepCheckpoint
                  onDecide={handleStepDecision}
                  disabled={running}
                />
              </div>
            )}
            {isExpertSession && isExpertIterationGate && (
              <div className="expert-checkpoint-card">
                <div className="expert-checkpoint-title">{t.checkpointIteration}</div>
                <IterationDecision
                  onDecide={handleIterationDecision}
                  disabled={running}
                />
              </div>
            )}
            </div>
          </div>
          <div className="composer">
            <div className="composer-textarea-wrap">
              <textarea
                rows={4}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={onComposerKeyDown}
                placeholder={isWaitingForInteraction ? composerWaitingText : t.composerPlaceholder}
                disabled={running || quotaExhausted || isWaitingForInteraction}
              />
            </div>
            <div className={`composer-row${showLocalModelControls ? " composer-row--expert" : ""}`}>
              <div className="composer-controls">
                <div className="composer-mode-switch" role="group" aria-label={t.chatModeAria}>
                  <button
                    type="button"
                    className={`composer-mode-btn${chatMode === "science_agent" ? " is-active" : ""}`}
                    aria-pressed={chatMode === "science_agent"}
                    title={t.modeScienceAgent}
                    disabled={running}
                    onClick={() => handleChatModeChange("science_agent")}
                  >
                    {t.modeAgentShort}
                  </button>
                  <button
                    type="button"
                    className={`composer-mode-btn${chatMode === "science_expert" ? " is-active" : ""}`}
                    aria-pressed={chatMode === "science_expert"}
                    title={t.modeScienceExpert}
                    disabled={running}
                    onClick={() => handleChatModeChange("science_expert")}
                  >
                    {t.modeExpertShort}
                  </button>
                </div>
                {showLocalModelControls && (
                  <>
                    <label className="composer-select-shell">
                      <span className="composer-select-meta">
                        <span className="composer-select-label">{t.modelLabel}</span>
                        {chatMode === "science_agent" && (
                          <span className="composer-select-hint">{t.modelViaKimi}</span>
                        )}
                      </span>
                      <span className="composer-select-field">
                        <select
                          className="composer-model-select"
                          value={selectedModel}
                          onChange={(e) => handleModelChange(e.target.value)}
                          aria-label={t.modelAria}
                          title={
                            selectedProvider
                              ? selectedProviderHasKey
                                ? t.titleProviderWithKey(selectedProvider)
                                : t.titleProviderNoKey(selectedProvider)
                              : undefined
                          }
                        >
                          {modelOptions.map((m) => (
                            <option key={m.value} value={m.value} disabled={m.disabled} title={m.title}>
                              {m.label}
                              {m.disabled ? t.gatewayRequiredSuffix : ""}
                            </option>
                          ))}
                          {chatMode === "science_expert" && (
                            <option value={OTHER_MODEL_OPTION}>{t.otherModel}</option>
                          )}
                        </select>
                        <span className="composer-select-chevron" aria-hidden="true">
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="6 9 12 15 18 9" />
                          </svg>
                        </span>
                      </span>
                    </label>
                    {/* VF provider keys apply to Expert/graph only; Agent uses kimi's own catalog. */}
                    {chatMode === "science_expert" &&
                      selectedModelSpec &&
                      !isKimiEngineModel(selectedModelSpec) && (
                      <button
                        type="button"
                        className={`model-key-chip${selectedProviderHasKey ? " is-ok" : " is-missing"}`}
                        title={
                          selectedProviderHasKey
                            ? t.titleKeyConfigured(selectedProvider)
                            : t.titleMissingKey(selectedProvider)
                        }
                        onClick={() => openKeyPanelForProvider(selectedProvider)}
                      >
                        <span className="model-key-dot" aria-hidden="true" />
                        {selectedProviderHasKey ? t.keyOk : t.setKey}
                      </button>
                    )}
                    {chatMode === "science_expert" && (registry.data?.gateways?.length ?? 0) > 0 && (
                      <label className="composer-select-shell composer-select-shell--compact">
                        <span className="composer-select-meta">
                          <span className="composer-select-label">{t.gatewayLabel}</span>
                        </span>
                        <span className="composer-select-field">
                          <select
                            className="composer-gateway-select"
                            value={registry.data?.active_gateway || ""}
                            onChange={(e) => void handleGatewayChange(e.target.value)}
                            aria-label={t.gatewayAria}
                            title={t.activeGateway}
                          >
                            <option value="">{t.noGateway}</option>
                            {(registry.data?.gateways || []).map((g) => (
                              <option key={g.id} value={g.id}>
                                {g.label}
                              </option>
                            ))}
                          </select>
                          <span className="composer-select-chevron" aria-hidden="true">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                              <polyline points="6 9 12 15 18 9" />
                            </svg>
                          </span>
                        </span>
                      </label>
                    )}
                  </>
                )}
              </div>
              <div className="composer-actions">
                <div className="file-source-inline" ref={fileSourceMenuRef}>
                  <button
                    type="button"
                    className={`file-upload-icon-btn${running || quotaExhausted ? " disabled" : ""}${fileSourceMenuOpen ? " is-open" : ""}`}
                    title={t.uploadFiles}
                    aria-label={t.fileSourceAria}
                    aria-haspopup="menu"
                    aria-expanded={fileSourceMenuOpen}
                    disabled={running || quotaExhausted}
                    onClick={() => setFileSourceMenuOpen((v) => !v)}
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                    </svg>
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    onChange={(e) => {
                      setFiles(Array.from(e.target.files || []));
                      setFileSourceMenuOpen(false);
                    }}
                    disabled={running || quotaExhausted}
                    className="file-upload-hidden"
                  />
                  {fileSourceMenuOpen && (
                    <div className="file-source-menu" role="menu" aria-label={t.fileSourceAria}>
                      <button
                        type="button"
                        role="menuitem"
                        className="file-source-menu-item"
                        onClick={() => {
                          setFileSourceMenuOpen(false);
                          fileInputRef.current?.click();
                        }}
                      >
                        <span className="file-source-menu-title">{t.fileSourceLocal}</span>
                        <span className="file-source-menu-hint">{t.fileSourceLocalHint}</span>
                      </button>
                      <button
                        type="button"
                        role="menuitem"
                        className={`file-source-menu-item${workspaceEnabled ? "" : " is-disabled"}`}
                        disabled={!workspaceEnabled}
                        title={workspaceEnabled ? t.fromWorkspaceHint : t.fromWorkspaceOnlineHint}
                        onClick={() => {
                          if (!workspaceEnabled) return;
                          setFileSourceMenuOpen(false);
                          setWorkspacePickerOpen(true);
                        }}
                      >
                        <span className="file-source-menu-title">{t.fromWorkspace}</span>
                        <span className="file-source-menu-hint">
                          {workspaceEnabled ? t.fromWorkspaceHint : t.fromWorkspaceOnlineHint}
                        </span>
                      </button>
                    </div>
                  )}
                  <WorkspaceFilePicker
                    workspaceEnabled={workspaceEnabled}
                    disabled={running || quotaExhausted}
                    allowMultiple
                    hideTrigger
                    open={workspacePickerOpen}
                    onOpenChange={setWorkspacePickerOpen}
                    buttonLabel={t.fromWorkspace}
                    onPick={(picked) => {
                      setWorkspaceFiles(picked);
                      setWorkspacePickerOpen(false);
                    }}
                  />
                </div>
                {chatQuota?.enforced && (
                  <span
                    className={`chat-quota-pill${quotaExhausted ? " exhausted" : ""}`}
                    title={t.quotaPillTitle(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)}
                  >
                    {quotaExhausted
                      ? t.quotaPillExhausted(chatQuota.used ?? 0, chatQuota.limit ?? 3)
                      : t.quotaPillLabel(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)}
                  </span>
                )}
                <button
                  type="button"
                  className="composer-icon-btn"
                  onClick={() => void exportCurrentSession()}
                  disabled={running || !sessionId}
                  title={t.exportTooltip}
                  aria-label={t.export}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </button>
                {running ? (
                  <button
                    type="button"
                    className="composer-icon-btn composer-stop-btn"
                    onClick={abortRun}
                    title={t.stopTooltip}
                    aria-label={t.stop}
                  >
                    <span className="composer-stop-square" aria-hidden="true" />
                  </button>
                ) : (
                  <button
                    type="button"
                    className="composer-icon-btn composer-send-btn"
                    onClick={() => void sendMessage()}
                    disabled={quotaExhausted}
                    title={sendTooltip}
                    aria-label={t.sendTooltipShort}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M12 19V5" />
                      <path d="M5 12l7-7 7 7" />
                    </svg>
                  </button>
                )}
              </div>
            </div>
            {modelSwitchNotice && (
              <div className="model-switch-notice" role="status">
                {modelSwitchNotice}
              </div>
            )}
            {keyPanelProvider && (
              <div
                className="model-key-panel"
                role="dialog"
                aria-label={t.keyPanelAria(keyPanelProvider)}
                style={{
                  display: "flex",
                  gap: "8px",
                  alignItems: "center",
                  padding: "8px 10px",
                  marginTop: "6px",
                  border: "1px solid #e0c080",
                  background: "#fff8e6",
                  borderRadius: "6px",
                }}
              >
                <div style={{ display: "grid", gap: 4, flex: 1, minWidth: 0 }}>
                  <span style={{ fontSize: "13px" }}>
                    {t.keyPanelLabelPre}<strong>{keyPanelProvider}</strong>{t.keyPanelLabelPost}
                  </span>
                  <span style={{ fontSize: "12px", color: "var(--muted)" }}>{t.keyPanelHint}</span>
                </div>
                <input
                  type="password"
                  value={keyPanelValue}
                  onChange={(e) => setKeyPanelValue(e.target.value)}
                  placeholder="sk-..."
                  disabled={keyPanelSaving}
                  style={{ flex: 1, minWidth: 0 }}
                  autoFocus
                />
                <button
                  className="btn-primary"
                  onClick={() => void submitProviderKey()}
                  disabled={keyPanelSaving || !keyPanelValue.trim()}
                >
                  {keyPanelSaving ? t.keyPanelSaving : t.save}
                </button>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    setKeyPanelValue("");
                    void (async () => {
                      if (!keyPanelProvider) return;
                      setKeyPanelSaving(true);
                      setError("");
                      try {
                        await setProviderKey(keyPanelProvider, "");
                        setKeyPanelProvider("");
                        setKeyPanelValue("");
                        registry.refresh();
                      } catch (err) {
                        setError(err instanceof Error ? err.message : t.errSaveKey);
                      } finally {
                        setKeyPanelSaving(false);
                      }
                    })();
                  }}
                  disabled={keyPanelSaving}
                >
                  {t.clearKey}
                </button>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    setKeyPanelProvider("");
                    setKeyPanelValue("");
                  }}
                  disabled={keyPanelSaving}
                >
                  {t.cancel}
                </button>
              </div>
            )}
            {(files.length > 0 || workspaceFiles.length > 0) && (
              <div className="file-preview">
                {files.map((f) => (
                  <span key={f.name}>{f.name}</span>
                ))}
                {workspaceFiles.map((f) => (
                  <span key={f.id}>{f.display_name} {t.workspaceSuffix}</span>
                ))}
              </div>
            )}
            {error && <ErrorAlert message={error} onDismiss={() => setError("")} t={t} />}
          </div>
        </section>

        <aside
          className={`chat-panel right${logsCollapsed ? " collapsed" : ""}`}
          data-collapsed-label={t.executionStatus}
          onClick={logsCollapsed ? () => setLogsCollapsed(false) : undefined}
        >
          <div className="panel-toggle-head term-head" onClick={() => setLogsCollapsed(!logsCollapsed)}>
            <span className="term-head-dots">
              <span className="term-dot dot-red" />
              <span className="term-dot dot-yellow" />
              <span className="term-dot dot-green" />
            </span>
            <span className="term-head-title">{t.executionStatus}</span>
            <span className="panel-toggle-icon">{logsCollapsed ? "+" : "-"}</span>
          </div>
          {!logsCollapsed && (
            <>
              <div className="term-body">
                {!terminalData ? (
                  <div className="term-line"><span className="term-muted">{t.termWaiting}</span></div>
                ) : (
                  <>
                    <div className="term-section">
                      <div className="term-line">
                        <span className="term-prompt">$</span>
                        <span className="term-cmd">status</span>
                        <span className={`term-status-badge ${terminalData.status === "completed" ? "st-done" : runStatus === "running" ? "st-run" : "st-idle"}`}>
                          {terminalData.status}
                        </span>
                      </div>
                      <div className="term-line">
                        <span className="term-prompt">$</span>
                        <span className="term-cmd">info</span>
                        <span className="term-val">{t.termMessagesTools(terminalData.messages, terminalData.toolRuns)}</span>
                      </div>
                    </div>
                    {terminalData.tools.length > 0 && (
                      <div className="term-section">
                        <div className="term-line"><span className="term-prompt">$</span><span className="term-cmd">tools --recent</span></div>
                        {terminalData.tools.map((t, i) => (
                          <div key={i} className="term-line term-indent">
                            <span className={`term-tool-dot ${t.status === "failed" ? "dot-fail" : "dot-ok"}`} />
                            <span className="term-tool-name">{t.name}</span>
                            {t.ts && <span className="term-ts">{t.ts.split("T")[1]?.slice(0, 8) || t.ts}</span>}
                          </div>
                        ))}
                      </div>
                    )}
                    {terminalData.conv.length > 0 && (
                      <div className="term-section">
                        <div className="term-line"><span className="term-prompt">$</span><span className="term-cmd">log --tail 6</span></div>
                        {terminalData.conv.map((c, i) => (
                          <div key={i} className="term-line term-indent term-log-line">
                            <span className={`term-role ${c.role === "user" ? "role-user" : "role-agent"}`}>{c.role}</span>
                            <span className="term-log-content">{c.content || t.termEmpty}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
              <SessionFilesPanel
                sessionId={sessionId}
                authHeaders={sessionId ? getChatSessionAuthHeaders(sessionId) : undefined}
                liveRefresh={running}
                pollMs={5000}
              />
            </>
          )}
        </aside>
      </section>
      {showCustomModelModal && isLocalMode && (
        <div className="modal-backdrop">
          <div className="custom-model-modal">
            <h3>{t.addCustomModel}</h3>
            <label>
              {t.displayName}
              <input value={customModelLabel} onChange={(e) => setCustomModelLabel(e.target.value)} placeholder={t.displayNamePlaceholder} />
            </label>
            <label>
              {t.modelName}
              <input value={customModelName} onChange={(e) => setCustomModelName(e.target.value)} placeholder={t.modelNamePlaceholder} />
            </label>
            <label>
              {t.apiKey}
              <input type="password" value={customModelApiKey} onChange={(e) => setCustomModelApiKey(e.target.value)} placeholder="sk-..." />
            </label>
            <label>
              {t.baseUrl}
              <input value={customModelBaseUrl} onChange={(e) => setCustomModelBaseUrl(e.target.value)} placeholder="https://api.openai.com/v1" />
            </label>
            <div className="custom-model-modal-actions">
              <button className="btn-secondary" onClick={() => setShowCustomModelModal(false)}>{t.cancel}</button>
              <button className="btn-primary" onClick={saveCustomModel}>{t.confirm}</button>
            </div>
            <div className="custom-model-list">
              <h4>{t.addedModels}</h4>
              {customModels.length === 0 ? (
                <div className="custom-model-empty">{t.noCustomModels}</div>
              ) : (
                customModels.map((m) => (
                  <div key={m.id} className="custom-model-item">
                    <div className="custom-model-item-main">
                      <strong>{m.label}</strong>
                      <small>{m.modelName} | {m.baseUrl}</small>
                    </div>
                    <button className="btn-secondary custom-model-delete-btn" onClick={() => void removeCustomModel(m.id)}>
                      {t.deleteBtn}
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
      <PageFooter />
    </div>
  );
}
