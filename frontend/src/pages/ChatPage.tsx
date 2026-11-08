import { useChatStream } from "../lib/useChatStream";
import { sanitizeChatSnapshot } from "../lib/chatStreamState";
import { mergeHistoryActivity, submissionHistoryRow, type HistoryActivity } from "../lib/optimisticHistory";
import { NewChatStart } from "../components/NewChatStart";
import { ChatModeIcon } from "../components/ChatModeIcon";
import { UnifiedChatHistory } from "../components/UnifiedChatHistory";
import { claimNewChatDraft, newChatLocation, CHAT_MODE_LABELS, historyRowKey, type ChatHistoryRow, type ResearchHistoryRow } from "../lib/unifiedChatHistory";
import { InteractionCard } from "../components/InteractionCard";
import type { UserResponse } from "../lib/interactions";
import { manualQuestion } from "../lib/manualCatalog";
import { moveResearchDraft } from "../lib/researchContinuity";
import { chatHistoryWithPending, hasAcceptedTurn, type PendingChatTurn } from "../lib/pendingChatTurn";
import { useFileTransfer } from "../lib/useFileTransfer";
import { mergeFiles } from "../lib/fileTransfer";
import { validateUploads } from "../lib/uploadLimits";
import { QuotaNotice } from "../components/QuotaNotice";
import { openQuota, QUOTA_CHANGED, quotaRetryLabel, isDailyQuotaMessage } from "../lib/onlineQuota";
import { SidebarContent, useSidebar } from "../components/AppSidebar";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { chatSessionLocation, chatRouteKey } from "../lib/chatSessionRoute";
import { ResearcherPanel, type ResearcherPanelHandle } from "../components/ResearcherPanel";
import "../researcher.css";
import "../chat-layout.css";
import { ChatNotice, useDismissibleError } from "../components/ChatNotice";
import { AgentExecutionPicker } from "../components/AgentExecutionPicker";
import { ContextUsageRing } from "../components/ContextUsageRing";
import { isRunActive } from "../lib/runProgress";
import { RunSpinner } from "../components/RunSpinner";
import { activeSessionStorageKey, sessionsForMode, sessionCandidates } from "../lib/chatSessions";
import { SessionPreviewCache } from "../lib/sessionPreviewCache";
import { PageLoading } from "../components/PageLoading";
import { isFreeResearchModel } from "../lib/researchUi";
import { ScienceGoalPanel } from "../components/ScienceGoalPanel";
import { ChatTimeline } from "../components/ChatTimeline";
import { openStructure, isStructureFile } from "../lib/structurePreview";
import { SessionFilesPanel } from "../components/SessionFilesPanel";
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
  type AgentExecutionSelection,
  type ChatQuota,
  type ChatSnapshot,
  type ClarificationAnswer,
  type PlanStep,
  uploadFiles
} from "../lib/api";
import { streamSSEFromPost, streamSSEFromGet } from "../lib/sse";
import { WorkspaceFilePicker } from "../components/WorkspaceFilePicker";
import { type WorkspaceFile } from "../lib/workspaceApi";
import {
  useModelRegistry,
  setProviderKey,
  setActiveGateway,
  persistChatMode,
  pickExpertModelId,
  isKimiEngineModel,
  isGraphEngineModel,
  SCIENCE_AGENT_MODEL_ID,
  getExpertModelId,
  type ChatMode,
  type ModelSpec,
} from "../lib/useModelRegistry";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const STRINGS = {
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

type SessionMeta = {
  session_id: string;
  created_at: string;
  model_name: string;
  history_size: number;
  status: string;
  title?: string;
  has_content?: boolean;
  chat_mode?: ChatMode;
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

type ChatPageProps = {
  workspaceEnabled?: boolean;
};

export function ChatPage({ workspaceEnabled = false }: ChatPageProps) {
  const { lang, t: translate } = useLang();
  const t = translate(STRINGS);
  useDocumentMeta({ title: t.docTitle, description: t.docDescription });
  const { sessionId: routeSessionId = "" } = useParams();
  const navigate = useNavigate();
  const handledRoute = useRef<string | null>(null);
  const routeInitialized = useRef(false);
  const [sessionId, setSessionId] = useState<string>("");
  const sessionLoad = useRef({ target: "", revision: 0, pending: false });
  const sessionPreviews = useRef(new SessionPreviewCache<ChatSnapshot>());
  const sessionCreation = useRef<Promise<string> | null>(null);
  useLayoutEffect(() => {
    if (!sessionLoad.current.pending && sessionLoad.current.target !== sessionId) {
      sessionLoad.current = { target: sessionId, revision: sessionLoad.current.revision + 1, pending: false };
    }
  }, [sessionId]);
  const { snapshot, streamingIdx, setSnapshot, stopStream, discardStream, updateStream } = useChatStream();
  const [pendingTurn, setPendingTurn] = useState<PendingChatTurn | null>(null);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [historyActivity, setHistoryActivity] = useState<Record<string, HistoryActivity>>({});
  const sessionHistoryRevision = useRef(0);
  const onHistoryActivity = useCallback((key: string, row: HistoryActivity | null) => {
    sessionHistoryRevision.current++;
    setHistoryActivity(previous => { const next = { ...previous }; if (row) next[key] = row; else delete next[key]; return next; });
  }, []);
  const [message, setMessage] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [composerElement, setComposerElement] = useState<HTMLDivElement | null>(null);
  const [addingFiles, setAddingFiles] = useState(false);
  const addingFilesRef = useRef(false);
  const [workspaceFiles, setWorkspaceFiles] = useState<WorkspaceFile[]>([]);
  const [running, setRunning] = useState(false);
  const [scientistRunning, setGoalRunning] = useState(false);
  const [scientistSessionId, setScientistSessionId] = useState("");
  const [scientistDraftRevision, setScientistDraftRevision] = useState(0);
  const [goalStatuses, setGoalStatuses] = useState<Record<string, string>>({});
  const [runStatus, setRunStatus] = useState<RunStatus>("stopped");
  const [error, setError, dismissError] = useDismissibleError();
  const [noticeTarget, setNoticeTarget] = useState<HTMLDivElement | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const pendingSessionModel = useRef<string | null>(null);
  const [workspaceParams, setWorkspaceParams] = useSearchParams();
  const researcherMode = workspaceParams.get('mode') === 'researcher';
  const visibleModeRef = useRef(workspaceParams.get('mode'));
  visibleModeRef.current = workspaceParams.get('mode');
  const [researcherRunning, setResearcherRunning] = useState(false);
  const [chatMode, setChatMode] = useState<ChatMode>(() => workspaceParams.get("mode") === "scientist" ? "science_expert" : "science_agent");
  const activeModeRef = useRef(chatMode);
  const researcherPanelRef = useRef<ResearcherPanelHandle>(null);
  const [researchReports, setResearchReports] = useState<ResearchHistoryRow[]>([]);
  const goalRunning = chatMode === "science_expert" && scientistRunning;
  const [researchToolbar, setResearchToolbar] = useState<HTMLDivElement | null>(null);
  const [scienceComposer, setScienceComposer] = useState<HTMLDivElement | null>(null);
  const [scienceActions, setScienceActions] = useState<HTMLDivElement | null>(null);
  const [scienceFooter, setScienceFooter] = useState<HTMLDivElement | null>(null);
  const scrollPositions = useRef<Record<string, number>>({});
  const [agentExecution, setAgentExecution] = useState<AgentExecutionSelection | null>(null);
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
  const [startingTurn, setStartingTurn] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const agentWorking = !researcherMode && chatMode === "science_agent" && (running || isRunActive(snapshot?.status));
  function detachStream() {
    sessionPreviews.current.remember(snapshotRef.current);
    const controller = abortRef.current;
    abortRef.current = null;
    controller?.abort();
    runningRef.current = false;
    discardStream();
    setPendingTurn(null);
    setRunning(false); setStartingTurn(false); stopStream();
  }

  const snapshotRef = useRef<ChatSnapshot | null>(null);
  const runningRef = useRef(false);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const chatLayoutRef = useRef<HTMLDivElement | null>(null);
  useLayoutEffect(() => {
    const timeline = timelineRef.current;
    const layout = chatLayoutRef.current;
    if (!timeline || !layout) return;
    // Keep the composer aligned with content even with non-overlay scrollbars.
    const align = () => layout.style.setProperty(
      "--conversation-scrollbar-gutter", `${timeline.offsetWidth - timeline.clientWidth}px`,
    );
    align();
    const observer = new ResizeObserver(align);
    observer.observe(timeline);
    return () => observer.disconnect();
  }, []);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const fileSourceMenuRef = useRef<HTMLDivElement | null>(null);
  const [fileSourceMenuOpen, setFileSourceMenuOpen] = useState(false);
  const [workspacePickerOpen, setWorkspacePickerOpen] = useState(false);
  const LEGACY_SESSION_STORAGE_KEY = "vf2_active_session_id";
  const SESSION_CACHE_KEY = "vf2_session_list_cache_v2";
  const SESSION_OWNED_KEY = "vf2_owned_session_ids";
  const COPY_HINT_MS = 1200;
  const [copiedSessionId, setCopiedSessionId] = useState("");
  const modeChoiceRevision = useRef(0);
  const [agentCatalogReady, setAgentCatalogReady] = useState(false);
  const { closeOnNarrow } = useSidebar();
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

  const freshDraftId = workspaceParams.get("new") || "";
  const routeMode = researcherMode ? "researcher" : workspaceParams.get("mode") === "scientist" ? "science_expert" : "science_agent";
  useEffect(() => {
    const helpQuestion = !routeSessionId && routeMode === "science_agent" ? manualQuestion(workspaceParams.get("help"), lang) : "";
    const key = chatRouteKey(routeSessionId, routeMode) + (helpQuestion ? ":help:" + workspaceParams.get("help") : "") + (!routeSessionId && freshDraftId ? ":new:" + freshDraftId : "");
    if (handledRoute.current === key) return;
    const initial = !routeInitialized.current;
    routeInitialized.current = true;
    handledRoute.current = key;
    if (runningRef.current && (routeSessionId !== sessionLoad.current.target || routeMode !== activeModeRef.current)) {
      detachStream();
    }
    if (freshDraftId && !routeSessionId) {
      if (routeMode !== "researcher") {
        if (routeMode !== activeModeRef.current || researcherMode) handleChatModeChange(routeMode, "", false);
        createAndActivateSession(false, claimNewChatDraft(routeMode, freshDraftId, sessionStorage));
      }
      void refreshChatQuota().then(q => fetchSessions(q?.mode === "local")).catch(() => {});
      return;
    }
    if (routeMode === "researcher") {
      void refreshChatQuota().then(q => fetchSessions(q?.mode === "local")).catch(() => {});
      return;
    }
    if (helpQuestion) {
      if (activeModeRef.current !== 'science_agent') handleChatModeChange('science_agent', undefined, false);
      createAndActivateSession(false);
      setMessage(helpQuestion);
      void refreshChatQuota().then(q => fetchSessions(q?.mode === 'local')).catch(() => {});
      return;
    }
    // A URL is an explicit selection. Never replace it with a cached or
    // newest conversation when loading fails or the history list is stale.
    if (routeMode !== activeModeRef.current) {
      handleChatModeChange(routeMode, routeSessionId || undefined, false);
    } else if (routeSessionId) {
      void refreshChatQuota();
      void fetchSessions().catch(() => {});
      void refreshCurrentSession(routeSessionId, { syncComposer: true, updateUrl: false })
        .catch(err => setError(err instanceof Error ? err.message : t.errHintSession));
    } else if (initial) {
      void bootstrapSession();
    } else {
      createAndActivateSession(false);
    }
  }, [routeSessionId, routeMode, workspaceParams.get('help'), freshDraftId]);

  // Leaving Chat (other routes / remount) must detach the SSE reader so a
  // new mount can poll fresh state. Backend keeps the graph running.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, []);

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);
  useEffect(() => {
    runningRef.current = running;
  }, [running]);

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

  const visibleHistory = useMemo(() => chatHistoryWithPending(snapshot?.history || [], pendingTurn, sessionId), [snapshot?.history, pendingTurn, sessionId]);
  useEffect(() => {
    if (pendingTurn?.sessionId === sessionId && hasAcceptedTurn(snapshot?.history || [], pendingTurn)) setPendingTurn(null);
  }, [snapshot?.history, pendingTurn, sessionId]);
  useLayoutEffect(() => {
    if (pendingTurn && chatMode === "science_agent" && !researcherMode && timelineRef.current) timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
  }, [pendingTurn, chatMode, researcherMode]);
  const lastContentLen = visibleHistory[visibleHistory.length - 1]?.content?.length ?? 0;
  useEffect(() => {
    const el = timelineRef.current;
    if (!el || chatMode === "science_expert" || !visibleHistory.length) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 150) {
      el.scrollTop = el.scrollHeight;
    }
  }, [visibleHistory.length, lastContentLen, chatMode]);

  useLayoutEffect(() => {
    const el = timelineRef.current;
    if (el) el.scrollTop = scrollPositions.current[`${sessionId}:${chatMode}`] ??
      (chatMode === "science_agent" ? el.scrollHeight : 0);
  }, [chatMode, sessionId]);

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
      s === "waiting_for_input" ||
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

  // Soft-rejoin without hard refresh:
  // - bfcache / SPA return can restore a frozen Thinking UI
  // - background graph may finish after the first GET
  // Pull on mount, focus, pageshow(bfcache), visibility; burst-poll while
  // in-progress or a Thinking placeholder is still on screen.
  useEffect(() => {
    if (!sessionId || researcherMode || chatMode !== "science_agent") return;
    let cancelled = false;
    let ticks = 0; let pulling = false;

    const hasLiveThinking = (snap: ChatSnapshot | null) => {
      const hist = snap?.history || [];
      return hist.some(
        (h) =>
          h.kind === "thinking" ||
          h.phase === "thinking" ||
          /preparing clarification|准备澄清|analyzing your request|分析你的请求/i.test(
            h.content || ""
          )
      );
    };

    const isSettled = (status: string) => {
      const s = status.toLowerCase();
      return (
        !s ||
        s === "stopped" ||
        s === "completed" ||
        s === "error" ||
        s === "planning_failed" ||
        s === "execution_failed" ||
        s.startsWith("waiting_")
      );
    };

    const pull = (force = false) => {
      // Interval sync skips while a live SSE owns the UI; focus/bfcache
      // must force-refresh even if a frozen mount left running=true.
      // Never syncComposer — polling must not overwrite Expert/Agent toggle.
      if (cancelled || pulling || document.visibilityState === "hidden" || (!force && runningRef.current)) return;
      pulling = true;
      void refreshCurrentSession(sessionId, { syncComposer: false }).catch(() => {
        /* transient */
      }).finally(() => { pulling = false; });
    };

    pull(true);

    const onFocus = () => pull(true);
    const onVisible = () => {
      if (document.visibilityState === "visible") pull(true);
    };
    const onPageShow = (ev: PageTransitionEvent) => {
      // Back-forward cache restores the old React tree without remounting —
      // clear a zombie "running" flag so the UI can rejoin the checkpoint.
      if (ev.persisted) {
        setRunning(false);
        stopStream();
        setRunStatus("stopped");
        abortRef.current = null;
      }
      pull(true);
    };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("pageshow", onPageShow);

    const id = window.setInterval(() => {
      ticks += 1;
      const snap = snapshotRef.current;
      const status = String(snap?.status || "");
      const needsSync = !isSettled(status) || hasLiveThinking(snap);
      // Always burst a few times after mount (race with background save).
      if (needsSync || ticks <= 2) pull(false);
      if (!needsSync && ticks > 2) {
        window.clearInterval(id);

      }
    }, 3000);

    return () => {
      cancelled = true;
      window.clearInterval(id);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("pageshow", onPageShow);
    };
    // refreshCurrentSession closes over latest helpers; rebind on session change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, chatMode, researcherMode]);

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
        if ((item.title || "") === nextTitle && item.history_size === historySize && item.status === snapshot?.status) return item;
        changed = true;
        return { ...item, title: nextTitle, history_size: historySize, status: snapshot?.status || "" };
      });
      if (!prev.some(item => item.session_id === sessionId)) {
        changed = true;
        next.unshift({ session_id: sessionId, created_at: snapshot?.created_at || new Date().toISOString(),
          model_name: snapshot?.model_name || "", history_size: historySize,
          status: snapshot?.status || "", title: activeSessionTitle, has_content: true, chat_mode: activeModeRef.current });
      }
      if (!changed) return prev;
      try {
        localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  }, [sessionId, activeSessionTitle, snapshot?.history?.length, snapshot?.status]);

  useEffect(() => {
    let alive = true, pending = false;
    const refresh = async () => {
      if (!alive || pending) return;
      pending = true;
      try { await fetchSessions(); } catch { /* Reconnect on the next tick. */ }
      finally { pending = false; }
    };
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [chatQuota?.mode]);

  async function fetchSessions(localMode = chatQuota?.mode === "local") {
    const revision = sessionHistoryRevision.current;
    const data = await listChatSessions();
    const allServer = (data.sessions || []).filter(hasSessionContent);
    const list = allServer; // Server-side visitor ownership is authoritative.
    // Preserve locally derived titles when the API omits them (older server
    // process, or race before the first message is persisted).
    let merged: SessionMeta[] = [];
    if (revision !== sessionHistoryRevision.current) return list;
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
    // React may defer the updater; return the fetched identities immediately.
    return list;
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
    try { localStorage.setItem(SESSION_OWNED_KEY, JSON.stringify(Array.from(new Set(ids.filter(Boolean))))); } catch { /* Server owns the session list. */ }
  }

  function rememberOwnedSession(nextSessionId: string) {
    if (!nextSessionId) return;
    writeOwnedSessionIds([...readOwnedSessionIds(), nextSessionId]);
  }

  function forgetOwnedSession(targetSessionId: string) {
    writeOwnedSessionIds(readOwnedSessionIds().filter((item) => item !== targetSessionId));
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

  useEffect(() => {
    const changed = () => { void refreshChatQuota(); };
    window.addEventListener(QUOTA_CHANGED, changed);
    window.addEventListener("focus", changed);
    window.addEventListener("storage", changed);
    return () => { window.removeEventListener(QUOTA_CHANGED, changed); window.removeEventListener("focus", changed); window.removeEventListener("storage", changed); };
  }, []);

  function hasSessionContent(item: SessionMeta) {
    return item.has_content ?? Boolean(item.history_size || item.title?.trim());
  }

  function syncSessionUrl(sid: string, mode: ChatMode | "researcher", replace = false) {
    handledRoute.current = chatRouteKey(sid, mode);
    navigate(chatSessionLocation(lang, sid, mode, workspaceParams), { replace });
  }

  function rememberActiveSession(mode: ChatMode, sid: string) {
    try { sessionStorage.setItem(activeSessionStorageKey(mode), sid); } catch { /* URL restores the session. */ }
    if (mode === "science_expert") setScientistSessionId(sid);
  }

  async function ensureGoalSession() {
    if (scientistSessionId) return scientistSessionId;
    if (sessionCreation.current) return sessionCreation.current;
    sessionCreation.current = (async () => {
      const created = await createChatSession("science_expert");
      rememberOwnedSession(created.session_id);
      moveResearchDraft("scientist:draft", "scientist:" + created.session_id);
      rememberActiveSession("science_expert", created.session_id);
      if (activeModeRef.current === "science_expert" && visibleModeRef.current !== "researcher") {
        setSessionId(created.session_id);
        syncSessionUrl(created.session_id, "science_expert", true);
      }
      return created.session_id;
    })();
    try { return await sessionCreation.current; }
    finally { sessionCreation.current = null; }
  }

  function createAndActivateSession(updateUrl = true, resetDraft = true) {
    if (startingTurn || sessionCreation.current) return;
    detachStream();
    if (updateUrl) syncSessionUrl("", activeModeRef.current);
    // A new conversation starts as a local draft, not a persisted empty row.
    sessionLoad.current = { target: "", revision: sessionLoad.current.revision + 1, pending: false };
    rememberActiveSession(activeModeRef.current, "");
    setSessionId("");
    setSnapshot(null);
    snapshotRef.current = null;
    if (activeModeRef.current === "science_agent") {
      if (resetDraft) { setMessage(""); setFiles([]); setWorkspaceFiles([]); }
    } else if (resetDraft) {
      setScientistDraftRevision(value => value + 1);
    }
    setError("");
    setRunStatus("stopped");
    stopStream();
    setModelSwitchNotice("");
    closeOnNarrow();
  }

  async function deleteAndSelectNextSession(targetId: string) {
    const deletingActive = !researcherMode && targetId === sessionId;
    if (running && deletingActive) return false;
    setError("");
    try {
      await deleteChatSession(targetId);
      sessionPreviews.current.forget(targetId);
      forgetOwnedSession(targetId);
      if (deletingActive) {
        rememberActiveSession(activeModeRef.current, "");
        setSessionId("");
        setSnapshot(null);
        syncSessionUrl("", activeModeRef.current, true);
      }
      const list = await fetchSessions();
      if (!deletingActive) return true;

      const next = sessionsForMode(list, activeModeRef.current).find((item) => item.session_id !== targetId);
      if (next) {
        await refreshCurrentSession(next.session_id);
      }
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errDeleteSession);
      return false;
    }
  }

  async function bootstrapSession() {
    let bootstrapRevision = sessionLoad.current.revision;
    setError("");
    const quotaRequest = refreshChatQuota();
    const sessionsRequest = listChatSessions().catch(() => null);
    let list: SessionMeta[] = [];
    try {
      const raw = localStorage.getItem(SESSION_CACHE_KEY);
      if (raw) {
        const cached = JSON.parse(raw) as SessionMeta[];
        if (Array.isArray(cached) && cached.length > 0) {
          list = cached.filter(hasSessionContent);
          setSessions(list);
        }
      }
    } catch {
      // best effort cache read
    }
    try {
      const [quota, data] = await Promise.all([quotaRequest, sessionsRequest]);
      if (data) { list = (data.sessions || []).filter(hasSessionContent); setSessions(list); }
    } catch {
      // keep cached list if server refresh fails
    }
    const mode = activeModeRef.current;
    const remembered = sessionStorage.getItem(activeSessionStorageKey(mode));
    if (remembered === "") return;
    const candidates = sessionCandidates(list, mode, remembered, sessionStorage.getItem(LEGACY_SESSION_STORAGE_KEY));

    for (const sid of candidates) {
      if (sessionLoad.current.revision !== bootstrapRevision) return;
      if (!list.find((s) => s.session_id === sid)) continue;
      try {
        await refreshCurrentSession(sid, { localMode: (await quotaRequest)?.mode === "local", syncComposer: modeChoiceRevision.current === 0 });
        return;
      } catch {
        if (sessionLoad.current.target !== sid) return;
        bootstrapRevision = sessionLoad.current.revision;
        // inaccessible session — already cleaned up by refreshCurrentSession, try next
      }
    }

    if (sessionLoad.current.revision === bootstrapRevision) createAndActivateSession();
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

  async function refreshCurrentSession(
    targetId?: string,
    opts?: { localMode?: boolean; syncComposer?: boolean; mode?: ChatMode; updateUrl?: boolean }
  ) {
    const sid = targetId || sessionId;
    const requestedMode = opts?.mode ?? activeModeRef.current;
    if (!sid) return;
    // Soft-rejoin / focus polls pass the same session id — only sync the
    // composer mode when opening a different session (or first bootstrap).
    const switchingSession = !sessionId || Boolean(targetId && targetId !== sessionId);
    const syncComposer = opts?.syncComposer ?? switchingSession;
    const requestedModeRevision = modeChoiceRevision.current;
    const navigation = syncComposer || !sessionLoad.current.target;
    if (navigation) {
      if (sid !== sessionLoad.current.target) detachStream();
      sessionLoad.current = { target: sid, revision: sessionLoad.current.revision + 1, pending: true };
      setSessionId(sid);
      const cached = sessionPreviews.current.peek(sid);
      const preview = cached && (cached.chat_mode === "science_expert" ? "science_expert" : "science_agent") === requestedMode ? cached : null;
      setSnapshot(preview);
      snapshotRef.current = preview;
      if (opts?.updateUrl !== false) syncSessionUrl(sid, requestedMode);
    } else if (sessionLoad.current.pending || sessionLoad.current.target !== sid) return;
    const requestedSessionRevision = sessionLoad.current.revision;
    const isCurrentRequest = () => activeModeRef.current === requestedMode && sessionLoad.current.revision === requestedSessionRevision && sessionLoad.current.target === sid;
    try {
      const s = sanitizeChatSnapshot(await getChatSession(sid));
      if (!isCurrentRequest()) return;
      sessionPreviews.current.remember(s);
      const savedMode = s.chat_mode === "science_expert" ? "science_expert" : "science_agent";
      if (syncComposer && savedMode !== requestedMode) {
        sessionLoad.current.pending = false;
        handleChatModeChange(savedMode, sid);
        return;
      }
      setSnapshot(s);
      stopStream();
      setSessionId(sid);
      if (syncComposer) setModelSwitchNotice("");
      rememberActiveSession(requestedMode, sid);
      if (requestedMode === "science_agent" && s.active_run?.kind === "agent" && isRunActive(s.active_run.status) && !abortRef.current) void reconnectRun(s, isCurrentRequest);
      if (!syncComposer || requestedModeRevision !== modeChoiceRevision.current) return;
      const hasHistory = (s.history?.length ?? 0) > 0;
      const nextMode = requestedMode;
      // Opening a row cannot change its workspace or borrow another mode's model.
      if (nextMode === "science_expert") return;
      setAgentExecution(s.execution ? { harness: s.execution.harness, model_id: s.execution.model_id } : null);
      const savedModel = hasHistory ? String(s.kimi_context?.model || s.model_name || "") : "";
      const preferredModel = savedModel === SCIENCE_AGENT_MODEL_ID ? "" : savedModel;
      // Session and registry requests finish independently. Defer resolution
      // rather than freezing the old fallback before metadata is available.
      if (!registry.data) {
        pendingSessionModel.current = preferredModel;
        setSelectedModel("");
        return;
      }
      const normalizedModel = registry.data.models.find(m => m.id === preferredModel || `${m.provider}/${m.id}` === preferredModel)?.id || preferredModel;
      const modelId = normalizedModel || getExpertModelId(registry.data);
      const isLocal = opts?.localMode ?? chatQuota?.mode === "local";
      if (nextMode === "science_agent" && !isLocal) {
        setSelectedModel(SCIENCE_AGENT_MODEL_ID);
      } else {
        const graphId = isKimiEngineModel({ id: modelId, engine: s.engine })
          ? pickExpertModelId(registry.data?.models || [], null, {
              expertModel: getExpertModelId(registry.data),
            })
          : modelId;
        setSelectedModel(graphId);
        rememberModelFromSession(s.model_name);
      }
    } catch (err) {
      if (!isCurrentRequest()) return;
      const msg = err instanceof Error ? err.message : "";
      // A preview cannot stand in for a failed authenticated refresh.
      // A transient background poll must not blank the already loaded chat.
      if (navigation || /403|404|authentication|forbidden|not found|expired|denied/i.test(msg)) {
        sessionPreviews.current.forget(sid);
        setSnapshot(null);
        snapshotRef.current = null;
      }
      if (msg.includes("404") || msg.includes("not found") || msg.includes("Not Found")) {
        forgetOwnedSession(sid);
        if (sid === sessionId && routeSessionId !== sid) {
          setSessionId("");
          setSnapshot(null);
          rememberActiveSession(activeModeRef.current, "");
        }
        setSessions((prev) => prev.filter((s) => s.session_id !== sid));
      }
      throw err;
    } finally {
      if (isCurrentRequest()) sessionLoad.current.pending = false;
    }
  }

  async function reconnectRun(saved: ChatSnapshot, current: () => boolean) {
    const run = saved.active_run;
    if (!run || abortRef.current || !current()) return;
    const controller = new AbortController();
    abortRef.current = controller; runningRef.current = true; setRunning(true);
    const ownsView = () => abortRef.current === controller && current();
    try {
      await streamSSEFromGet(`/api/chat/sessions/${encodeURIComponent(saved.session_id)}/runs/${encodeURIComponent(run.id)}/events`,
        event => { if (ownsView()) handleStreamEvent(event); }, controller.signal,
        getChatSessionAuthHeaders(saved.session_id), saved.event_cursor || "");
    } catch (error) {
      if (ownsView() && !controller.signal.aborted) setError(error instanceof Error ? error.message : t.errStreamMsg);
    } finally {
      if (ownsView()) {
        abortRef.current = null; runningRef.current = false; setRunning(false); stopStream();
        await refreshCurrentSession(saved.session_id, { syncComposer: false });
      }
      await fetchSessions();
    }
  }

  function handleStreamEvent(event: { event: string; data: string }) {
    if (event.event !== "error") {
      const payload = updateStream(event);
      if (payload?.execution) setAgentExecution({ harness: payload.execution.harness, model_id: payload.execution.model_id });
      return;
    }
    // Interaction ACK failures arrive as SSE frames, not rejected requests.
    if (event.data) {
      try {
        const payload = JSON.parse(event.data) as { message?: string; detail?: string };
        const message = String(payload.message || payload.detail || "").trim();
        if (message) setError(message);
      } catch {
        const message = String(event.data).trim();
        if (message) setError(message);
      }
    }
    stopStream();
    setRunStatus("stopped");
  }

  async function addComposerFiles(picked: File[]) {
    if (!picked.length) return;
    if (addingFilesRef.current || running || goalRunning || isWaitingForInteraction) { setError(lang === "zh" ? "请等待当前操作完成后添加文件。" : "Wait for the current operation before attaching files."); return; }
    addingFilesRef.current = true; setAddingFiles(true);
    try {
      await validateUploads(picked);
      setFiles(previous => mergeFiles(previous, picked));
      setFileSourceMenuOpen(false); setError("");
    } catch (error) { setError((error as Error).message); }
    finally { addingFilesRef.current = false; setAddingFiles(false); }
  }

  async function sendMessage() {
    if (addingFilesRef.current || runningRef.current) return;
    if (goalRunning || ["science_expert"].includes(chatMode) || agentBlockedReason || isWaitingForInteraction) return;
    if (sessionLoad.current.pending || ((sessionId || routeSessionId) && !snapshot)) return;
    const composedText = message;
    const selectedCustomModel = customModels.find((item) => item.id === selectedModel);
    if (agentWorking) return;
    if (!message.trim() && files.length === 0 && workspaceFiles.length === 0) return;
    const submittedFiles = files;
    const submittedWorkspaceFiles = workspaceFiles;
    let pending: PendingChatTurn = {
      sessionId, text: composedText,
      attachments: [...files.map(file => file.name), ...workspaceFiles.map(file => file.display_name)],
      previousUsers: (snapshot?.history || []).filter(item => item.role === "user"),
    };
    // Paint this turn before creating the session, uploading files or waiting
    // for the first SSE state. Do not insert it into persisted server history.
    setPendingTurn(pending);
    runningRef.current = true;
    setError("");
    setRunning(true);
    setRunStatus("running");
    setMessage("");
    const controller = new AbortController();
    abortRef.current = controller;
    setStartingTurn(true);
    const ownsView = () => abortRef.current === controller;
    let activeSessionId = sessionId;
    let accepted = false;
    const checkSubmission = () => {
      if (!ownsView() || controller.signal.aborted) throw new DOMException("Aborted", "AbortError");
    };
    // Optimistic sidebar title from the first user turn.
    const optimisticTitle = abbreviateSessionTitle(composedText || pending.attachments.join(", ")) || t.newChat;
    const activityKey = crypto.randomUUID(), activityDate = Date.now();
    const showActivity = (id: string) => onHistoryActivity(activityKey, submissionHistoryRow("science_agent", activityKey, optimisticTitle, id, activityDate));
    showActivity(sessionId);
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
      if (!activeSessionId) {
        const created = await createChatSession("science_agent");
        activeSessionId = created.session_id;
        rememberOwnedSession(activeSessionId);
        checkSubmission();
        pending = { ...pending, sessionId: activeSessionId };
        showActivity(activeSessionId);
        setPendingTurn(pending);
        sessionLoad.current = { target: activeSessionId, revision: sessionLoad.current.revision + 1, pending: false };
        setSessionId(activeSessionId);
        rememberActiveSession("science_agent", activeSessionId);
        syncSessionUrl(activeSessionId, "science_agent", true);
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
            chat_mode: "science_agent",
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
      if (submittedFiles.length > 0) {
        const uploaded = await uploadFiles(activeSessionId, submittedFiles);
        checkSubmission();
        attachmentPaths = uploaded.files.map((f) => f.path);
      }
      if (submittedWorkspaceFiles.length > 0) {
        attachmentPaths = [...attachmentPaths, ...submittedWorkspaceFiles.map((item) => item.storage_path)];
      }
      checkSubmission();

      // Online Agent always sends the kimi sentinel; local Agent sends the
      // picker value so the backend can map it into kimi agent_config.model.
      // Custom OpenAI-style models stay Expert/local-only (not wired into kimi).
      const streamModel = selectedCustomModel
        ? selectedCustomModel.modelName
        : !isLocalMode
          ? chatMode === "science_agent"
            ? SCIENCE_AGENT_MODEL_ID
            : getExpertModelId(registry.data)
          : chatMode === "science_agent" && isKimiEngineModel({ id: selectedModel, engine: "kimi-code" })
            ? SCIENCE_AGENT_MODEL_ID
            : selectedModel;
      if (ownsView()) { setFiles([]); setWorkspaceFiles([]); }
      await streamSSEFromPost(
        `/api/chat/sessions/${encodeURIComponent(activeSessionId)}/messages/stream`,
        {
          text: composedText,
          execution: chatMode === "science_agent" && agentExecution ? agentExecution : undefined,
          model: agentExecution && chatMode === "science_agent" ? undefined : streamModel,
          chat_mode: chatMode,
          engine: agentExecution && chatMode === "science_agent" ? undefined : chatMode === "science_agent" ? "kimi-code" : "graph",
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
        (event) => {
          if (ownsView() && sessionLoad.current.target === activeSessionId) {
            if (event.event === "state" && event.data) {
              const state = JSON.parse(event.data) as ChatSnapshot;
              if (hasAcceptedTurn(state.history || [], pending)) { accepted = true; setPendingTurn(null); }
            }
            setStartingTurn(false); handleStreamEvent(event);
          }
        },
        controller.signal,
        getChatSessionAuthHeaders(activeSessionId)
      );
      void fetchSessions().catch(() => {});
      if (ownsView()) {
        const delivered = accepted || hasAcceptedTurn(snapshotRef.current?.history || [], pending);
        setMessage(delivered ? "" : composedText);
        setFiles(delivered ? [] : submittedFiles);
        setWorkspaceFiles(delivered ? [] : submittedWorkspaceFiles);
      }
    } catch (err) {
      if (!ownsView()) return;
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
      } else {
        const text = err instanceof Error ? err.message : t.errStreamMsg;
        setError(text.includes("new conversation allowance") || text.includes("quota key is invalid") ? "" : text);
        setRunStatus("stopped");
      }
      // Restore a rejected draft; an accepted turn must not become a duplicate
      // message in the composer after a later stream disconnect.
      if (!accepted && !hasAcceptedTurn(snapshotRef.current?.history || [], pending)) {
        setMessage(composedText); setFiles(submittedFiles); setWorkspaceFiles(submittedWorkspaceFiles);
      }
    } finally {
      onHistoryActivity(activityKey, null);
      if (!accepted && !pending.previousUsers.length && !hasAcceptedTurn(snapshotRef.current?.history || [], pending)) {
        setSessions(previous => previous.filter(item => item.session_id !== activeSessionId));
      }
      if (ownsView()) {
        runningRef.current = false;
        setPendingTurn(null);
        setRunning(false); setStartingTurn(false);
        stopStream();
        abortRef.current = null;
      }
      await refreshChatQuota();
    }
  }

  async function retryLastMessage() {
    if (!sessionId || agentWorking) return;
    if (!snapshot?.history?.some((h) => h.role === "user")) {
      setError(t.errNoUserMsg);
      return;
    }
    setError("");
    setRunning(true);
    setRunStatus("running");
    const controller = new AbortController();
    abortRef.current = controller;
    setStartingTurn(true);
    const ownsView = () => abortRef.current === controller;
    try {
      await streamSSEFromPost(
        `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages/retry/stream`,
        {},
        event => { if (ownsView()) { setStartingTurn(false); handleStreamEvent(event); } },
        controller.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (!ownsView()) return;
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
      } else {
        setError(err instanceof Error ? err.message : t.errRetryMsg);
        setRunStatus("stopped");
      }
    } finally {
      if (ownsView()) {
        setRunning(false); setStartingTurn(false);
        stopStream();
        abortRef.current = null;
      }
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
    if (!sessionId || agentWorking) return;
    setError("");
    setRunning(true);
    setRunStatus("running");
    const controller = new AbortController();
    abortRef.current = controller;
    setStartingTurn(true);
    const ownsView = () => abortRef.current === controller;
    try {
      await streamSSEFromPost(
        getClarificationRespondUrl(sessionId),
        { answers },
        event => { if (ownsView()) { setStartingTurn(false); handleStreamEvent(event); } },
        controller.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (!ownsView()) return;
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errClarification);
      setRunStatus("stopped");
    } finally {
      if (ownsView()) {
        setRunning(false); setStartingTurn(false);
        stopStream();
        abortRef.current = null;
      }
    }
  }

  async function submitInteraction(answer: UserResponse) {
    if (!sessionId || agentWorking) return;
    const sid = sessionId;
    setError(""); setRunning(true); setRunStatus("running"); setStartingTurn(true);
    const controller = new AbortController(); abortRef.current = controller;
    const ownsView = () => abortRef.current === controller;
    try {
      await streamSSEFromPost(`/api/chat/sessions/${encodeURIComponent(sid)}/interactions/respond/stream`, answer,
        event => { if (ownsView()) { setStartingTurn(false); handleStreamEvent(event); } },
        controller.signal, getChatSessionAuthHeaders(sid));
      await fetchSessions();
    } catch (err) {
      if (ownsView() && !(err instanceof DOMException && err.name === "AbortError")) {
        setError(err instanceof Error ? err.message : t.errAskUser);
        await refreshCurrentSession(sid);
      }
    } finally {
      if (ownsView()) { setRunning(false); setStartingTurn(false); stopStream(); abortRef.current = null; }
    }
  }

  async function submitAskUser(answers: ClarificationAnswer[]) {
    if (!sessionId || agentWorking) return;
    setError("");
    setModelSwitchNotice(t.noticeAskUserSubmitted);
    setRunning(true);
    setRunStatus("running");
    const controller = new AbortController();
    abortRef.current = controller;
    setStartingTurn(true);
    const ownsView = () => abortRef.current === controller;
    try {
      await streamSSEFromPost(
        getAskUserRespondUrl(sessionId),
        { answers },
        event => { if (ownsView()) { setStartingTurn(false); handleStreamEvent(event); } },
        controller.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (!ownsView()) return;
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errAskUser);
      setRunStatus("stopped");
    } finally {
      if (ownsView()) {
        setRunning(false); setStartingTurn(false);
        stopStream();
        abortRef.current = null;
      }
    }
  }

  async function submitApproval(decision: ApprovalDecision) {
    if (!sessionId || agentWorking) return;
    setError("");
    setModelSwitchNotice(
      decision.decision === "approved" ? t.noticeApprovalApproved : t.noticeApprovalRejected
    );
    setRunning(true);
    setRunStatus("running");
    const controller = new AbortController();
    abortRef.current = controller;
    setStartingTurn(true);
    const ownsView = () => abortRef.current === controller;
    try {
      await streamSSEFromPost(
        getApprovalDecideUrl(sessionId),
        {
          decision: decision.decision,
          selected_label: decision.selected_label || "",
          feedback: decision.feedback || "",
        },
        event => { if (ownsView()) { setStartingTurn(false); handleStreamEvent(event); } },
        controller.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (!ownsView()) return;
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errApproval);
      setRunStatus("stopped");
    } finally {
      if (ownsView()) {
        setRunning(false); setStartingTurn(false);
        stopStream();
        abortRef.current = null;
      }
    }
  }

  async function confirmPlan(plan: PlanStep[], autoExecute: boolean) {
    if (!sessionId || agentWorking) return;
    setError("");
    setRunning(true);
    setRunStatus("running");
    const controller = new AbortController();
    abortRef.current = controller;
    setStartingTurn(true);
    const ownsView = () => abortRef.current === controller;
    try {
      await streamSSEFromPost(
        getPlanConfirmUrl(sessionId),
        { plan, auto_execute: autoExecute },
        event => { if (ownsView()) { setStartingTurn(false); handleStreamEvent(event); } },
        controller.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (!ownsView()) return;
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errConfirmPlan);
      setRunStatus("stopped");
    } finally {
      if (ownsView()) {
        setRunning(false); setStartingTurn(false);
        stopStream();
        abortRef.current = null;
      }
    }
  }

  async function handleIterationDecision(action: "satisfied" | "modify_plan" | "continue") {
    if (!sessionId || agentWorking) return;
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
    if (!sessionId || agentWorking) return;
    setError("");
    setRunning(true);
    setRunStatus("running");
    const controller = new AbortController();
    abortRef.current = controller;
    setStartingTurn(true);
    const ownsView = () => abortRef.current === controller;
    try {
      await streamSSEFromPost(
        getStepDecideUrl(sessionId),
        { action },
        event => { if (ownsView()) { setStartingTurn(false); handleStreamEvent(event); } },
        controller.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
    } catch (err) {
      if (!ownsView()) return;
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errStepDecision);
      setRunStatus("stopped");
    } finally {
      if (ownsView()) {
        setRunning(false); setStartingTurn(false);
        stopStream();
        abortRef.current = null;
      }
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
    const controller = new AbortController();
    abortRef.current = controller;
    setStartingTurn(true);
    const ownsView = () => abortRef.current === controller;
    try {
      await streamSSEFromPost(
        getSubReportDecideUrl(sessionId),
        { action, comment: comment || "" },
        event => { if (ownsView()) { setStartingTurn(false); handleStreamEvent(event); } },
        controller.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
      return true;
    } catch (err) {
      if (!ownsView()) return false;
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return false;
      }
      setError(err instanceof Error ? err.message : t.errSubReport);
      setRunStatus("stopped");
      return false;
    } finally {
      if (ownsView()) {
        setRunning(false); setStartingTurn(false);
        stopStream();
        abortRef.current = null;
      }
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
    Boolean(snapshot?.pending_interaction) ||
    isKimiAskUser ||
    isKimiApproval ||
    isExpertClarification ||
    isExpertPlanGate ||
    isExpertStepGate ||
    isExpertIterationGate ||
    isSubReportGate;
  useFileTransfer(composerElement, {
    disabled: running || goalRunning || addingFiles || isWaitingForInteraction,
    label: lang === "zh" ? "松开以添加文件" : "Drop files to attach",
    unavailable: lang === "zh" ? "请等待当前操作完成后添加文件。" : "Wait for the current operation before attaching files.",
    unreadable: lang === "zh" ? "请选择文件，暂不支持文件夹。" : "Choose files; folders are not supported.",
    onFiles: addComposerFiles, onError: setError,
  }, !researcherMode && chatMode === "science_agent");
  const composerWaitingText = (() => {
    if (snapshot?.pending_interaction) return t.composerWaitingAskUser;
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
      ? quotaRetryLabel(chatQuota, lang)
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
      const disabled = !hasKey || !isFreeResearchModel(m) || (requiresAdapter && !adapterReady) || m.disabled === true;
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
  const showLocalModelControls = isLocalMode && ["science_agent"].includes(chatMode) && !agentExecution;

  // Keep selected model aligned with chat mode once registry is available.
  useEffect(() => {
    if (registry.loading || !registry.data) return;
    if (isLocalMode && pendingSessionModel.current !== null) {
      const saved = pendingSessionModel.current;
      pendingSessionModel.current = null;
      const preferred = registryModels.find(m => m.id === saved || `${m.provider}/${m.id}` === saved)?.id;
      setSelectedModel(pickExpertModelId(registryModels, preferred, {expertModel: getExpertModelId(registry.data)}));
      return;
    }
    // Online locks: Agent → kimi sentinel, Expert → model-config.yaml pin.
    if (!isLocalMode) {
      const locked =
        chatMode === "science_agent" ? SCIENCE_AGENT_MODEL_ID : getExpertModelId(registry.data);
      if (selectedModel !== locked) setSelectedModel(locked);
      return;
    }
    // Local Agent/Expert: keep a concrete graph (or Expert custom) model.
    const isCustomId =
      chatMode === "science_expert" && customModels.some((m) => m.id === selectedModel);
    const isExpertRegistryId = expertRegistryModels.some((m) => m.id === selectedModel);
    if (!isExpertRegistryId && !isCustomId) {
      setSelectedModel(
        pickExpertModelId(registryModels, selectedModel, {
          expertModel: getExpertModelId(registry.data),
        })
      );
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
          : getExpertModelId(registry.data)
      );
    }
  }, [isLocalMode, selectedModel, customModels, chatMode]);

  function openKeyPanelForProvider(provider: string) {
    setKeyPanelProvider(provider);
    setKeyPanelValue("");
  }

  function handleChatModeChange(next: ChatMode, targetId?: string, updateUrl = true) {
    if (startingTurn || (next === activeModeRef.current && !researcherMode)) return;
    detachStream();
    modeChoiceRevision.current += 1;
    activeModeRef.current = next;
    setChatMode(next);
    persistChatMode(next);
    setFileSourceMenuOpen(false);
    setWorkspacePickerOpen(false);
    setModelSwitchNotice("");
    setKeyPanelProvider("");
    setError("");
    stopStream();
    setRunStatus("stopped");
    const remembered = sessionStorage.getItem(activeSessionStorageKey(next));
    const nextId = targetId ?? remembered ?? ((next === "science_expert" ? scientistSessionId : "") || sessionCandidates(sessions, next, null, sessionStorage.getItem(LEGACY_SESSION_STORAGE_KEY))[0] || "");
    sessionLoad.current = { target: nextId, revision: sessionLoad.current.revision + 1, pending: false };
    setSessionId(nextId);
    setSnapshot(null);
    snapshotRef.current = null;
    rememberActiveSession(next, nextId);
    if (updateUrl) syncSessionUrl(nextId, next);
    if (nextId) void refreshCurrentSession(nextId, { syncComposer: true, mode: next, updateUrl: false })
      .catch(err => setError(err instanceof Error ? err.message : t.errCreateSession));
  }

  function handleModelChange(next: string) {
    // Online: model is server-fixed (no client picker).
    if (!isLocalMode) return;
    if (next === OTHER_MODEL_OPTION) {
      // Custom endpoints are Expert-only.
      if (chatMode !== "science_expert") {
        handleChatModeChange("science_expert");
        const nextParams = new URLSearchParams(workspaceParams);
        nextParams.set("mode", "scientist");
        setWorkspaceParams(nextParams, { replace: true });
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
            : pickExpertModelId(registryModels, defaultModelId, {
                expertModel: getExpertModelId(registry.data),
              })
        );
        setModelSwitchNotice(t.noticeCustomRemoved);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errRemoveCustomModel);
    }
  }

  function historyIsRunning(row: { session_id: string; status?: string }) {
    if (row.session_id === scientistSessionId && isRunActive(goalStatuses[row.session_id])) return true;
    if (row.session_id === sessionId && agentWorking) return true;
    return isRunActive(row.status);
  }

  const isNewChat = !routeSessionId && (researcherMode || !sessionId) && !startingTurn && !pendingTurn;
  const sharedHistory: ChatHistoryRow[] = mergeHistoryActivity([
    ...sessions.map(s => ({ id: s.session_id, mode: s.chat_mode || "science_agent" as const,
      title: (!researcherMode && s.session_id === sessionId ? activeSessionTitle : "") || s.title?.trim() || t.newChat,
      date: s.created_at, running: historyIsRunning(s), canDelete: true })),
    ...researchReports.map(r => ({ id: r.id, mode: "researcher" as const, title: r.title,
      date: r.created * 1000, running: ["queued", "running"].includes(r.status), canDelete: chatQuota?.mode === "online" })),
  ], historyActivity);
  const visibleHistoryMode = researcherMode ? "researcher" : chatMode;
  const historyActiveId = (researcherMode ? routeSessionId : sessionId) || Object.values(historyActivity).find(row => row.mode === visibleHistoryMode)?.id || "";
  const nativeUnavailable = registryModels.find(m => isKimiEngineModel(m))?.disabled;
  const agentBlockedReason = !agentCatalogReady
    ? (lang === "zh" ? "请选择可用的执行器和模型。" : "Choose an available executor and model.")
    : !isLocalMode ? ""
    : !agentExecution && nativeUnavailable
    ? (lang === "zh" ? "Kimi 暂不可用，请选择其他执行器。" : "Kimi is unavailable. Choose another executor.")
    : !agentExecution && (!selectedModelSpec || !isFreeResearchModel(selectedModelSpec))
    ? (lang === "zh" ? "请选择免费模型。" : "Select a free model.")
    : !agentExecution && !selectedProviderHasKey
    ? (lang === "zh" ? "请配置 API Key。" : "Add an API key for this model.") : "";

  return (
    <div className="chat-page" ref={chatLayoutRef} data-new-chat={isNewChat} data-research-mode={researcherMode ? "researcher" : chatMode}>
      <section className="chat-grid">
        <SidebarContent>
          <UnifiedChatHistory rows={sharedHistory} activeKey={historyRowKey({ id: historyActiveId, mode: visibleHistoryMode })}
            lang={lang} disabled={startingTurn}
            onOpen={row => { navigate(chatSessionLocation(lang, row.id, row.mode)); closeOnNarrow(); }}
            onDelete={row => row.mode === "researcher" ? researcherPanelRef.current?.deleteReport(row.id) || Promise.resolve(false) : deleteAndSelectNextSession(row.id)}/>
        </SidebarContent>

        <section className="chat-panel center quota-notice-anchor">
          <div className="chat-notice-stack" ref={setNoticeTarget}>
            <QuotaNotice quota={chatQuota} lang={lang} />
            {error && !isDailyQuotaMessage(error) && <ChatNotice lang={lang} message={/exceeds 10 MB|超过 10 MB|Each file must be 10 MB|turn limit|轮上限|message is too long|消息太长/i.test(error) ? error : friendlyErrorHint(error, t)} onDismiss={dismissError} />}
          </div>
          <div className="timeline-column">
            <div className="timeline-wrap" ref={timelineRef} onScroll={e => { scrollPositions.current[`${sessionId}:${chatMode}`] = e.currentTarget.scrollTop; }}>
            {isNewChat && <NewChatStart mode={routeMode} lang={lang} onChange={next => navigate(newChatLocation(lang, next, freshDraftId || crypto.randomUUID()), { replace: true })}/>}
            <ResearcherPanel onHistoryActivity={onHistoryActivity} ref={researcherPanelRef} freshDraftId={freshDraftId} onHistoryChange={setResearchReports} routeJobId={researcherMode ? routeSessionId : undefined} onJobChange={(id, replace = false) => { if (visibleModeRef.current === "researcher") syncSessionUrl(id, "researcher", replace); }} noticeTarget={noticeTarget} online={chatQuota?.mode === "online"} active={researcherMode} lang={lang} toolbarTarget={researchToolbar} composerTarget={scienceComposer} actionsTarget={scienceActions} footerTarget={scienceFooter} onRunning={setResearcherRunning}/>
            <ScienceGoalPanel onHistoryActivity={onHistoryActivity} noticeTarget={noticeTarget} online={!isLocalMode} toolbarTarget={researchToolbar} composerTarget={scienceComposer} actionsTarget={scienceActions} footerTarget={scienceFooter} sessionId={chatMode === "science_expert" ? sessionId : scientistSessionId} draftRevision={scientistDraftRevision} science={!researcherMode && chatMode === "science_expert"} busy={running}
              ensureSession={ensureGoalSession} onRunning={setGoalRunning}
              onGoalStatus={(sid, status) => setGoalStatuses(previous => previous[sid] === status ? previous : { ...previous, [sid]: status })}
              onChanged={async (sid) => { if (visibleModeRef.current !== "researcher" && activeModeRef.current === "science_expert" && sid === scientistSessionId) void refreshCurrentSession(sid, { syncComposer: false }); await fetchSessions(); }} />
            {!researcherMode && chatMode === "science_agent" && sessionLoad.current.pending && sessionId && !snapshot && !pendingTurn && <PageLoading/>}
            {!researcherMode && chatMode === "science_agent" && (!sessionId || snapshot || pendingTurn) && <ChatTimeline
              running={running}
              items={visibleHistory}
              streamingIndex={streamingIdx}
              onSuggestedPrompt={(text) => { setMessage(text); composerElement?.querySelector<HTMLTextAreaElement>("textarea")?.focus(); }}
              sessionId={sessionId}
              toolExecutions={(snapshot?.tool_executions || []).filter(
                (e) => !isResearchNoiseTool(e as ToolExecution)
              )}
              securityEvents={!researcherMode && snapshot?.security_events || []}
              onRetry={() => void retryLastMessage()}
              retryDisabled={running || goalRunning || Boolean(agentBlockedReason)}
              onQuoteReply={(text) => {
                const lines = text.split("\n").slice(0, 3);
                const quoted = lines.map((l) => `> ${l}`).join("\n");
                setMessage((prev) => `${quoted}\n\n${prev}`);
              }}
            />}
            {!researcherMode && chatMode === "science_agent" && snapshot?.pending_interaction &&
              <InteractionCard key={snapshot.pending_interaction.id} interaction={snapshot.pending_interaction}
                sessionId={sessionId} disabled={agentWorking} onSubmit={submitInteraction}/>}
            {/* Science Agent (kimi-code) gates — never mix with Expert forms */}
            {!researcherMode && isAgentSession &&
              streamingIdx < 0 &&
              isKimiAskUser &&
              (snapshot?.clarification_questions?.length ?? 0) > 0 && (
                <div className="agent-gate-wrap">
                  <AskUserCard
                    key={`ask-${waitingFor}-${!researcherMode && snapshot?.kimi_pending_question?.question_id || ""}-${(snapshot?.clarification_questions || []).map((q) => q.question).join("|").slice(0, 80)}`}
                    questions={snapshot!.clarification_questions}
                    onSubmit={submitAskUser}
                    disabled={running}
                  />
                </div>
              )}
            {!researcherMode && isAgentSession && streamingIdx < 0 && isKimiApproval && (
                <div className="agent-gate-wrap">
                  <ApprovalCard
                    key={`appr-${waitingFor}-${!researcherMode && snapshot?.kimi_pending_approval?.approval_id || ""}`}
                    toolName={!researcherMode && snapshot?.kimi_pending_approval?.tool_name || ""}
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
            {/* Science Expert (LangGraph PI→CB→MLS→SC) checkpoints only.
                Waiting gates must mount even if streamingIdx briefly lags —
                otherwise ClarificationForm stays hidden behind a Thinking pulse. */}
            {!researcherMode && isExpertSession &&
              isExpertClarification &&
              (snapshot?.clarification_questions?.length ?? 0) > 0 && (
                <div className="expert-checkpoint-card">
                  <div className="expert-checkpoint-title">{t.checkpointClarify}</div>
                  <ClarificationForm
                    key={`${!researcherMode && snapshot?.waiting_for}-${(snapshot?.clarification_questions || []).map((q) => q.question).join("|").slice(0, 120)}`}
                    questions={snapshot!.clarification_questions}
                    onSubmit={submitClarification}
                    disabled={running}
                  />
                </div>
              )}
            {!researcherMode && isExpertSession &&
              isExpertPlanGate &&
              (snapshot?.plan?.length ?? 0) > 0 && (
                <div className="expert-checkpoint-card">
                  <div className="expert-checkpoint-title">{t.checkpointPlan}</div>
                  <PlanEditor
                    plan={snapshot!.plan}
                    onConfirm={confirmPlan}
                    disabled={running}
                  />
                </div>
              )}
            {!researcherMode && isExpertSession && isSubReportGate && (
              <div className="expert-checkpoint-card">
                <div className="expert-checkpoint-title">{t.checkpointSubReport}</div>
                <SubReportCheckpoint
                  onDecide={handleSubReportDecision}
                  disabled={running}
                />
              </div>
            )}
            {!researcherMode && isExpertSession && isExpertStepGate && (
              <div className="expert-checkpoint-card">
                <div className="expert-checkpoint-title">{t.checkpointStep}</div>
                <StepCheckpoint
                  onDecide={handleStepDecision}
                  disabled={running}
                />
              </div>
            )}
            {!researcherMode && isExpertSession && isExpertIterationGate && (
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
          <div className="composer" ref={setComposerElement} data-science={!researcherMode && chatMode === "science_expert"}>
            <div className="scientist-composer-input" ref={setScienceComposer} />
            {!researcherMode && chatMode === "science_agent" && (<div className="composer-textarea-wrap">
              <textarea
                rows={4}
                aria-label={lang === "zh" ? "消息" : "Message"}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={onComposerKeyDown}
                placeholder={isWaitingForInteraction ? composerWaitingText : t.composerPlaceholder}
                disabled={agentWorking || goalRunning || isWaitingForInteraction}
              />
            </div>)}
            <div className={`composer-row${showLocalModelControls ? " composer-row--expert" : ""}`}>
              <div className={`composer-controls${!researcherMode && chatMode === "science_agent" ? " composer-controls--agent" : ""}`}>
        <span className="research-workspace-header chat-current-mode" aria-label={lang === "zh" ? "当前模式" : "Current mode"}>
          <ChatModeIcon mode={researcherMode ? "researcher" : chatMode}/><span>{CHAT_MODE_LABELS[researcherMode ? "researcher" : chatMode]}</span>
        </span>
        <div className="research-execution-controls" ref={setResearchToolbar}>
{<div className="agent-execution-controls" hidden={researcherMode || chatMode !== "science_agent"}><AgentExecutionPicker online={!isLocalMode}
                  value={agentExecution} onChange={(next, userInitiated = true) => {
                    if (userInitiated) modeChoiceRevision.current += 1;
                    setAgentExecution(next);
                  }} lang={lang} onReadyChange={setAgentCatalogReady}
                  locked={Boolean(snapshot?.execution) || Boolean(snapshot?.history?.some(h => h.role === "user"))}
                  busy={running || goalRunning || Boolean(sessionId && !snapshot)}
                /></div>}
                {!researcherMode && showLocalModelControls && (
                  <>
                    <label className="composer-select-shell">
                      <span className="composer-select-meta">
                        <span className="composer-select-label">{t.modelLabel}</span>
                        {!researcherMode && chatMode === "science_agent" && (
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
                          {!researcherMode && chatMode === "science_expert" && (
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
                    {!researcherMode && chatMode === "science_expert" &&
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
                    {!researcherMode && chatMode === "science_expert" && (registry.data?.gateways?.length ?? 0) > 0 && (
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
              </div>
              <div className="composer-actions">
                <div className="scientist-composer-actions" ref={setScienceActions} />
                {!researcherMode && chatMode === "science_agent" && <>
                <ContextUsageRing context={snapshot?.agent_context} nativeContext={snapshot?.kimi_context} lang={lang} />
                <div className="file-source-inline" ref={fileSourceMenuRef}>
                  <button
                    type="button"
                    className={`file-upload-icon-btn${running ? " disabled" : ""}${fileSourceMenuOpen ? " is-open" : ""}`}
                    title={t.uploadFiles}
                    aria-label={t.fileSourceAria}
                    aria-haspopup="menu"
                    aria-expanded={fileSourceMenuOpen}
                    disabled={running}
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
                      const picked = Array.from(e.target.files || []);
                      e.target.value = "";
                      void addComposerFiles(picked);
                    }}
                    disabled={running}
                    className="file-upload-hidden" hidden tabIndex={-1}
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
                    disabled={running}
                    allowMultiple
                    hideTrigger
                    open={workspacePickerOpen}
                    onOpenChange={setWorkspacePickerOpen}
                    buttonLabel={t.fromWorkspace}
                    onPick={(picked) => {
                      setWorkspaceFiles(previous => [...previous, ...picked.filter(file => !previous.some(old => old.id === file.id))]);
                      setWorkspacePickerOpen(false);
                    }}
                  />
                </div>

                {agentWorking ? (
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
                    disabled={addingFiles || Boolean((sessionId || routeSessionId) && !snapshot) || goalRunning || Boolean(agentBlockedReason) || isWaitingForInteraction || (!message.trim() && !files.length && !workspaceFiles.length)}
                    title={sendTooltip}
                    aria-label={t.sendTooltipShort}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M12 19V5" />
                      <path d="M5 12l7-7 7 7" />
                    </svg>
                  </button>
                )}
                </>}
              </div>
            </div>
            <div className="composer-guidance">
              {!researcherMode && chatMode === "science_agent" && <span className="composer-help" role={agentBlockedReason || modelSwitchNotice ? "status" : undefined} title={agentBlockedReason || modelSwitchNotice || undefined}>{agentBlockedReason || modelSwitchNotice || (lang === "zh" ? "Enter 发送 · Shift + Enter 换行" : "Enter to send · Shift + Enter for a new line")}</span>}


              <div className="scientist-composer-footer" ref={setScienceFooter}/>
            </div>
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
            {!researcherMode && chatMode === "science_agent" && (files.length > 0 || workspaceFiles.length > 0) && (
              <div className="file-preview" aria-label={lang === "zh" ? "待发送文件" : "Attached files"}>
                {files.map((f, index) => (
                  <span key={`${f.name}:${f.size}:${f.lastModified}`}><span title={f.name}>{isStructureFile(f.name) ? <button type="button" className="attachment-structure-link" onClick={() => { const url = URL.createObjectURL(f); openStructure({ path: f.name, name: f.name, url }); }}>{f.name}</button> : f.name}</span><button type="button" disabled={running} aria-label={`${lang === "zh" ? "移除 " : "Remove "}${f.name}`} onClick={() => setFiles(previous => previous.filter((_, i) => i !== index))}>×</button></span>
                ))}
                {workspaceFiles.map((f) => (
                  <span key={f.id}><span title={f.display_name}>{f.display_name} {t.workspaceSuffix}</span><button type="button" disabled={running} aria-label={`${lang === "zh" ? "移除 " : "Remove "}${f.display_name}`} onClick={() => setWorkspaceFiles(previous => previous.filter(file => file.id !== f.id))}>×</button></span>
                ))}
              </div>
            )}
          </div>
        </section>

        {researcherMode && <div className="chat-file-actions">{chatQuota?.enforced && (
                  <span
                    className={`chat-quota-pill${quotaExhausted ? " exhausted" : ""}`}
                    role="button" tabIndex={0} onClick={() => openQuota()} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openQuota(); } }}
                    title={t.quotaPillTitle(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)}
                  >
                    {quotaExhausted
                      ? t.quotaPillExhausted(chatQuota.used ?? 0, chatQuota.limit ?? 3)
                      : t.quotaPillLabel(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)}
                  </span>
                )}</div>}
        {!researcherMode && <SessionFilesPanel sessionId={sessionId} authHeaders={sessionId ? getChatSessionAuthHeaders(sessionId) : undefined} liveRefresh={running || goalRunning} pollMs={5000} actions={<button
                  type="button"
                  className="session-files-toggle chat-export-button"
                  onClick={() => void exportCurrentSession()}
                  disabled={running || goalRunning || !sessionId}
                  title={t.exportTooltip}
                  aria-label={t.export}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </button>} quota={<>{chatQuota?.enforced && (
                  <span
                    className={`chat-quota-pill${quotaExhausted ? " exhausted" : ""}`}
                    role="button" tabIndex={0} onClick={() => openQuota()} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openQuota(); } }}
                    title={t.quotaPillTitle(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)}
                  >
                    {quotaExhausted
                      ? t.quotaPillExhausted(chatQuota.used ?? 0, chatQuota.limit ?? 3)
                      : t.quotaPillLabel(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)}
                  </span>
                )}</>}/>}
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
    </div>
  );
}
