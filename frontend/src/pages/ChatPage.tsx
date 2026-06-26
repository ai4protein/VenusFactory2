import { useEffect, useMemo, useRef, useState } from "react";
import { ChatTimeline } from "../components/ChatTimeline";
import { SessionFilesPanel } from "../components/SessionFilesPanel";
import { PipelineProgress } from "../components/PipelineProgress";
import { ClarificationForm } from "../components/ClarificationForm";
import { IterationDecision } from "../components/IterationDecision";
import { PlanEditor } from "../components/PlanEditor";
import { StepCheckpoint } from "../components/StepCheckpoint";
import { SubReportCheckpoint } from "../components/SubReportCheckpoint";
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
  type ModelSpec,
} from "../lib/useModelRegistry";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const STRINGS = {
  en: {
    docTitle: "Agent Chat — VenusFactory2",
    docDescription: "Chat with the protein-engineering agent to run predictions, training and analysis tasks.",
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
    noticeCustomRemoved: "Custom model removed. Active session switched back to default model context.",
    // Tooltips for model selector
    titleGatewayRequired: "Requires a configured gateway",
    titleMissingKey: (provider: string) => `Missing API key for ${provider}. Click to add.`,
    titleModelInfo: (label: string, provider: string) => `${label} (${provider})`,
    titleProviderWithKey: (provider: string) => `Provider: ${provider} (key configured)`,
    titleProviderNoKey: (provider: string) => `Provider: ${provider} (no API key)`,
    titleKeyConfigured: (provider: string) => `API key configured for ${provider}`,
    // Tooltips
    sendMessage: "Send message",
    regenerateLast: "Regenerate last message",
    quotaReached: (used: number, limit: number) => `Daily quota reached (${used}/${limit}).`,
    quotaRemaining: (remaining: number, limit: number) => `Online mode quota: ${remaining}/${limit} chats remaining for this IP today.`,
    quotaRegenerate: (remaining: number, limit: number) => `Regenerate also consumes quota. Remaining: ${remaining}/${limit}.`,
    // Header
    chat: "Chat",
    modeOnline: "Mode: Online",
    modeOnlineTooltip: (limit: number) => `Mode: Online. Per-user daily limit: ${limit} chats.`,
    onlineLocalHint: "For unlimited and more efficient usage, local deployment is recommended.",
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
    composerWaiting: "Please respond to the form above...",
    composerPlaceholder: "Ask anything about AI protein engineering...",
    quotaPillTitle: (limit: number) => `Per-IP daily limit in online mode: ${limit}`,
    modelAria: "Model",
    gatewayRequiredSuffix: " (gateway required)",
    otherModel: "Other Model...",
    noKeySuffix: " (no key)",
    customSuffix: " (Custom)",
    keyOk: "key ok",
    setKey: "set key",
    gatewayAria: "Gateway",
    activeGateway: "Active gateway",
    noGateway: "No gateway",
    uploadFiles: "Upload files",
    fromWorkspace: "From Workspace",
    regenerate: "Regenerate",
    export: "Export",
    stop: "Stop",
    send: "Send",
    runningEllipsis: "Running...",
    searchMessages: "Search messages",
    searchPlaceholder: "Search messages...",
    pipelineDismiss: "Dismiss",
    // Key panel
    keyPanelAria: (provider: string) => `Set API key for ${provider}`,
    keyPanelLabelPre: "API key for ",
    keyPanelLabelPost: ":",
    keyPanelSaving: "Saving...",
    save: "Save",
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
    docTitle: "智能体对话 — VenusFactory2",
    docDescription: "通过对话使用蛋白质工程智能体，运行预测、训练与分析任务。",
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
    noticeCustomRemoved: "自定义模型已删除。当前会话已切换回默认模型上下文。",
    // Tooltips for model selector
    titleGatewayRequired: "需要先配置网关",
    titleMissingKey: (provider: string) => `缺少 ${provider} 的 API 密钥，点击添加。`,
    titleModelInfo: (label: string, provider: string) => `${label}（${provider}）`,
    titleProviderWithKey: (provider: string) => `服务商：${provider}（密钥已配置）`,
    titleProviderNoKey: (provider: string) => `服务商：${provider}（无 API 密钥）`,
    titleKeyConfigured: (provider: string) => `已为 ${provider} 配置 API 密钥`,
    // Tooltips
    sendMessage: "发送消息",
    regenerateLast: "重新生成上一条消息",
    quotaReached: (used: number, limit: number) => `每日配额已用尽（${used}/${limit}）。`,
    quotaRemaining: (remaining: number, limit: number) => `在线模式配额：本 IP 今日剩余 ${remaining}/${limit} 次对话。`,
    quotaRegenerate: (remaining: number, limit: number) => `重新生成同样会消耗配额。剩余：${remaining}/${limit}。`,
    // Header
    chat: "对话",
    modeOnline: "模式：在线",
    modeOnlineTooltip: (limit: number) => `模式：在线。每位用户每日上限：${limit} 次对话。`,
    onlineLocalHint: "如需无限制且更高效的使用，建议本地部署。",
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
    composerWaiting: "请先回复上方的表单……",
    composerPlaceholder: "随时提问 AI 蛋白质工程相关的任何问题……",
    quotaPillTitle: (limit: number) => `在线模式按 IP 每日上限：${limit}`,
    modelAria: "模型",
    gatewayRequiredSuffix: "（需要网关）",
    otherModel: "其他模型……",
    noKeySuffix: "（无密钥）",
    customSuffix: "（自定义）",
    keyOk: "密钥已配置",
    setKey: "设置密钥",
    gatewayAria: "网关",
    activeGateway: "当前网关",
    noGateway: "无网关",
    uploadFiles: "上传文件",
    fromWorkspace: "从工作区选择",
    regenerate: "重新生成",
    export: "导出",
    stop: "停止",
    send: "发送",
    runningEllipsis: "运行中……",
    searchMessages: "搜索消息",
    searchPlaceholder: "搜索消息……",
    pipelineDismiss: "关闭",
    // Key panel
    keyPanelAria: (provider: string) => `为 ${provider} 设置 API 密钥`,
    keyPanelLabelPre: "为 ",
    keyPanelLabelPost: " 设置 API 密钥：",
    keyPanelSaving: "保存中……",
    save: "保存",
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
};

// Last-resort fallback identifier used only when the model registry has not
// loaded yet AND we cannot read any model id from the active session. The real
// model list comes from GET /api/models via useModelRegistry().
const FALLBACK_MODEL_ID = "kimi-code";
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
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const SESSION_STORAGE_KEY = "vf2_active_session_id";
  const SESSION_CACHE_KEY = "vf2_session_list_cache";
  const SESSION_OWNED_KEY = "vf2_owned_session_ids";
  const COPY_HINT_MS = 1200;
  const [copiedSessionId, setCopiedSessionId] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
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
    if (s === "stopped" || s === "waiting_for_clarification" || s === "waiting_for_plan_confirmation" || s === "waiting_for_iteration" || s === "waiting_for_step_review" || s === "waiting_for_sub_report_review") {
      setRunStatus("stopped");
      return;
    }
    if (s !== "completed") {
      setRunStatus((prev) => {
        if (prev !== "running" && prev !== "stopping") setPipelineDismissed(false);
        return prev === "stopping" ? prev : "running";
      });
      return;
    }
    setRunStatus("stopped");
  }, [snapshot?.status]);

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
    setSessions(list);
    localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(list));
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

  async function refreshChatQuota() {
    try {
      const quota = await getChatQuota();
      setChatQuota(quota);
    } catch {
      setChatQuota(null);
    }
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
      setSelectedModel(modelLabelFromInternal(created.model_name));
      rememberModelFromSession(created.model_name);
      const newMeta: SessionMeta = {
        session_id: created.session_id,
        created_at: created.created_at,
        model_name: created.model_name,
        history_size: 0,
        status: "",
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
    await refreshChatQuota();
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
        await refreshCurrentSession(sid);
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

  async function refreshCurrentSession(targetId?: string) {
    const sid = targetId || sessionId;
    if (!sid) return;
    try {
      const s = await getChatSession(sid);
      setSnapshot(s);
      setSessionId(sid);
      setModelSwitchNotice("");
      sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
      setSelectedModel(modelLabelFromInternal(s.model_name));
      rememberModelFromSession(s.model_name);
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

  function handleStreamEvent({ event, data }: { event: string; data: string }) {
    if (event === "state" && data) {
      const payload = JSON.parse(data) as ChatSnapshot;
      setSnapshot(payload);
      setStreamingIdx(-1);
    } else if (event === "stream_start" && data) {
      const info = JSON.parse(data) as { role_id?: string };
      setSnapshot(prev => {
        if (!prev) return prev;
        const history = [...prev.history];
        const last = history[history.length - 1];
        // Prefer the structured `phase` marker (set by backend on placeholder
        // messages). Fall back to legacy substring matching while older backends
        // still in the wild may not emit `phase` yet.
        const isPlaceholder =
          last && last.role === "assistant" && (
            Boolean(last.phase) ||
            last.content.includes("Thinking") || last.content.includes("思考中") ||
            last.content.includes("Summarizing") || last.content.includes("正在总结") ||
            last.content.includes("汇总") || last.content.includes("撰写小报告") ||
            last.content.includes("writing sub-report") ||
            last.content.includes("撰写研究草案") ||
            last.content.includes("writing the draft report")
          );
        if (isPlaceholder) {
          history[history.length - 1] = { role: "assistant", content: "", role_id: info.role_id || last.role_id };
        } else {
          history.push({ role: "assistant", content: "", role_id: info.role_id });
        }
        setStreamingIdx(history.length - 1);
        return { ...prev, history };
      });
    } else if (event === "token" && data) {
      const token = JSON.parse(data) as { content?: string; role_id?: string };
      if (token.content) {
        setSnapshot(prev => {
          if (!prev) return prev;
          const history = [...prev.history];
          const last = history[history.length - 1];
          // kimi-code may have pushed a `kind:"thinking"` block right before
          // the answer text — never merge text deltas into the thinking
          // block. Append to the last assistant TEXT message, or push a new
          // one if the tail is a user / thinking message.
          if (last && last.role === "assistant" && last.kind !== "thinking") {
            history[history.length - 1] = { ...last, content: last.content + token.content };
            setStreamingIdx(history.length - 1);
          } else {
            history.push({ role: "assistant", content: token.content || "", role_id: token.role_id, kind: "text" });
            setStreamingIdx(history.length - 1);
          }
          return { ...prev, history };
        });
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
    }
  }

  async function sendMessage() {
    const composedText = message;
    const selectedCustomModel = customModels.find((item) => item.id === selectedModel);
    if (running) return;
    if (!message.trim() && files.length === 0 && workspaceFiles.length === 0) return;
    if (chatQuota?.enforced && (chatQuota.remaining ?? 0) <= 0) {
      const limit = chatQuota.limit ?? 10;
      setError(t.errOnlineLimit(limit));
      return;
    }
    setError("");
    setRunning(true);
    setRunStatus("running");
    setMessage("");
    abortRef.current = new AbortController();

    try {
      let activeSessionId = sessionId;
      if (!activeSessionId) {
        const created = await createChatSession();
        activeSessionId = created.session_id;
        rememberOwnedSession(activeSessionId);
        setSessionId(activeSessionId);
        sessionStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
        setSelectedModel(modelLabelFromInternal(created.model_name));
        rememberModelFromSession(created.model_name);
        setSessions((prev) => {
          const exists = prev.some((s) => s.session_id === activeSessionId);
          if (exists) return prev;
          return [{
            session_id: activeSessionId,
            created_at: created.created_at,
            model_name: created.model_name,
            history_size: 0,
            status: "",
          }, ...prev];
        });
      }

      let attachmentPaths: string[] = [];
      if (files.length > 0) {
        const uploaded = await uploadFiles(activeSessionId, files);
        attachmentPaths = uploaded.files.map((f) => f.path);
      }
      if (workspaceFiles.length > 0) {
        attachmentPaths = [...attachmentPaths, ...workspaceFiles.map((item) => item.storage_path)];
      }

      await streamSSEFromPost(
        `/api/chat/sessions/${encodeURIComponent(activeSessionId)}/messages/stream`,
        {
          text: composedText,
          model: selectedCustomModel ? selectedCustomModel.modelName : selectedModel,
          custom_model_config: selectedCustomModel
            ? {
                model_name: selectedCustomModel.modelName,
                api_key: selectedCustomModel.apiKey,
                base_url: selectedCustomModel.baseUrl,
              }
            : undefined,
          custom_model_id: selectedCustomModel ? selectedCustomModel.id : "",
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
      await refreshChatQuota();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setMessage(composedText);
      setError(err instanceof Error ? err.message : t.errStreamMsg);
      await refreshChatQuota();
      setRunStatus("stopped");
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      setMessage("");
      setFiles([]);
      setWorkspaceFiles([]);
      abortRef.current = null;
    }
  }

  async function retryLastMessage() {
    if (!sessionId || running) return;
    if (chatQuota?.enforced && (chatQuota.remaining ?? 0) <= 0) {
      const limit = chatQuota.limit ?? 10;
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
    try {
      await streamSSEFromPost(
        `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages/retry/stream`,
        {},
        handleStreamEvent,
        abortRef.current.signal,
        getChatSessionAuthHeaders(sessionId)
      );
      await fetchSessions();
      await refreshChatQuota();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errRetryMsg);
      await refreshChatQuota();
      setRunStatus("stopped");
    } finally {
      setRunning(false);
      setStreamingIdx(-1);
      abortRef.current = null;
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

  async function handleSubReportDecision(action: "continue" | "skip" | "rewrite", comment?: string) {
    if (!sessionId || running) return;
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
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setRunStatus("stopped");
        return;
      }
      setError(err instanceof Error ? err.message : t.errSubReport);
      setRunStatus("stopped");
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

  const isWaitingForInteraction = snapshot?.status === "waiting_for_clarification" || snapshot?.status === "waiting_for_plan_confirmation" || snapshot?.status === "waiting_for_iteration" || snapshot?.status === "waiting_for_step_review" || snapshot?.status === "waiting_for_sub_report_review";
  const hasReportData = Boolean(snapshot && (snapshot.tool_executions.length > 0 || snapshot.plan.length > 0));
  const quotaExhausted = Boolean(chatQuota?.enforced && (chatQuota.remaining ?? 0) <= 0);
  const sendTooltip = chatQuota?.enforced
    ? quotaExhausted
      ? t.quotaReached(chatQuota.limit ?? 10, chatQuota.limit ?? 10)
      : t.quotaRemaining(chatQuota.remaining ?? 0, chatQuota.limit ?? 10)
    : t.sendMessage;
  const regenerateTooltip = chatQuota?.enforced
    ? quotaExhausted
      ? t.quotaReached(chatQuota.limit ?? 10, chatQuota.limit ?? 10)
      : t.quotaRegenerate(chatQuota.remaining ?? 0, chatQuota.limit ?? 10)
    : t.regenerateLast;
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
  const modelOptions: ModelOption[] = [
    ...registryModels.map((m) => {
      // kimi-code manages its own provider auth via the local daemon, so our
      // own key_status / requires_adapter checks don't apply. Use the
      // backend-provided `disabled` / `disabled_reason` directly.
      if (m.engine === "kimi-code") {
        return {
          value: m.id,
          label: m.label,
          disabled: m.disabled === true,
          title: m.disabled ? (m.disabled_reason || "") : t.titleModelInfo(m.label, m.provider),
          provider: m.provider,
          requiresAdapter: false,
          hasKey: true,
        };
      }
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
    ...(isLocalMode
      ? customModels.map((m) => ({ value: m.id, label: `${m.label}${t.customSuffix}` }))
      : []),
  ];

  const selectedModelSpec = registryModels.find((m) => m.id === selectedModel);
  const selectedProvider = selectedModelSpec?.provider || "";
  const selectedProviderHasKey = selectedProvider ? keyStatus[selectedProvider] === true : true;

  // Once the registry loads, if the currently-selected id is not in the
  // registry (and not a known custom model), reset to the registry default.
  useEffect(() => {
    if (registry.loading || !registry.data) return;
    const isRegistryId = registryModels.some((m) => m.id === selectedModel);
    const isCustomId = customModels.some((m) => m.id === selectedModel);
    if (!isRegistryId && !isCustomId) {
      setSelectedModel(defaultModelId);
    }
  }, [registry.loading, registry.data, defaultModelId, selectedModel, registryModels, customModels]);

  useEffect(() => {
    if (isLocalMode) return;
    if (customModels.some((m) => m.id === selectedModel)) {
      setSelectedModel(defaultModelId);
    }
  }, [isLocalMode, selectedModel, customModels, defaultModelId]);

  function openKeyPanelForProvider(provider: string) {
    setKeyPanelProvider(provider);
    setKeyPanelValue("");
  }

  function handleModelChange(next: string) {
    if (!isLocalMode && next === OTHER_MODEL_OPTION) return;
    if (next === OTHER_MODEL_OPTION) {
      setShowCustomModelModal(true);
      return;
    }
    const prev = selectedModel;
    setSelectedModel(next);
    if (prev !== next && (snapshot?.history?.length || 0) > 0) {
      setModelSwitchNotice(t.noticeModelSwitched);
    }
    // If the newly selected registry model is missing a key, surface the
    // inline key input so the user can configure it without leaving the page.
    const spec = registryModels.find((m) => m.id === next);
    if (spec && keyStatus[spec.provider] !== true) {
      openKeyPanelForProvider(spec.provider);
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
        setSelectedModel(defaultModelId);
        setModelSwitchNotice(t.noticeCustomRemoved);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errRemoveCustomModel);
    }
  }

  return (
    <div className="chat-page">
      {chatQuota?.enforced && (
        <header className="chat-header chat-header-slim">
          <div>
            <div className="chat-header-title-row">
              <span
                className="chat-mode-online-pill"
                title={t.modeOnlineTooltip(chatQuota.limit ?? 10)}
              >
                {t.modeOnline}
              </span>
            </div>
            <p className="chat-online-local-hint">{t.onlineLocalHint}</p>
          </div>
        </header>
      )}

      <section className={`chat-grid${sessionsCollapsed ? " left-collapsed" : ""}${logsCollapsed ? " right-collapsed" : ""}`}>
        <aside
          className={`chat-panel left${sessionsCollapsed ? " collapsed" : ""}`}
          onClick={sessionsCollapsed ? () => setSessionsCollapsed(false) : undefined}
        >
          <div className="session-panel-head" onClick={() => setSessionsCollapsed(!sessionsCollapsed)}>
            <h3>{t.sessions} <span className="panel-toggle-icon">{sessionsCollapsed ? "›" : "‹"}</span></h3>
          </div>
          <button
            type="button"
            className="session-new-btn"
            onClick={() => void createAndActivateSession()}
            disabled={running}
          >
            {t.newSession}
          </button>
          <div className="session-list" style={sessionsCollapsed ? { display: "none" } : undefined}>
            {sessions.map((s) => (
              <div
                key={s.session_id}
                className={s.session_id === sessionId ? "session-item active" : "session-item"}
              >
                <button
                  className="session-select-btn"
                  onClick={() => void refreshCurrentSession(s.session_id)}
                  disabled={running && s.session_id === sessionId}
                  title={s.session_id}
                >
                  <span className="session-id-label">{s.session_id.slice(0, 8)}</span>
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
            ))}
            {sessions.length === 0 && <div className="session-empty">{t.noSessions}</div>}
          </div>
          {sessionId && !sessionsCollapsed && (
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
        </aside>

        <section className="chat-panel center">
          <div className="timeline-wrap" ref={timelineRef}>
            <div className="timeline-sticky-header">
              {(snapshot?.history?.length ?? 0) > 0 && (
                <div className="timeline-toolbar">
                  <button
                    className={`timeline-search-toggle${searchOpen ? " active" : ""}`}
                    onClick={() => { setSearchOpen(!searchOpen); if (searchOpen) setSearchQuery(""); }}
                    title={t.searchMessages}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
                    </svg>
                  </button>
                  {searchOpen && (
                    <input
                      className="timeline-search-input"
                      type="text"
                      placeholder={t.searchPlaceholder}
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      autoFocus
                    />
                  )}
                </div>
              )}
            </div>
            <ChatTimeline
              items={snapshot?.history || []}
              streamingIndex={streamingIdx}
              onSuggestedPrompt={(text) => setMessage(text)}
              sessionId={sessionId}
              searchQuery={searchQuery}
              toolExecutions={snapshot?.tool_executions || []}
              securityEvents={snapshot?.security_events || []}
              onRetry={() => void retryLastMessage()}
              retryDisabled={running || quotaExhausted}
              onQuoteReply={(text) => {
                const lines = text.split("\n").slice(0, 3);
                const quoted = lines.map((l) => `> ${l}`).join("\n");
                setMessage((prev) => `${quoted}\n\n${prev}`);
              }}
            />
            {snapshot?.status === "waiting_for_clarification" &&
              snapshot.clarification_questions?.length > 0 && (
                <div className="chat-msg assistant with-avatar">
                  <img
                    className="chat-msg-avatar"
                    src="/img/agent_role/principal_investigator.png"
                    alt={t.rolePI}
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src =
                        "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                    }}
                  />
                  <div className="chat-msg-content">
                    <div className="chat-msg-role">{t.rolePI}</div>
                    <ClarificationForm
                      questions={snapshot.clarification_questions}
                      onSubmit={submitClarification}
                      disabled={running}
                    />
                  </div>
                </div>
              )}
            {snapshot?.status === "waiting_for_plan_confirmation" &&
              snapshot.plan?.length > 0 && (
                <div className="chat-msg assistant with-avatar">
                  <img
                    className="chat-msg-avatar"
                    src="/img/agent_role/computational_biologist.png"
                    alt={t.roleCompBio}
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src =
                        "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                    }}
                  />
                  <div className="chat-msg-content">
                    <div className="chat-msg-role">{t.roleCompBio}</div>
                    <PlanEditor
                      plan={snapshot.plan}
                      onConfirm={confirmPlan}
                      disabled={running}
                    />
                  </div>
                </div>
              )}
            {snapshot?.status === "waiting_for_sub_report_review" && (
              <div className="chat-msg assistant with-avatar">
                <img
                  className="chat-msg-avatar"
                  src="/img/agent_role/principal_investigator.png"
                  alt={t.rolePI}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src =
                      "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                  }}
                />
                <div className="chat-msg-content">
                  <div className="chat-msg-role">{t.rolePI}</div>
                  <SubReportCheckpoint
                    onDecide={handleSubReportDecision}
                    disabled={running}
                  />
                </div>
              </div>
            )}
            {snapshot?.status === "waiting_for_step_review" && (
              <div className="chat-msg assistant with-avatar">
                <img
                  className="chat-msg-avatar"
                  src="/img/agent_role/machine_learning_specialist.png"
                  alt={t.roleMLSpec}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src =
                      "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                  }}
                />
                <div className="chat-msg-content">
                  <div className="chat-msg-role">{t.roleMLSpec}</div>
                  <StepCheckpoint
                    onDecide={handleStepDecision}
                    disabled={running}
                  />
                </div>
              </div>
            )}
            {snapshot?.status === "waiting_for_iteration" && (
              <div className="chat-msg assistant with-avatar">
                <img
                  className="chat-msg-avatar"
                  src="/img/agent_role/principal_investigator.png"
                  alt={t.rolePI}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src =
                      "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                  }}
                />
                <div className="chat-msg-content">
                  <div className="chat-msg-role">{t.rolePI}</div>
                  <IterationDecision
                    onDecide={handleIterationDecision}
                    disabled={running}
                  />
                </div>
              </div>
            )}
          </div>
          <div className="composer">
            <div className="composer-textarea-wrap">
              <textarea
                rows={4}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={onComposerKeyDown}
                placeholder={isWaitingForInteraction ? t.composerWaiting : t.composerPlaceholder}
                disabled={running || quotaExhausted || isWaitingForInteraction}
              />
              {chatQuota?.enforced && (
                <span
                  className={`chat-quota-pill${quotaExhausted ? " exhausted" : ""}`}
                  title={t.quotaPillTitle(chatQuota.limit ?? 10)}
                >
                  {quotaExhausted
                    ? `${chatQuota.used}/${chatQuota.limit ?? 10}`
                    : `${chatQuota.remaining ?? 0}/${chatQuota.limit ?? 10}`}
                </span>
              )}
            </div>
            <div className="composer-row">
              <select
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
                {isLocalMode && <option value={OTHER_MODEL_OPTION}>{t.otherModel}</option>}
              </select>
              {selectedModelSpec && (
                <span
                  className="model-key-status"
                  title={
                    selectedProviderHasKey
                      ? t.titleKeyConfigured(selectedProvider)
                      : t.titleMissingKey(selectedProvider)
                  }
                  onClick={() => {
                    if (!selectedProviderHasKey) openKeyPanelForProvider(selectedProvider);
                  }}
                  style={{
                    cursor: selectedProviderHasKey ? "default" : "pointer",
                    color: selectedProviderHasKey ? "#2e7d32" : "#b26a00",
                    fontSize: "12px",
                    marginLeft: "4px",
                    userSelect: "none",
                  }}
                  role={selectedProviderHasKey ? undefined : "button"}
                >
                  {selectedProviderHasKey ? t.keyOk : t.setKey}
                </span>
              )}
              {(registry.data?.gateways?.length ?? 0) > 0 && (
                <select
                  value={registry.data?.active_gateway || ""}
                  onChange={(e) => void handleGatewayChange(e.target.value)}
                  aria-label={t.gatewayAria}
                  title={t.activeGateway}
                  style={{ marginLeft: "4px" }}
                >
                  <option value="">{t.noGateway}</option>
                  {(registry.data?.gateways || []).map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.label}
                    </option>
                  ))}
                </select>
              )}
              <div className="file-source-inline">
                <label className={`file-upload-icon-btn${running || quotaExhausted ? " disabled" : ""}`} title={t.uploadFiles}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                  </svg>
                  <input
                    type="file"
                    multiple
                    onChange={(e) => setFiles(Array.from(e.target.files || []))}
                    disabled={running || quotaExhausted}
                    className="file-upload-hidden"
                  />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running || quotaExhausted}
                  allowMultiple
                  buttonLabel={t.fromWorkspace}
                  onPick={(picked) => setWorkspaceFiles(picked)}
                />
              </div>
              <button className="btn-secondary" onClick={() => void retryLastMessage()} disabled={running || quotaExhausted} title={regenerateTooltip}>
                {t.regenerate}
              </button>
              <button className="btn-secondary" onClick={() => void exportCurrentSession()} disabled={running || !sessionId}>
                {t.export}
              </button>
              <button
                className={`btn-secondary btn-stop${running ? " btn-stop-active" : ""}`}
                onClick={abortRun}
                disabled={!running}
                title="Stop the current run"
              >
                <span className="btn-stop-square" aria-hidden="true" /> {t.stop}
              </button>
              <button className="btn-primary" onClick={() => void sendMessage()} disabled={running || quotaExhausted} title={sendTooltip}>
                {running ? t.runningEllipsis : t.send}
              </button>
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
                <span style={{ fontSize: "13px" }}>
                  {t.keyPanelLabelPre}<strong>{keyPanelProvider}</strong>{t.keyPanelLabelPost}
                </span>
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
          )}
          {/* Always visible — pulled out of the collapsible term-body so the
              session's working directory stays in view even when the user
              hides the terminal log. */}
          <SessionFilesPanel
            sessionId={sessionId}
            authHeaders={sessionId ? getChatSessionAuthHeaders(sessionId) : undefined}
            liveRefresh={running}
            pollMs={5000}
          />
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
