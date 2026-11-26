import { conversationHistory } from "./chat/conversationHistory";
import { ConversationCheckpoints } from "./chat/ConversationCheckpoints";
import { checkpointForSession, checkpointPlaceholder } from "./chat/conversationCheckpoint";
import { chatMessages, friendlyErrorHint } from "./chat/chatMessages";
import { useChatStream } from "../lib/useChatStream";
import { sanitizeChatSnapshot } from "../lib/chatStreamState";
import { submissionHistoryRow } from "../lib/optimisticHistory";
import { NewChatStart } from "../components/NewChatStart";
import { ChatModeIcon } from "../components/ChatModeIcon";
import { UnifiedChatHistory } from "../components/UnifiedChatHistory";
import { claimNewChatDraft, newChatLocation, CHAT_MODE_LABELS, type ResearchHistoryRow } from "../lib/unifiedChatHistory";
import type { UserResponse } from "../lib/interactions";
import { manualQuestion } from "../lib/manualCatalog";
import { moveResearchDraft } from "../lib/researchContinuity";
import { chatHistoryWithPending, hasAcceptedTurn, type PendingChatTurn } from "../lib/pendingChatTurn";
import { useFileTransfer } from "../lib/useFileTransfer";
import { mergeFiles } from "../lib/fileTransfer";
import { validateUploads } from "../lib/uploadLimits";
import { QuotaNotice } from "../components/QuotaNotice";
import { openQuota, QUOTA_CHANGED, QUOTA_STORAGE_KEY, quotaRetryLabel, isDailyQuotaMessage } from "../lib/onlineQuota";
import { SidebarContent, useSidebar } from "../components/AppSidebar";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { chatSessionLocation, chatRouteKey } from "../lib/chatSessionRoute";
import { ResearcherPanel, type ResearcherPanelHandle } from "../components/ResearcherPanel";
import "../researcher.css";
import "../chat-layout.css";
import { ChatNotice, useDismissibleError } from "../components/ChatNotice";
import { AgentExecutionPicker } from "../components/AgentExecutionPicker";
import { ContextUsageRing } from "../components/ContextUsageRing";
import { isRunActive } from "../lib/runProgress";
import { activeSessionStorageKey, sessionsForMode, sessionCandidates, abbreviateSessionTitle } from "../lib/chatSessions";
import { useChatSessionHistory } from "../lib/useChatSessionHistory";
import { SessionPreviewCache } from "../lib/sessionPreviewCache";
import { PageLoading } from "../components/PageLoading";
import { isFreeResearchModel } from "../lib/researchUi";
import { ScienceGoalPanel } from "../components/ScienceGoalPanel";
import { ChatTimeline } from "../components/ChatTimeline";
import { SessionFilesPanel } from "../components/SessionFilesPanel";
import type { ApprovalDecision } from "../components/ApprovalCard";
import { isResearchNoiseTool, type ToolExecution } from "../components/ToolExecutionCard";
import {
  cancelChatSession,
  createChatSession,
  exportChatSessionBundle,
  deleteChatSession,
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
  type AgentExecutionSelection,
  type ChatQuota,
  type ChatSnapshot,
  type ClarificationAnswer,
  type PlanStep,
  uploadFiles
} from "../lib/api";
import { streamSSEFromPost, streamSSEFromGet } from "../lib/sse";
import { runChatStreamAction, runChatUpdateAction, cancelChatStreamAction } from "../lib/chatActions";
import { AttachedFiles, ComposerFilePicker, type AttachmentPanel } from "./chat/ComposerAttachments";
import { type WorkspaceFile } from "../lib/workspaceApi";
import {
  useModelRegistry,
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

// Last-resort fallback identifier used only when the model registry has not
// loaded yet AND we cannot read any model id from the active session. The real
// model list comes from GET /api/models via useModelRegistry().
const FALLBACK_MODEL_ID = SCIENCE_AGENT_MODEL_ID;
type RunStatus = "running" | "stopping" | "stopped";

type OpenAIStyleModel = {
  id: string;
  label: string;
  modelName: string;
  apiKey: string;
  baseUrl: string;
};

type ChatPageProps = {
  workspaceEnabled?: boolean;
};

const CUSTOM_MODELS_STORAGE_KEY = "vf2_custom_openai_style_models";

export function ChatPage({ workspaceEnabled = false }: ChatPageProps) {
  const { lang, t: translate } = useLang();
  const t = translate(chatMessages);
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
  const [workspaceParams] = useSearchParams();
  const researcherMode = workspaceParams.get('mode') === 'researcher';
  const visibleModeRef = useRef(workspaceParams.get('mode'));
  visibleModeRef.current = workspaceParams.get('mode');
  const [researcherRunning, setResearcherRunning] = useState(false);
  const [chatMode, setChatMode] = useState<ChatMode>(() => workspaceParams.get("mode") === "scientist" ? "science_expert" : "science_agent");
  const activeModeRef = useRef(chatMode);
  const { sessions, setSessions, historyActivity, onHistoryActivity, activeSessionTitle, fetchSessions } =
    useChatSessionHistory(sessionId, snapshot, chatMode);
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
  const [customModels, setCustomModels] = useState<OpenAIStyleModel[]>(() => {
    try {
      const parsed = JSON.parse(localStorage.getItem(CUSTOM_MODELS_STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed.filter(item => item && item.id && item.modelName && item.baseUrl) : [];
    } catch { return []; }
  });
  const [modelSwitchNotice, setModelSwitchNotice] = useState("");
  const [chatQuota, setChatQuota] = useState<ChatQuota | null>(null);
  const [startingTurn, setStartingTurn] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const viewMounted = useRef(true);
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
  const [filePicker, setFilePicker] = useState<AttachmentPanel>(null);
  const LEGACY_SESSION_STORAGE_KEY = "vf2_active_session_id";
  const modeChoiceRevision = useRef(0);
  const [agentCatalogReady, setAgentCatalogReady] = useState(false);
  const { closeOnNarrow } = useSidebar();
  const [pipelineDismissed, setPipelineDismissed] = useState(false);
  useEffect(() => {
    try {
      localStorage.setItem(CUSTOM_MODELS_STORAGE_KEY, JSON.stringify(customModels));
    } catch { /* The current tab can still use its in-memory settings. */ }
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
      void Promise.all([refreshChatQuota(), fetchSessions()]).catch(() => {});
      return;
    }
    if (routeMode === "researcher") {
      void Promise.all([refreshChatQuota(), fetchSessions()]).catch(() => {});
      return;
    }
    if (helpQuestion) {
      if (activeModeRef.current !== 'science_agent') handleChatModeChange('science_agent', undefined, false);
      createAndActivateSession(false);
      setMessage(helpQuestion);
      void Promise.all([refreshChatQuota(), fetchSessions()]).catch(() => {});
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
    viewMounted.current = true;
    return () => {
      viewMounted.current = false;
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
        abortRef.current?.abort();
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
    const onStorage = (event: StorageEvent) => {
      if (event.storageArea === window.localStorage && (event.key === null || event.key === QUOTA_STORAGE_KEY)) changed();
    };
    window.addEventListener("storage", onStorage);
    return () => { window.removeEventListener(QUOTA_CHANGED, changed); window.removeEventListener("focus", changed); window.removeEventListener("storage", onStorage); };
  }, []);

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
    const ownsSelection = currentSessionView(targetId);
    if (running && deletingActive) return false;
    setError("");
    try {
      await deleteChatSession(targetId);
      sessionPreviews.current.forget(targetId);
      setSessions(previous => previous.filter(row => row.session_id !== targetId));
      const list = await fetchSessions().catch(() => sessions);
      // A delayed delete may update history, but cannot navigate a different view.
      if (!deletingActive || !ownsSelection()) return true;
      const next = sessionsForMode(list, activeModeRef.current).find(row => row.session_id !== targetId);
      if (next) {
        const opening = refreshCurrentSession(next.session_id);
        const ownsNext = currentSessionView(next.session_id);
        try { await opening; }
        catch (err) {
          if (ownsNext()) {
            createAndActivateSession(true, false);
            setError(err instanceof Error ? err.message : t.errHintSession);
          }
        }
      } else createAndActivateSession(true, false);
      return true;
    } catch (err) {
      if (!deletingActive || ownsSelection()) setError(err instanceof Error ? err.message : t.errDeleteSession);
      return false;
    }
  }

  async function bootstrapSession() {
    let bootstrapRevision = sessionLoad.current.revision;
    setError("");
    const quotaRequest = refreshChatQuota();
    const sessionsRequest = fetchSessions().catch(() => sessions);
    const [, list] = await Promise.all([quotaRequest, sessionsRequest]);
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
    const isCurrentRequest = () => viewMounted.current && activeModeRef.current === requestedMode && sessionLoad.current.revision === requestedSessionRevision && sessionLoad.current.target === sid;
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
      setFilePicker(previous => previous === "sources" ? null : previous); setError("");
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

  function currentSessionView(sid = sessionId) {
    const revision = sessionLoad.current.revision;
    const mode = activeModeRef.current;
    return () => viewMounted.current && sessionLoad.current.target === sid && sessionLoad.current.revision === revision &&
      activeModeRef.current === mode && visibleModeRef.current !== "researcher";
  }

  async function submitSessionAction(url: string, body: unknown, errorMessage: string, refreshOnFailure = false) {
    const sid = sessionId;
    return runChatStreamAction({
      stream: abortRef,
      current: currentSessionView(sid),
      request: (signal, onEvent) => streamSSEFromPost(url, body, onEvent, signal, getChatSessionAuthHeaders(sid)),
      onStart: () => {
        runningRef.current = true;
        setError(""); setRunning(true); setRunStatus("running"); setStartingTurn(true);
      },
      onEvent: event => { setStartingTurn(false); handleStreamEvent(event); },
      onSuccess: () => { void fetchSessions().catch(() => {}); },
      onError: async error => {
        setError(error instanceof Error ? error.message : errorMessage);
        setRunStatus("stopped");
        if (refreshOnFailure) await refreshCurrentSession(sid, { syncComposer: false }).catch(() => {});
      },
      onFinish: () => {
        runningRef.current = false;
        setRunning(false); setStartingTurn(false); stopStream();
      },
    });
  }

  async function retryLastMessage() {
    if (!sessionId || agentWorking) return;
    if (!snapshot?.history?.some((h) => h.role === "user")) {
      setError(t.errNoUserMsg);
      return;
    }
    await submitSessionAction(`/api/chat/sessions/${encodeURIComponent(sessionId)}/messages/retry/stream`, {}, t.errRetryMsg);
    await refreshChatQuota();
  }

  async function abortRun() {
    if (!sessionId) return;
    const sid = sessionId;
    const current = currentSessionView(sid);
    await cancelChatStreamAction({
      stream: abortRef, current,
      request: () => cancelChatSession(sid),
      onStart: () => setRunStatus("stopping"),
      onSuccess: () => {
        setRunning(false);
        // Refresh only the cancelled view, without interrupting a later turn.
        setTimeout(() => {
          if (current() && !abortRef.current) void refreshCurrentSession(sid, { syncComposer: false }).catch(() => {});
        }, 600);
      },
      onError: error => {
        setRunStatus("running");
        setError(error instanceof Error ? error.message : t.errStopRun);
      },
    });
  }

  async function submitClarification(answers: ClarificationAnswer[]) {
    if (!sessionId || agentWorking) return;
    await submitSessionAction(getClarificationRespondUrl(sessionId), { answers }, t.errClarification);
  }

  async function submitInteraction(answer: UserResponse) {
    if (!sessionId || agentWorking) return;
    await submitSessionAction(`/api/chat/sessions/${encodeURIComponent(sessionId)}/interactions/respond/stream`, answer, t.errAskUser, true);
  }

  async function submitAskUser(answers: ClarificationAnswer[]) {
    if (!sessionId || agentWorking) return;
    setModelSwitchNotice(t.noticeAskUserSubmitted);
    await submitSessionAction(getAskUserRespondUrl(sessionId), { answers }, t.errAskUser);
  }

  async function submitApproval(decision: ApprovalDecision) {
    if (!sessionId || agentWorking) return;
    setModelSwitchNotice(decision.decision === "approved" ? t.noticeApprovalApproved : t.noticeApprovalRejected);
    await submitSessionAction(getApprovalDecideUrl(sessionId), {
      decision: decision.decision,
      selected_label: decision.selected_label || "",
      feedback: decision.feedback || "",
    }, t.errApproval);
  }

  async function confirmPlan(plan: PlanStep[], autoExecute: boolean) {
    if (!sessionId || agentWorking) return;
    await submitSessionAction(getPlanConfirmUrl(sessionId), { plan, auto_execute: autoExecute }, t.errConfirmPlan);
  }

  async function handleIterationDecision(action: "satisfied" | "modify_plan" | "continue") {
    if (!sessionId || agentWorking) return;
    const sid = sessionId;
    const current = currentSessionView(sid);
    setError("");
    await runChatUpdateAction({
      current: () => current() && !abortRef.current,
      request: () => iterationDecide(sid, action),
      onSuccess: () => refreshCurrentSession(sid, { syncComposer: false }),
      onError: error => setError(error instanceof Error ? error.message : t.errIteration),
    });
  }

  async function handleStepDecision(action: "continue" | "abort") {
    if (!sessionId || agentWorking) return;
    await submitSessionAction(getStepDecideUrl(sessionId), { action }, t.errStepDecision);
  }

  async function handleSubReportDecision(action: "continue" | "skip" | "rewrite", comment?: string): Promise<boolean> {
    if (!sessionId || running) return false;
    return submitSessionAction(getSubReportDecideUrl(sessionId), { action, comment: comment || "" }, t.errSubReport);
  }

  function onComposerKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter") return;
    // Keep Shift+Enter for newline; ignore IME composition Enter.
    if (e.shiftKey || e.nativeEvent.isComposing) return;
    e.preventDefault();
    void sendMessage();
  }

  const checkpoint = checkpointForSession(snapshot, chatMode);
  const isWaitingForInteraction = Boolean(snapshot?.pending_interaction || checkpoint);
  useFileTransfer(composerElement, {
    disabled: running || goalRunning || addingFiles || isWaitingForInteraction,
    label: lang === "zh" ? "松开以添加文件" : "Drop files to attach",
    unavailable: lang === "zh" ? "请等待当前操作完成后添加文件。" : "Wait for the current operation before attaching files.",
    unreadable: lang === "zh" ? "请选择文件，暂不支持文件夹。" : "Choose files; folders are not supported.",
    onFiles: addComposerFiles, onError: setError,
  }, !researcherMode && chatMode === "science_agent");
  const composerWaitingText = checkpointPlaceholder(checkpoint, Boolean(snapshot?.pending_interaction), t);
  const quotaExhausted = Boolean(chatQuota?.enforced && (chatQuota.remaining ?? 0) <= 0);
  const sendTooltip = chatQuota?.enforced
    ? quotaExhausted
      ? quotaRetryLabel(chatQuota, lang)
      : t.quotaRemaining(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)
    : t.sendMessage;
  const isLocalMode = chatQuota?.mode === "local";
  const registryModels: ModelSpec[] = registry.data?.models || [];
  const keyStatus: Record<string, boolean> = registry.data?.key_status || {};
  const expertRegistryModels = useMemo(
    () => registryModels.filter((m) => isGraphEngineModel(m)),
    [registryModels]
  );
  const modelOptions = expertRegistryModels.map((m) => {
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
    };
  });

  const selectedModelSpec = registryModels.find((m) => m.id === selectedModel);
  const selectedProvider = selectedModelSpec?.provider || "";
  const selectedProviderHasKey = selectedProvider ? keyStatus[selectedProvider] === true : true;
  // Keep the local fallback picker available before a harness selection exists.
  const showLocalModelControls = isLocalMode && chatMode === "science_agent" && !agentExecution;

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

  function handleChatModeChange(next: ChatMode, targetId?: string, updateUrl = true) {
    if (startingTurn || (next === activeModeRef.current && !researcherMode)) return;
    detachStream();
    modeChoiceRevision.current += 1;
    activeModeRef.current = next;
    setChatMode(next);
    persistChatMode(next);
    setFilePicker(null);
    setModelSwitchNotice("");
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
    if (!isLocalMode) return;
    if (selectedModel !== next && snapshot?.history?.length) setModelSwitchNotice(t.noticeModelSwitched);
    setSelectedModel(next);
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

  const isNewChat = !routeSessionId && (researcherMode || !sessionId) && !startingTurn && !pendingTurn;
  const history = conversationHistory({
    sessions, reports: researchReports, activity: historyActivity,
    activeMode: researcherMode ? "researcher" : chatMode,
    activeId: researcherMode ? routeSessionId : sessionId, activeTitle: activeSessionTitle,
    agentRunId: agentWorking ? sessionId : "",
    scientistRunId: isRunActive(goalStatuses[scientistSessionId]) ? scientistSessionId : "",
    online: chatQuota?.mode === "online", untitled: t.newChat,
  });
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

  const quotaPill = chatQuota?.enforced && (
    <span
      className={`chat-quota-pill${quotaExhausted ? " exhausted" : ""}`}
      role="button" tabIndex={0} onClick={() => openQuota()} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openQuota(); } }}
      title={t.quotaPillTitle(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)}
    >
      {quotaExhausted
        ? t.quotaPillExhausted(chatQuota.used ?? 0, chatQuota.limit ?? 3)
        : t.quotaPillLabel(chatQuota.remaining ?? 0, chatQuota.limit ?? 3)}
    </span>
  );

  return (
    <div className="chat-page" ref={chatLayoutRef} data-new-chat={isNewChat} data-research-mode={researcherMode ? "researcher" : chatMode}>
      <section className="chat-grid">
        <SidebarContent>
          <UnifiedChatHistory rows={history.rows} activeKey={history.activeKey}
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
            {!researcherMode && snapshot && (
              <ConversationCheckpoints
                checkpoint={checkpoint} snapshot={snapshot} sessionId={sessionId} agentView={chatMode === "science_agent"}
                agentWorking={agentWorking} streaming={streamingIdx >= 0} running={running} t={t}
                submitInteraction={submitInteraction} submitAskUser={submitAskUser} submitApproval={submitApproval}
                submitClarification={submitClarification} confirmPlan={confirmPlan} handleSubReportDecision={handleSubReportDecision}
                handleStepDecision={handleStepDecision} handleIterationDecision={handleIterationDecision}
              />
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
                        </select>
                        <span className="composer-select-chevron" aria-hidden="true">
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="6 9 12 15 18 9" />
                          </svg>
                        </span>
                      </span>
                    </label>
                  </>
                )}
        </div>
              </div>
              <div className="composer-actions">
                <div className="scientist-composer-actions" ref={setScienceActions} />
                {!researcherMode && chatMode === "science_agent" && <>
                <ContextUsageRing context={snapshot?.agent_context} nativeContext={snapshot?.kimi_context} lang={lang} />
                <ComposerFilePicker panel={filePicker} onChange={setFilePicker} running={running} t={t}
                  workspaceEnabled={workspaceEnabled} addComposerFiles={addComposerFiles}
                  onWorkspaceFiles={picked => setWorkspaceFiles(previous => [...previous, ...picked.filter(file => !previous.some(old => old.id === file.id))])} />
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
            {!researcherMode && chatMode === "science_agent" && (files.length > 0 || workspaceFiles.length > 0) && (
              <AttachedFiles files={files} workspaceFiles={workspaceFiles} running={running} lang={lang}
                workspaceSuffix={t.workspaceSuffix} onRemoveFile={index => setFiles(previous => previous.filter((_, i) => i !== index))}
                onRemoveWorkspaceFile={id => setWorkspaceFiles(previous => previous.filter(file => file.id !== id))} />
            )}
          </div>
        </section>

        {researcherMode && <div className="chat-file-actions">{quotaPill}</div>}
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
                </button>} quota={quotaPill}/>}
      </section>
    </div>
  );
}
