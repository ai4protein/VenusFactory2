import { useEffect, useMemo, useRef, useState } from "react";
import { ChatTimeline } from "../components/ChatTimeline";
import { PipelineProgress } from "../components/PipelineProgress";
import { ClarificationForm } from "../components/ClarificationForm";
import { IterationDecision } from "../components/IterationDecision";
import { PlanEditor } from "../components/PlanEditor";
import { StepCheckpoint } from "../components/StepCheckpoint";
import { SubReportCheckpoint } from "../components/SubReportCheckpoint";
import {
  cancelChatSession,
  createChatSession,
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

type SessionMeta = {
  session_id: string;
  created_at: string;
  model_name: string;
  history_size: number;
  status: string;
};

const MODELS = ["Gemini-2.5-Pro", "ChatGPT-4o", "Claude-3.7", "DeepSeek-R1"];
type RunStatus = "running" | "stopping" | "stopped";

function friendlyErrorHint(msg: string): string {
  const m = msg.toLowerCase();
  if (m.includes("quota") || m.includes("limit reached"))
    return "You've reached the daily usage limit for online mode. Try again tomorrow, or deploy locally for unlimited access.";
  if (m.includes("timeout") || m.includes("timed out"))
    return "The request took too long. This can happen with complex tasks or heavy server load. Please try again.";
  if (m.includes("network") || m.includes("fetch") || m.includes("failed to fetch"))
    return "A network issue occurred. Please check your connection and try again.";
  if (m.includes("401") || m.includes("unauthorized") || m.includes("auth"))
    return "Authentication failed. Your session may have expired — try refreshing the page.";
  if (m.includes("403") || m.includes("forbidden") || m.includes("access denied"))
    return "Access was denied. You may not have permission for this action.";
  if (m.includes("429") || m.includes("rate") || m.includes("too many"))
    return "Too many requests in a short time. Please wait a moment and try again.";
  if (m.includes("500") || m.includes("internal server"))
    return "The server encountered an internal error. This is usually temporary — please retry shortly.";
  if (m.includes("model") || m.includes("llm") || m.includes("api key"))
    return "There was an issue with the AI model service. The model may be temporarily unavailable.";
  if (m.includes("session") || m.includes("not found"))
    return "The session could not be found. It may have expired — try creating a new session.";
  return "Something went wrong. This is usually temporary — please try again or start a new session.";
}

function ErrorAlert({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div className="error-box">
      <div className="error-box-header">
        <span className="error-box-icon">!</span>
        <span className="error-box-hint">{friendlyErrorHint(message)}</span>
        <button className="error-box-dismiss" onClick={onDismiss} title="Dismiss">&times;</button>
      </div>
      <details className="error-box-details">
        <summary>Details</summary>
        <pre className="error-box-raw">{message}</pre>
      </details>
    </div>
  );
}

type ChatPageProps = {
  workspaceEnabled?: boolean;
};

export function ChatPage({ workspaceEnabled = false }: ChatPageProps) {
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
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
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
      sessionStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
      setSelectedModel(modelLabelFromInternal(created.model_name));
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
      await fetchSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session.");
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
      setError(err instanceof Error ? err.message : "Failed to delete session.");
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

  function modelLabelFromInternal(modelName: string) {
    if (modelName === "gpt-4o") return "ChatGPT-4o";
    if (modelName === "claude-3-7-sonnet-20250219") return "Claude-3.7";
    if (modelName === "deepseek-r1-0528") return "DeepSeek-R1";
    return "Gemini-2.5-Pro";
  }

  async function refreshCurrentSession(targetId?: string) {
    const sid = targetId || sessionId;
    if (!sid) return;
    try {
      const s = await getChatSession(sid);
      setSnapshot(s);
      setSessionId(sid);
      sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
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
        if (last && last.role === "assistant" && (
          last.content.includes("Thinking") || last.content.includes("思考中") ||
          last.content.includes("Summarizing") || last.content.includes("正在总结") ||
          last.content.includes("汇总") || last.content.includes("撰写小报告") ||
          last.content.includes("writing sub-report") ||
          last.content.includes("撰写研究草案") ||
          last.content.includes("writing the draft report")
        )) {
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
          if (last && last.role === "assistant") {
            history[history.length - 1] = { ...last, content: last.content + token.content };
            setStreamingIdx(history.length - 1);
          } else {
            history.push({ role: "assistant", content: token.content || "", role_id: token.role_id });
            setStreamingIdx(history.length - 1);
          }
          return { ...prev, history };
        });
      }
    }
  }

  async function sendMessage() {
    const composedText = message;
    if (running) return;
    if (!message.trim() && files.length === 0 && workspaceFiles.length === 0) return;
    if (chatQuota?.enforced && (chatQuota.remaining ?? 0) <= 0) {
      const limit = chatQuota.limit ?? 10;
      setError(`Online mode limit reached: up to ${limit} chats per user per day.`);
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
          model: selectedModel,
          attachment_paths: attachmentPaths
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
      setError(err instanceof Error ? err.message : "Failed to stream message.");
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
      setError(`Online mode limit reached: up to ${limit} chats per user per day.`);
      return;
    }
    if (!snapshot?.history?.some((h) => h.role === "user")) {
      setError("No previous user message in current session.");
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
      setError(err instanceof Error ? err.message : "Failed to retry message.");
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
      setError(err instanceof Error ? err.message : "Failed to submit clarification.");
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
      setError(err instanceof Error ? err.message : "Failed to confirm plan.");
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
      setError(err instanceof Error ? err.message : "Failed to process iteration decision.");
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
      setError(err instanceof Error ? err.message : "Failed to process step decision.");
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
      setError(err instanceof Error ? err.message : "Failed to process sub-report decision.");
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
      setError("Failed to copy session id.");
    }
  }

  async function handleDownloadReport() {
    if (!sessionId) return;
    setError("");
    try {
      await downloadExperimentReport(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to download report.");
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
      ? `Daily quota reached (${chatQuota.limit ?? 10}/${chatQuota.limit ?? 10}).`
      : `Online mode quota: ${chatQuota.remaining ?? 0}/${chatQuota.limit ?? 10} chats remaining for this IP today.`
    : "Send message";
  const regenerateTooltip = chatQuota?.enforced
    ? quotaExhausted
      ? `Daily quota reached (${chatQuota.limit ?? 10}/${chatQuota.limit ?? 10}).`
      : `Regenerate also consumes quota. Remaining: ${chatQuota.remaining ?? 0}/${chatQuota.limit ?? 10}.`
    : "Regenerate last message";

  return (
    <div className="chat-page">
      <header className="chat-header">
        <div>
          <div className="chat-header-title-row">
            <h2>Chat</h2>
            {chatQuota?.enforced && (
              <span
                className="chat-mode-online-pill"
                title={`Mode: Online. Per-user daily limit: ${chatQuota.limit ?? 10} chats.`}
              >
                Mode: Online
              </span>
            )}
          </div>
          {chatQuota?.enforced && (
            <p className="chat-online-local-hint">
              For unlimited and more efficient usage, local deployment is recommended.
            </p>
          )}
        </div>
        <div className="chat-header-actions">
          <div className={`run-status-bar ${runStatus}`}>
            <span className="run-status-dot" />
            <span className="run-status-text">
              {runStatus === "running" && "Running"}
              {runStatus === "stopping" && "Stopping"}
              {runStatus === "stopped" && "Stopped"}
            </span>
          </div>
          <button onClick={() => void fetchSessions()}>Refresh</button>
          {hasReportData && (
            <button
              className="report-download-btn"
              onClick={() => void handleDownloadReport()}
              disabled={running}
              title="Download structured experiment report"
            >
              Report
            </button>
          )}
        </div>
      </header>

      <section className={`chat-grid${sessionsCollapsed ? " left-collapsed" : ""}${logsCollapsed ? " right-collapsed" : ""}`}>
        <aside
          className={`chat-panel left${sessionsCollapsed ? " collapsed" : ""}`}
          onClick={sessionsCollapsed ? () => setSessionsCollapsed(false) : undefined}
        >
          <div className="session-panel-head" onClick={() => setSessionsCollapsed(!sessionsCollapsed)}>
            <h3>Sessions <span className="panel-toggle-icon">{sessionsCollapsed ? "›" : "‹"}</span></h3>
          </div>
          <button
            type="button"
            className="session-new-btn"
            onClick={() => void createAndActivateSession()}
            disabled={running}
          >
            + New Session
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
                  <small className="session-meta-label">{s.status || "idle"} · {s.history_size} msgs</small>
                </button>
                <button
                  type="button"
                  className="session-delete-btn"
                  onClick={(e) => { e.stopPropagation(); void deleteAndSelectNextSession(s.session_id); }}
                  disabled={running}
                  title="Delete session"
                >
                  ✕
                </button>
              </div>
            ))}
            {sessions.length === 0 && <div className="session-empty">No sessions yet</div>}
          </div>
          {sessionId && !sessionsCollapsed && (
            <div className="session-sidebar-footer">
              <button
                type="button"
                className="session-copy-btn"
                onClick={() => void copySessionId(sessionId)}
                title="Copy full session id"
              >
                {copiedSessionId === sessionId ? "✓ Copied" : "Copy Session ID"}
              </button>
            </div>
          )}
        </aside>

        <section className="chat-panel center">
          <div className="timeline-wrap" ref={timelineRef}>
            <div className="timeline-sticky-header">
              {snapshot && !pipelineDismissed && snapshot.status && snapshot.status !== "idle" && (
                <div className="pipeline-wrap">
                  <PipelineProgress
                    status={snapshot.status}
                    plan={snapshot.plan || []}
                    toolExecutions={snapshot.tool_executions || []}
                  />
                  <button
                    className="pipeline-dismiss"
                    onClick={() => setPipelineDismissed(true)}
                    title="Dismiss"
                  >&times;</button>
                </div>
              )}
              {(snapshot?.history?.length ?? 0) > 0 && (
                <div className="timeline-toolbar">
                  <button
                    className={`timeline-search-toggle${searchOpen ? " active" : ""}`}
                    onClick={() => { setSearchOpen(!searchOpen); if (searchOpen) setSearchQuery(""); }}
                    title="Search messages"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
                    </svg>
                  </button>
                  {searchOpen && (
                    <input
                      className="timeline-search-input"
                      type="text"
                      placeholder="Search messages..."
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
                    alt="Principal Investigator"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src =
                        "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                    }}
                  />
                  <div className="chat-msg-content">
                    <div className="chat-msg-role">Principal Investigator</div>
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
                    alt="Computational Biologist"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src =
                        "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                    }}
                  />
                  <div className="chat-msg-content">
                    <div className="chat-msg-role">Computational Biologist</div>
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
                  alt="Principal Investigator"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src =
                      "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                  }}
                />
                <div className="chat-msg-content">
                  <div className="chat-msg-role">Principal Investigator</div>
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
                  alt="Machine Learning Specialist"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src =
                      "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                  }}
                />
                <div className="chat-msg-content">
                  <div className="chat-msg-role">Machine Learning Specialist</div>
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
                  alt="Principal Investigator"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src =
                      "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
                  }}
                />
                <div className="chat-msg-content">
                  <div className="chat-msg-role">Principal Investigator</div>
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
                placeholder={isWaitingForInteraction ? "Please respond to the form above..." : "Ask anything about AI protein engineering..."}
                disabled={running || quotaExhausted || isWaitingForInteraction}
              />
              {chatQuota?.enforced && (
                <span
                  className={`chat-quota-pill${quotaExhausted ? " exhausted" : ""}`}
                  title={`Per-IP daily limit in online mode: ${chatQuota.limit ?? 10}`}
                >
                  {quotaExhausted
                    ? `${chatQuota.used}/${chatQuota.limit ?? 10}`
                    : `${chatQuota.remaining ?? 0}/${chatQuota.limit ?? 10}`}
                </span>
              )}
            </div>
            <div className="composer-row">
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} aria-label="Model">
                {MODELS.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <div className="file-source-inline">
                <label className={`file-upload-icon-btn${running || quotaExhausted ? " disabled" : ""}`} title="Upload files">
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
                  buttonLabel="From Workspace"
                  onPick={(picked) => setWorkspaceFiles(picked)}
                />
              </div>
              <button className="btn-secondary" onClick={() => void retryLastMessage()} disabled={running || quotaExhausted} title={regenerateTooltip}>
                Regenerate
              </button>
              <button className="btn-secondary" onClick={abortRun} disabled={!running}>
                Stop
              </button>
              <button className="btn-primary" onClick={() => void sendMessage()} disabled={running || quotaExhausted} title={sendTooltip}>
                {running ? "Running..." : "Send"}
              </button>
            </div>
            {(files.length > 0 || workspaceFiles.length > 0) && (
              <div className="file-preview">
                {files.map((f) => (
                  <span key={f.name}>{f.name}</span>
                ))}
                {workspaceFiles.map((f) => (
                  <span key={f.id}>{f.display_name} (workspace)</span>
                ))}
              </div>
            )}
            {error && <ErrorAlert message={error} onDismiss={() => setError("")} />}
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
            <span className="term-head-title">Execution Status</span>
            <span className="panel-toggle-icon">{logsCollapsed ? "+" : "-"}</span>
          </div>
          {!logsCollapsed && (
            <div className="term-body">
              {!terminalData ? (
                <div className="term-line"><span className="term-muted">$ waiting for session...</span></div>
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
                      <span className="term-val">{terminalData.messages} messages, {terminalData.toolRuns} tool runs</span>
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
                          <span className="term-log-content">{c.content || "(empty)"}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </aside>
      </section>
      <PageFooter />
    </div>
  );
}
