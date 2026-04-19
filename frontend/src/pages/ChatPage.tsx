import { useEffect, useMemo, useRef, useState } from "react";
import { ChatTimeline } from "../components/ChatTimeline";
import { PipelineProgress } from "../components/PipelineProgress";
import { ClarificationForm } from "../components/ClarificationForm";
import { IterationDecision } from "../components/IterationDecision";
import { PlanEditor } from "../components/PlanEditor";
import {
  cancelChatSession,
  createChatSession,
  deleteChatSession,
  getChatQuota,
  getChatSession,
  getChatSessionAuthHeaders,
  getClarificationRespondUrl,
  getPlanConfirmUrl,
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

function createPageInstanceId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
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
  const pageInstanceIdRef = useRef(createPageInstanceId());
  const SESSION_STORAGE_KEY = `vf2_active_session_id__${pageInstanceIdRef.current}`;
  const SESSION_CACHE_KEY = `vf2_session_list_cache__${pageInstanceIdRef.current}`;
  const SESSION_OWNED_KEY = `vf2_session_ids__${pageInstanceIdRef.current}`;
  const COPY_HINT_MS = 1200;
  const [copiedSessionId, setCopiedSessionId] = useState("");

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
    if (s === "stopped" || s === "waiting_for_clarification" || s === "waiting_for_plan_confirmation" || s === "waiting_for_iteration") {
      setRunStatus("stopped");
      return;
    }
    if (s !== "completed") {
      setRunStatus((prev) => (prev === "stopping" ? prev : "running"));
      return;
    }
    setRunStatus("stopped");
  }, [snapshot?.status]);

  const logsPreview = useMemo(() => {
    if (!snapshot) return "No logs yet.";
    const conv = snapshot.conversation_log.slice(-6);
    const tools = snapshot.tool_executions.slice(-12);
    const lines = [
      `status: ${snapshot.status}`,
      `messages: ${snapshot.history.length}`,
      `tool_runs: ${snapshot.tool_executions.length}`,
      "",
      "recent conversation:"
    ];
    conv.forEach((entry, idx) => {
      lines.push(`${idx + 1}. ${(entry.role as string) || "unknown"}: ${(entry.content as string) || ""}`);
    });
    lines.push("", "recent tools:");
    tools.forEach((entry, idx) => {
      lines.push(`${idx + 1}. ${String(entry.tool_name || "tool")} (${String(entry.timestamp || "")})`);
    });
    return lines.join("\n");
  }, [snapshot]);

  async function fetchSessions() {
    const data = await listChatSessions();
    const list = filterOwnedSessions(data.sessions || []);
    setSessions(list);
    sessionStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(list));
    return list;
  }

  function readOwnedSessionIds(): string[] {
    try {
      const parsed = JSON.parse(sessionStorage.getItem(SESSION_OWNED_KEY) || "[]") as unknown;
      return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string" && Boolean(item)) : [];
    } catch {
      return [];
    }
  }

  function writeOwnedSessionIds(ids: string[]) {
    sessionStorage.setItem(SESSION_OWNED_KEY, JSON.stringify(Array.from(new Set(ids.filter(Boolean)))));
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
    const created = await createChatSession();
    rememberOwnedSession(created.session_id);
    setSessionId(created.session_id);
    sessionStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
    setSelectedModel(modelLabelFromInternal(created.model_name));
    const s = await getChatSession(created.session_id);
    setSnapshot(s);
    await fetchSessions();
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
      const raw = sessionStorage.getItem(SESSION_CACHE_KEY);
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
    if (remembered && list.find((item) => item.session_id === remembered)) {
      await refreshCurrentSession(remembered);
      return;
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
    const s = await getChatSession(sid);
    setSnapshot(s);
    setSessionId(sid);
    sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
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

  async function confirmPlan(plan: PlanStep[]) {
    if (!sessionId || running) return;
    setError("");
    setRunning(true);
    setRunStatus("running");
    abortRef.current = new AbortController();
    try {
      await streamSSEFromPost(
        getPlanConfirmUrl(sessionId),
        { plan },
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

  async function copySessionId(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedSessionId(value);
      window.setTimeout(() => setCopiedSessionId(""), COPY_HINT_MS);
    } catch {
      setError("Failed to copy session id.");
    }
  }

  function onComposerKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter") return;
    // Keep Shift+Enter for newline; ignore IME composition Enter.
    if (e.shiftKey || e.nativeEvent.isComposing) return;
    e.preventDefault();
    void sendMessage();
  }

  const isWaitingForInteraction = snapshot?.status === "waiting_for_clarification" || snapshot?.status === "waiting_for_plan_confirmation" || snapshot?.status === "waiting_for_iteration";
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
          <div className="chat-header-subrow">
            <p>Chat with the AI assistant for protein engineering workflows and analysis.</p>
          </div>
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
          <button onClick={() => void createAndActivateSession()}>New Session</button>
          <button onClick={() => void fetchSessions()}>Refresh Sessions</button>
          <button onClick={() => void deleteAndSelectNextSession(sessionId)} disabled={!sessionId || running}>
            Delete Session
          </button>
        </div>
      </header>

      <section className="chat-grid">
        <aside className="chat-panel left">
          <div className="session-panel-head">
            <h3>Sessions</h3>
          </div>
          <div className="session-list">
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
                  <span>{s.session_id.slice(0, 8)}</span>
                  <small>{new Date(s.created_at).toLocaleString()}</small>
                  <small>{s.status || "idle"} - {s.history_size} messages</small>
                </button>
              </div>
            ))}
            {sessions.length === 0 && <div className="session-empty">No sessions yet.</div>}
          </div>
          {sessionId && (
            <button
              type="button"
              className="session-copy-btn"
              onClick={() => void copySessionId(sessionId)}
              title="Copy full session id"
            >
              {copiedSessionId === sessionId ? "Session ID copied" : "Copy Session ID"}
            </button>
          )}
        </aside>

        <section className="chat-panel center">
          <div className="timeline-wrap" ref={timelineRef}>
            {snapshot && running && (
              <PipelineProgress
                status={snapshot.status}
                plan={snapshot.plan || []}
                toolExecutions={snapshot.tool_executions || []}
              />
            )}
            <ChatTimeline
              items={snapshot?.history || []}
              streamingIndex={streamingIdx}
              onSuggestedPrompt={(text) => setMessage(text)}
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
            {chatQuota?.enforced && (
              <div className="chat-quota-hint" title={`Per-user daily limit in online mode: ${chatQuota.limit ?? 10}`}>
                {quotaExhausted
                  ? `Online mode quota reached (${chatQuota.used}/${chatQuota.limit ?? 10}).`
                  : `Online mode quota: ${chatQuota.remaining ?? 0}/${chatQuota.limit ?? 10} remaining for this user today.`}
              </div>
            )}
            <textarea
              rows={4}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder={isWaitingForInteraction ? "Please respond to the form above..." : "Ask anything about AI protein engineering..."}
              disabled={running || quotaExhausted || isWaitingForInteraction}
            />
            <div className="composer-row">
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} aria-label="Model">
                {MODELS.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <div className="file-source-inline">
                <input
                  type="file"
                  multiple
                  onChange={(e) => setFiles(Array.from(e.target.files || []))}
                  disabled={running || quotaExhausted}
                />
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running || quotaExhausted}
                  allowMultiple
                  buttonLabel="From Workspace"
                  onPick={(picked) => setWorkspaceFiles(picked)}
                />
              </div>
              <button onClick={() => void sendMessage()} disabled={running || quotaExhausted} title={sendTooltip}>
                {running ? "Running..." : "Send"}
              </button>
              <button onClick={() => void retryLastMessage()} disabled={running || quotaExhausted} title={regenerateTooltip}>
                Regenerate
              </button>
              <button onClick={abortRun} disabled={!running}>
                Stop
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
            {error && <div className="error-box">{error}</div>}
          </div>
        </section>

        <aside className="chat-panel right">
          <h3>Execution Status</h3>
          <pre>{logsPreview}</pre>
        </aside>
      </section>
      <PageFooter />
    </div>
  );
}
