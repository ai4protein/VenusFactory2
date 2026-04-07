import { useEffect, useMemo, useRef, useState } from "react";
import { ChatTimeline } from "../components/ChatTimeline";
import {
  cancelChatSession,
  createChatSession,
  deleteChatSession,
  getChatQuota,
  getChatSession,
  getChatSessionAuthHeaders,
  listChatSessions,
  type ChatQuota,
  type ChatSnapshot,
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
  const [error, setError] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
  const [chatQuota, setChatQuota] = useState<ChatQuota | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const SESSION_STORAGE_KEY = "vf2_active_session_id";
  const SESSION_CACHE_KEY = "vf2_session_list_cache";
  const COPY_HINT_MS = 1200;
  const [copiedSessionId, setCopiedSessionId] = useState("");

  useEffect(() => {
    void bootstrapSession();
  }, []);

  useEffect(() => {
    if (timelineRef.current) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
  }, [snapshot?.history.length]);

  useEffect(() => {
    const s = (snapshot?.status || "").toLowerCase();
    if (!s) return;
    if (s === "stopped") {
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
    const list = data.sessions || [];
    setSessions(list);
    localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(list));
    return list;
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
    if (!list.length) {
      setSessionId("");
      setSnapshot(null);
      return;
    }

    const remembered = sessionStorage.getItem(SESSION_STORAGE_KEY);
    const target =
      (remembered && list.find((item) => item.session_id === remembered)?.session_id) ||
      list[0].session_id;
    await refreshCurrentSession(target);
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
        ({ event, data }) => {
          if (event === "state" && data) {
            const payload = JSON.parse(data) as ChatSnapshot;
            setSnapshot(payload);
          }
        },
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
        ({ event, data }) => {
          if (event === "state" && data) {
            const payload = JSON.parse(data) as ChatSnapshot;
            setSnapshot(payload);
          }
        },
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
            <ChatTimeline items={snapshot?.history || []} />
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
              placeholder="Ask anything about AI protein engineering..."
              disabled={running || quotaExhausted}
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
