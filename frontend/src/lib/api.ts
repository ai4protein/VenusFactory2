export type ChatHistoryItem = {
  role: string;
  content: string;
  role_id?: string;
};

export type ChatSnapshot = {
  session_id: string;
  model_name: string;
  created_at: string;
  history: ChatHistoryItem[];
  conversation_log: Array<Record<string, unknown>>;
  tool_executions: Array<Record<string, unknown>>;
  status: string;
};

export type ChatQuota = {
  mode: string;
  enforced: boolean;
  limit: number | null;
  used: number;
  remaining: number | null;
};

const API_ROOT = "";
const CHAT_SESSION_TOKEN_MAP_KEY = "vf2_chat_session_token_map";

type ChatSessionTokenEntry = {
  token: string;
  expires_at: string;
};

type ChatSessionTokenMap = Record<string, ChatSessionTokenEntry>;

function loadTokenMap(): ChatSessionTokenMap {
  try {
    const raw = localStorage.getItem(CHAT_SESSION_TOKEN_MAP_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as ChatSessionTokenMap;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveTokenMap(map: ChatSessionTokenMap) {
  localStorage.setItem(CHAT_SESSION_TOKEN_MAP_KEY, JSON.stringify(map));
}

function persistSessionToken(sessionId: string, token: string, expiresAt: string) {
  if (!sessionId || !token) return;
  const map = loadTokenMap();
  map[sessionId] = { token, expires_at: expiresAt };
  saveTokenMap(map);
}

function readSessionToken(sessionId: string): string {
  const map = loadTokenMap();
  const entry = map[sessionId];
  if (!entry?.token) return "";
  if (entry.expires_at) {
    const ts = Date.parse(entry.expires_at);
    if (!Number.isNaN(ts) && Date.now() >= ts) {
      delete map[sessionId];
      saveTokenMap(map);
      return "";
    }
  }
  return entry.token;
}

export function getChatSessionAuthHeaders(sessionId: string): Record<string, string> {
  const token = readSessionToken(sessionId);
  return token ? { "x-session-access-token": token } : {};
}

function parseErrorStatus(status: number, detail: string): string {
  if (status === 401 || status === 403) {
    return detail || "Session authentication failed.";
  }
  return detail || `Request failed (${status})`;
}

async function extractErrorDetail(res: Response): Promise<string> {
  try {
    const payload = (await res.json()) as { detail?: string | { message?: string } };
    const detail = payload?.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && typeof detail.message === "string") return detail.message;
  } catch {
    // ignore parse error
  }
  return "";
}

export async function createChatSession() {
  const res = await fetch(`${API_ROOT}/api/chat/sessions`, { method: "POST" });
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  const data = (await res.json()) as {
    session_id: string;
    model_name: string;
    created_at: string;
    session_access_token?: string;
    token_expires_at?: string;
  };
  if (data.session_access_token) {
    persistSessionToken(data.session_id, data.session_access_token, data.token_expires_at || "");
  }
  return data;
}

export async function listChatSessions() {
  const res = await fetch(`${API_ROOT}/api/chat/sessions`);
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  return res.json() as Promise<{ sessions: Array<{ session_id: string; created_at: string; model_name: string; history_size: number; status: string }> }>;
}

export async function getChatSession(sessionId: string) {
  const res = await fetch(`${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    headers: getChatSessionAuthHeaders(sessionId)
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  return res.json() as Promise<ChatSnapshot>;
}

export async function uploadFiles(sessionId: string, files: File[]) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const res = await fetch(
    `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/attachments`,
    {
      method: "POST",
      headers: getChatSessionAuthHeaders(sessionId),
      body: form
    }
  );
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  return res.json() as Promise<{ files: Array<{ name: string; path: string; size: number }> }>;
}

export async function cancelChatSession(sessionId: string) {
  const res = await fetch(`${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/cancel`, {
    method: "POST",
    headers: getChatSessionAuthHeaders(sessionId)
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  return res.json() as Promise<{ success: boolean; status: string }>;
}

export async function getChatQuota() {
  const res = await fetch(`${API_ROOT}/api/chat/quota`);
  if (!res.ok) {
    throw new Error(`Fetch chat quota failed: ${res.status}`);
  }
  return res.json() as Promise<ChatQuota>;
}
