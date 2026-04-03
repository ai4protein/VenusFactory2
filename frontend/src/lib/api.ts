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

export async function createChatSession() {
  const res = await fetch(`${API_ROOT}/api/chat/sessions`, { method: "POST" });
  if (!res.ok) {
    throw new Error(`Create session failed: ${res.status}`);
  }
  return res.json() as Promise<{ session_id: string; model_name: string; created_at: string }>;
}

export async function getChatSession(sessionId: string) {
  const res = await fetch(`${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}`);
  if (!res.ok) {
    throw new Error(`Fetch session failed: ${res.status}`);
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
      body: form
    }
  );
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }
  return res.json() as Promise<{ files: Array<{ name: string; path: string; size: number }> }>;
}

export async function getChatQuota() {
  const res = await fetch(`${API_ROOT}/api/chat/quota`);
  if (!res.ok) {
    throw new Error(`Fetch chat quota failed: ${res.status}`);
  }
  return res.json() as Promise<ChatQuota>;
}
