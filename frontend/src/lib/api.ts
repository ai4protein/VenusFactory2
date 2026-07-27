export type ChatHistoryItem = {
  role: string;
  content: string;
  role_id?: string;
  /** Marker for ephemeral placeholder messages (thinking / summarizing / writing-*).
   * Set by backend graph nodes so the frontend can detect a placeholder without
   * matching localized substrings. */
  phase?: string;
  /** kimi-code engine writes message kind. "thinking" → collapsible reasoning
   * block rendered above the answer; "status" → foldable research noise;
   * "text" (or undefined) → regular message. */
  kind?: "text" | "thinking" | "status" | "checkpoint";
  /** Optional kimi turn id; used to group thinking + text from the same turn. */
  turn_id?: string;
};

export type ClarificationQuestion = {
  question: string;
  options: string[];
  allow_multiple: boolean;
  header?: string;
  allow_other?: boolean;
  other_label?: string;
};

export type KimiPendingQuestion = {
  question_id: string;
  question_count: number;
};

export type KimiPendingApproval = {
  approval_id: string;
  tool_name: string;
  option_labels: string[];
  approval_prompt?: string;
  plan_markdown?: string;
};

export type ClarificationAnswer = {
  question_index: number;
  selected_options: number[];
  custom_text: string;
};

export type PlanStep = {
  step: number;
  task_description: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
};

export type SecurityEvent = {
  ts: string;
  tool: string;
  action: string;
  reason: string;
  input_preview: string;
};

export type ChatSnapshot = {
  session_id: string;
  model_name: string;
  created_at: string;
  history: ChatHistoryItem[];
  conversation_log: Array<Record<string, unknown>>;
  tool_executions: Array<Record<string, unknown>>;
  status: string;
  clarification_questions: ClarificationQuestion[];
  plan: PlanStep[];
  waiting_for: string;
  security_events?: SecurityEvent[];
  /** Product chat mode when backend persists it: science_agent | science_expert. */
  chat_mode?: string;
  /** Backend engine for the session/turn: kimi-code | graph. */
  engine?: string;
  /** Science Agent AskUserQuestion gate (redacted). */
  kimi_pending_question?: KimiPendingQuestion | null;
  /** Science Agent Approve/Reject gate (redacted). */
  kimi_pending_approval?: KimiPendingApproval | null;
  approval_prompt?: string;
  plan_markdown?: string;
  /** Explicit opt-in: pause after each research sub-report. */
  review_sub_reports?: boolean;
  /** SC paper-level manuscript (default on; kept for API compat). */
  full_manuscript?: boolean;
};

/** Stream body fields for POST .../messages/stream (extra keys ignored by older backends). */
export type ChatStreamBody = {
  text: string;
  model?: string;
  custom_model_config?: {
    model_name: string;
    api_key: string;
    base_url: string;
  };
  custom_model_id?: string;
  attachment_paths?: string[];
  lang?: string;
  /** Product mode: science_agent | science_expert */
  chat_mode?: string;
  /** Engine override / hint: kimi-code | graph */
  engine?: string;
};

export type ChatQuota = {
  mode: string;
  enforced: boolean;
  limit: number | null;
  used: number;
  remaining: number | null;
};

export type InsightsMeta = {
  generated_at: string;
  time_range: { from: string; to: string };
  filters_applied: Record<string, string | null>;
};

export type InsightsOverview = InsightsMeta & {
  data: {
    total_calls: number;
    successful_calls: number;
    failed_calls: number;
    success_rate: number;
    active_owners: number;
    unique_ips: number;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    estimated_cost_usd: number;
  };
};

export type InsightsToolCalls = InsightsMeta & {
  data: {
    group_by: string;
    rows: Array<{
      bucket: string;
      tool_name?: string;
      calls: number;
      successful_calls: number;
      failed_calls: number;
      avg_latency_ms: number;
      total_tokens: number;
    }>;
  };
};

export type InsightsIpDistribution = InsightsMeta & {
  data: {
    level: string;
    rows: Array<{
      country_code: string;
      region: string;
      pv: number;
      uv: number;
    }>;
  };
};

export type InsightsTokenUsage = InsightsMeta & {
  data: {
    group_by: string;
    rows: Array<{
      bucket: string;
      model: string;
      tool_name: string;
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
      estimated_cost_usd: number;
    }>;
  };
};

export type InsightsMap = InsightsMeta & {
  data: {
    rows: Array<{
      country_code: string;
      pv: number;
      uv: number;
      intensity: number;
    }>;
  };
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
    let raw = localStorage.getItem(CHAT_SESSION_TOKEN_MAP_KEY);
    if (!raw) {
      raw = sessionStorage.getItem(CHAT_SESSION_TOKEN_MAP_KEY);
      if (raw) {
        localStorage.setItem(CHAT_SESSION_TOKEN_MAP_KEY, raw);
        sessionStorage.removeItem(CHAT_SESSION_TOKEN_MAP_KEY);
      }
    }
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
  const res = await fetch(`${API_ROOT}/api/chat/sessions`, {
    method: "POST"
  });
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
  return res.json() as Promise<{
    sessions: Array<{
      session_id: string;
      created_at: string;
      model_name: string;
      history_size: number;
      status: string;
      title?: string;
    }>;
  }>;
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

export async function deleteChatSession(sessionId: string) {
  const res = await fetch(`${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    headers: getChatSessionAuthHeaders(sessionId)
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  return res.json() as Promise<{ success: boolean; session_id: string }>;
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

export async function deleteCustomModelCache(customModelId: string) {
  const res = await fetch(`${API_ROOT}/api/chat/models/custom/${encodeURIComponent(customModelId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  return res.json() as Promise<{ success: boolean; custom_model_id: string; cleared_sessions: string[] }>;
}

export async function getChatQuota() {
  const res = await fetch(`${API_ROOT}/api/chat/quota`);
  if (!res.ok) {
    throw new Error(`Fetch chat quota failed: ${res.status}`);
  }
  return res.json() as Promise<ChatQuota>;
}

export function getClarificationRespondUrl(sessionId: string): string {
  return `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/clarification/respond/stream`;
}

export function getAskUserRespondUrl(sessionId: string): string {
  return `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/ask-user/respond/stream`;
}

export function getApprovalDecideUrl(sessionId: string): string {
  return `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/approval/decide/stream`;
}

export function getPlanConfirmUrl(sessionId: string): string {
  return `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/plan/confirm/stream`;
}

export function getSubReportDecideUrl(sessionId: string): string {
  return `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/sub-report/decide/stream`;
}

export function getExperimentReportUrl(sessionId: string): string {
  return `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/report`;
}

export async function downloadExperimentReport(sessionId: string) {
  const res = await fetch(getExperimentReportUrl(sessionId), {
    headers: getChatSessionAuthHeaders(sessionId),
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  const disposition = res.headers.get("Content-Disposition") || "";
  const filenameMatch = disposition.match(/filename="?([^"]+)"?/);
  const filename = filenameMatch?.[1] || `experiment_report_${sessionId.slice(0, 8)}.md`;
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function exportChatSessionBundle(sessionId: string) {
  const res = await fetch(`${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/export`, {
    method: "POST",
    headers: getChatSessionAuthHeaders(sessionId),
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  const disposition = res.headers.get("Content-Disposition") || "";
  const filenameMatch = disposition.match(/filename="?([^"]+)"?/);
  const filename = filenameMatch?.[1] || `chat_export_${sessionId.slice(0, 8)}.zip`;
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function getStepDecideUrl(sessionId: string): string {
  return `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/step/decide/stream`;
}

export async function iterationDecide(
  sessionId: string,
  action: "satisfied" | "modify_plan" | "continue"
) {
  const res = await fetch(
    `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/iteration/decide`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getChatSessionAuthHeaders(sessionId),
      },
      body: JSON.stringify({ action }),
    }
  );
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  return res.json() as Promise<{ success: boolean; status: string; plan?: PlanStep[] }>;
}

export async function submitFeedback(
  sessionId: string,
  messageIndex: number,
  rating: "like" | "dislike",
  comment: string = ""
) {
  const res = await fetch(
    `${API_ROOT}/api/chat/sessions/${encodeURIComponent(sessionId)}/feedback`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getChatSessionAuthHeaders(sessionId),
      },
      body: JSON.stringify({ message_index: messageIndex, rating, comment }),
    }
  );
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(parseErrorStatus(res.status, detail));
  }
  return res.json() as Promise<{ success: boolean; session_id: string; message_index: number }>;
}

function withTimeRange(fromIso: string, toIso: string) {
  const params = new URLSearchParams();
  params.set("from", fromIso);
  params.set("to", toIso);
  return params;
}

export async function getInsightsOverview(fromIso: string, toIso: string) {
  const params = withTimeRange(fromIso, toIso);
  const res = await fetch(`${API_ROOT}/api/settings/insights/overview?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Fetch insights overview failed: ${res.status}`);
  }
  return res.json() as Promise<InsightsOverview>;
}

export async function getInsightsToolCalls(fromIso: string, toIso: string, groupBy: "tool" | "day" | "hour") {
  const params = withTimeRange(fromIso, toIso);
  params.set("group_by", groupBy);
  const res = await fetch(`${API_ROOT}/api/settings/insights/tool-calls?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Fetch tool calls failed: ${res.status}`);
  }
  return res.json() as Promise<InsightsToolCalls>;
}

export async function getInsightsIpDistribution(fromIso: string, toIso: string, level: "country" | "region") {
  const params = withTimeRange(fromIso, toIso);
  params.set("level", level);
  const res = await fetch(`${API_ROOT}/api/settings/insights/ip-distribution?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Fetch IP distribution failed: ${res.status}`);
  }
  return res.json() as Promise<InsightsIpDistribution>;
}

export async function getInsightsTokenUsage(fromIso: string, toIso: string, groupBy: "model" | "tool" | "day") {
  const params = withTimeRange(fromIso, toIso);
  params.set("group_by", groupBy);
  const res = await fetch(`${API_ROOT}/api/settings/insights/token-usage?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Fetch token usage failed: ${res.status}`);
  }
  return res.json() as Promise<InsightsTokenUsage>;
}

export async function getInsightsMap(fromIso: string, toIso: string) {
  const params = withTimeRange(fromIso, toIso);
  const res = await fetch(`${API_ROOT}/api/settings/insights/map?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Fetch map data failed: ${res.status}`);
  }
  return res.json() as Promise<InsightsMap>;
}
