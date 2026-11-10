import type { ChatMode } from "./useModelRegistry";
import type { HistoryActivity } from "./optimisticHistory";

export type ModeSession = { session_id: string; chat_mode?: ChatMode };

export function activeSessionStorageKey(mode: ChatMode) {
  return `vf2_active_session_id:${mode}`;
}

export function sessionsForMode<T extends ModeSession>(sessions: T[], mode: ChatMode): T[] {
  return sessions.filter(session => (session.chat_mode || "science_agent") === mode);
}

export function sessionCandidates(sessions: ModeSession[], mode: ChatMode, remembered: string | null, legacy: string | null = null) {
  if (remembered === "") return []; // An explicitly opened draft stays a draft.
  const ids = sessionsForMode(sessions, mode).map(session => session.session_id);
  const preferred = remembered ?? legacy;
  return preferred && ids.includes(preferred) ? [preferred, ...ids.filter(id => id !== preferred)] : ids;
}

export type SessionMeta = {
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
export function abbreviateSessionTitle(raw: string, maxLen = 22): string {
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

export function titleFromHistory(history: Array<{ role?: string; content?: unknown }> | undefined): string {
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

export const SESSION_CACHE_KEY = "vf2_session_list_cache_v2";

export function hasSessionContent(item: SessionMeta) {
  return item.has_content ?? Boolean(item.history_size || item.title?.trim());
}

export function readSessionCache(storage: Pick<Storage, "getItem">): SessionMeta[] {
  try {
    const value: unknown = JSON.parse(storage.getItem(SESSION_CACHE_KEY) || "[]");
    if (!Array.isArray(value)) return [];
    return value.filter((row): row is SessionMeta => row && typeof row === "object" &&
      typeof row.session_id === "string" && Boolean(row.session_id) &&
      typeof row.created_at === "string" && typeof row.model_name === "string" &&
      typeof row.history_size === "number" && typeof row.status === "string" &&
      (row.title === undefined || typeof row.title === "string") &&
      (row.has_content === undefined || typeof row.has_content === "boolean") &&
      (row.chat_mode === undefined || row.chat_mode === "science_agent" || row.chat_mode === "science_expert")
    ).filter(hasSessionContent);
  } catch { return []; }
}

export function writeSessionCache(storage: Pick<Storage, "setItem">, rows: SessionMeta[]) {
  try {
    // A newly submitted title is not yet a persisted conversation.
    const accepted = rows.filter(row => row.has_content === true || row.history_size > 0);
    storage.setItem(SESSION_CACHE_KEY, JSON.stringify(accepted));
  } catch { /* Authenticated server history remains authoritative. */ }
}

export function mergeSessionMetadata(previous: SessionMeta[], received: SessionMeta[]): SessionMeta[] {
  const byId = new Map(previous.map(row => [row.session_id, row]));
  return received.filter(hasSessionContent).map(row => {
    const title = row.title?.trim() || byId.get(row.session_id)?.title?.trim() || "";
    return { ...row, title: abbreviateSessionTitle(title) || title };
  });
}

export type SessionHistoryState = {
  sessions: SessionMeta[];
  activity: Record<string, HistoryActivity>;
  revision: number;
  request: { id: number; revision: number } | null;
};
export type SessionHistoryAction =
  | { type: "update"; update: SessionMeta[] | ((rows: SessionMeta[]) => SessionMeta[]) }
  | { type: "activity"; key: string; row: HistoryActivity | null }
  | { type: "request"; id: number }
  | { type: "received"; id: number; sessions: SessionMeta[] };

export function sessionHistoryState(sessions: SessionMeta[] = []): SessionHistoryState {
  return { sessions, activity: {}, revision: 0, request: null };
}

/** One ordering boundary for bootstrap, refresh and immediate user actions. */
export function reduceSessionHistory(state: SessionHistoryState, action: SessionHistoryAction): SessionHistoryState {
  switch (action.type) {
    case "update":
      return { ...state, revision: state.revision + 1,
        sessions: typeof action.update === "function" ? action.update(state.sessions) : action.update };
    case "activity": {
      const activity = { ...state.activity };
      if (action.row) activity[action.key] = action.row;
      else delete activity[action.key];
      return { ...state, activity, revision: state.revision + 1 };
    }
    case "request":
      return { ...state, request: { id: action.id, revision: state.revision } };
    case "received":
      if (state.request?.id !== action.id) return state;
      if (state.request.revision !== state.revision) return { ...state, request: null };
      return { ...state, sessions: mergeSessionMetadata(state.sessions, action.sessions), request: null };
  }
}
