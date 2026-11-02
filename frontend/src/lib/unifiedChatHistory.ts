export type WorkspaceChatMode = "science_agent" | "science_expert" | "researcher";
export const CHAT_MODE_LABELS: Record<WorkspaceChatMode, string> = {
  science_agent: "Agent", science_expert: "Scientist", researcher: "Researcher",
};
export type ChatHistoryRow = {
  id: string; mode: WorkspaceChatMode; title: string; date: string | number;
  running: boolean; canDelete: boolean; pending?: boolean;
};
export type ResearchHistoryRow = { id: string; title: string; created: number; status: string; job_number: string };
export const OPEN_NEW_CHAT = "vf2:open-new-chat";
export function claimNewChatDraft(mode: WorkspaceChatMode, nonce: string, storage: Pick<Storage, "getItem" | "setItem">) {
  const key = `vf2:new-chat-draft:${mode}`;
  try {
    if (storage.getItem(key) === nonce) return false;
    storage.setItem(key, nonce);
  } catch { /* Drafts also work when storage is unavailable. */ }
  return true;
}
export function historyRowKey(row: Pick<ChatHistoryRow, "id" | "mode">) { return `${row.mode}:${row.id}`; }
export function newChatLocation(lang: string, mode: WorkspaceChatMode, nonce: string) {
  const params = new URLSearchParams({ new: nonce });
  if (mode !== "science_agent") params.set("mode", mode === "science_expert" ? "scientist" : "researcher");
  return `/${lang === "zh" ? "zh" : "en"}/agent/chat?${params}`;
}
export function visibleChatHistory(rows: ChatHistoryRow[], query: string, pins: Record<WorkspaceChatMode, string[]>) {
  const needle = query.trim().toLocaleLowerCase();
  const unique = new Map(rows.map(row => [historyRowKey(row), row]));
  const time = (row: ChatHistoryRow) => new Date(row.date).getTime() || 0;
  return [...unique.values()].filter(row => !needle || `${row.title} ${CHAT_MODE_LABELS[row.mode]}`.toLocaleLowerCase().includes(needle))
    .sort((a, b) => Number(pins[b.mode].includes(b.id)) - Number(pins[a.mode].includes(a.id)) || time(b) - time(a) || historyRowKey(a).localeCompare(historyRowKey(b)));
}
