import type { ChatHistoryItem } from "./api";

/** Local display only. The server remains the source of persisted history. */
export type PendingChatTurn = {
  sessionId: string;
  text: string;
  attachments: string[];
  previousUsers: ChatHistoryItem[];
};

export function hasAcceptedTurn(history: ChatHistoryItem[], pending: PendingChatTurn): boolean {
  const users = history.filter(item => item.role === "user");
  const last = users[users.length - 1];
  if (!last) return false;
  if (users.length > pending.previousUsers.length) return true;
  // Compaction can shorten history while this request is starting.
  if (last.turn_id && !pending.previousUsers.some(item => item.turn_id === last.turn_id)) return true;
  const text = pending.text.trim();
  return Boolean(text && last.content.trim().startsWith(text)
    && !pending.previousUsers.some(item => item.content === last.content));
}

export function chatHistoryWithPending(history: ChatHistoryItem[], pending: PendingChatTurn | null, sessionId: string): ChatHistoryItem[] {
  if (!pending || pending.sessionId !== sessionId || hasAcceptedTurn(history, pending)) return history;
  const attachments = pending.attachments.map(name => `📎 ${name}`).join("\n");
  const content = [pending.text.trim(), attachments].filter(Boolean).join("\n\n");
  return [...history, { role: "user", content, kind: "text" }];
}
