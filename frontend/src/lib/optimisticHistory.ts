import { historyRowKey, type ChatHistoryRow } from "./unifiedChatHistory";

export type HistoryActivity = ChatHistoryRow & { pending?: boolean };
export type HistoryActivityCallback = (key: string, row: HistoryActivity | null) => void;

/** A submission is visible before a server identity exists. Never persist these rows. */
export function mergeHistoryActivity(rows: ChatHistoryRow[], activity: Record<string, HistoryActivity>): HistoryActivity[] {
  const result = new Map(rows.map(row => [historyRowKey(row), row]));
  for (const row of Object.values(activity)) {
    const key = historyRowKey(row), previous = result.get(key);
    result.set(key, { ...previous, ...row,
      title: previous?.title && !/^(New chat|New Chat|新对话)$/.test(previous.title) ? previous.title : row.title,
      canDelete: false,
    });
  }
  return [...result.values()];
}

export function submissionHistoryRow(mode: ChatHistoryRow["mode"], key: string, title: string, id = "", date = Date.now()): HistoryActivity {
  return { id: id || `pending:${key}`, mode, title: title.trim().replace(/\s+/g, " ").slice(0, 64),
    date, running: true, canDelete: false, pending: !id };
}
