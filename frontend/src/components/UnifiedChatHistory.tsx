import { useState } from "react";
import { useHistoryPins } from "../lib/useHistoryPins";
import { CHAT_MODE_LABELS, historyRowKey, visibleChatHistory, type ChatHistoryRow } from "../lib/unifiedChatHistory";
import { ChatModeIcon } from "./ChatModeIcon";
import { HistorySessionButton } from "./HistorySessionButton";
import { HistoryPinButton } from "./HistoryPinButton";
import { RunSpinner } from "./RunSpinner";
import "./UnifiedChatHistory.css";

export function UnifiedChatHistory({ rows, activeKey, lang, disabled, onOpen, onDelete }: {
  rows: ChatHistoryRow[]; activeKey: string; lang: string; disabled: boolean;
  onOpen: (row: ChatHistoryRow) => void; onDelete: (row: ChatHistoryRow) => Promise<boolean>;
}) {
  const [query, setQuery] = useState(""), [deleting, setDeleting] = useState("");
  const agent = useHistoryPins("science_agent"), scientist = useHistoryPins("science_expert"), researcher = useHistoryPins("researcher");
  const pins = { science_agent: agent, science_expert: scientist, researcher };
  const visible = visibleChatHistory(rows, query, { science_agent: agent.ids, science_expert: scientist.ids, researcher: researcher.ids });
  const zh = lang === "zh";
  return <aside className="chat-panel left unified-chat-history" aria-label={zh ? "对话记录" : "Conversations"}>
    <div className="sidebar-history-heading">{zh ? "最近对话" : "Recent Chats"}</div>
    <label className="session-search"><input type="search" value={query} onChange={event => setQuery(event.target.value)}
      aria-label={zh ? "搜索对话" : "Search conversations"} placeholder={zh ? "搜索对话…" : "Search conversations…"}/></label>
    <div className="session-list">{visible.map(row => {
      const key = historyRowKey(row), pinned = pins[row.mode].ids.includes(row.id);
      return <div className={`session-item history-entry${key === activeKey ? " active" : ""}`} key={key} data-chat-mode={row.mode}>
        <HistorySessionButton date={row.date} lang={lang} label={`${CHAT_MODE_LABELS[row.mode]} · ${row.title}`}
          className="session-select-btn" aria-current={key === activeKey ? "page" : undefined} aria-busy={row.running}
          disabled={disabled || row.pending} onClick={() => onOpen(row)}>
          <ChatModeIcon mode={row.mode}/><span className="session-title-label">{row.title}</span>{row.running && <RunSpinner lang={lang}/>}
        </HistorySessionButton>
        {!row.pending && <HistoryPinButton pinned={pinned} label={row.title} lang={lang} onToggle={() => pins[row.mode].setPinned(row.id, !pinned)}/>}
        {row.canDelete && <button type="button" className="session-delete-btn" disabled={disabled || row.running || !!deleting}
          title={zh ? "删除对话" : "Delete conversation"} aria-label={`${zh ? "删除" : "Delete"} ${row.title}`}
          onClick={() => { setDeleting(key); void onDelete(row).then(removed => { if (removed) pins[row.mode].setPinned(row.id, false); }).finally(() => setDeleting("")); }}>×</button>}
      </div>;
    })}{!visible.length && <div className="session-empty">{query ? (zh ? "没有匹配的对话" : "No matching conversations") : (zh ? "暂无对话" : "No conversations yet")}</div>}</div>
  </aside>;
}
