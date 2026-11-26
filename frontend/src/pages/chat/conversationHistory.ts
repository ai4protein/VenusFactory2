import type { SessionMeta } from "../../lib/chatSessions";
import { mergeHistoryActivity, type HistoryActivity } from "../../lib/optimisticHistory";
import { isRunActive } from "../../lib/runProgress";
import { historyRowKey, type ChatHistoryRow, type ResearchHistoryRow, type WorkspaceChatMode } from "../../lib/unifiedChatHistory";

type ConversationHistory = {
  sessions: SessionMeta[];
  reports: ResearchHistoryRow[];
  activity: Record<string, HistoryActivity>;
  activeMode: WorkspaceChatMode;
  activeId: string;
  activeTitle: string;
  agentRunId: string;
  scientistRunId: string;
  online: boolean;
  untitled: string;
};

export function conversationHistory({
  sessions, reports, activity, activeMode, activeId, activeTitle, agentRunId, scientistRunId, online, untitled,
}: ConversationHistory) {
  const chats: ChatHistoryRow[] = sessions.map(session => ({
    id: session.session_id,
    mode: session.chat_mode || "science_agent",
    title: (activeMode !== "researcher" && session.session_id === activeId && activeTitle) || session.title?.trim() || untitled,
    date: session.created_at,
    running: session.session_id === agentRunId || session.session_id === scientistRunId || isRunActive(session.status),
    canDelete: true,
  }));
  const research: ChatHistoryRow[] = reports.map(report => ({
    id: report.id,
    mode: "researcher",
    title: report.title,
    date: report.created * 1000,
    running: ["queued", "running"].includes(report.status),
    canDelete: online,
  }));
  const selected = activeId || Object.values(activity).find(row => row.mode === activeMode)?.id || "";
  return { rows: mergeHistoryActivity([...chats, ...research], activity), activeKey: historyRowKey({ mode: activeMode, id: selected }) };
}
