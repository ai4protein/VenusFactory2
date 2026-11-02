import { CHAT_MODE_LABELS, type WorkspaceChatMode } from "../lib/unifiedChatHistory";

export function ChatModeIcon({ mode }: { mode: WorkspaceChatMode }) {
  return <svg className="chat-mode-icon" viewBox="0 0 24 24" role="img" aria-label={CHAT_MODE_LABELS[mode]}>
    {mode === "science_agent" ? <><path d="M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 3V6a2 2 0 0 1 2-2Z"/><path d="M7 9h10M7 13h6"/></>
      : mode === "science_expert" ? <path d="M9 3h6M10 3v7L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2l-6-9V3M7 16h10"/>
      : <><circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6M10 7v6"/></>}
  </svg>;
}
