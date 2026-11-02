import { useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { warmPage } from "../lib/pageLoaders";
import { newChatLocation, OPEN_NEW_CHAT } from "../lib/unifiedChatHistory";

export function NewChatButton({ lang, onNavigate }: { lang: string; onNavigate: () => void }) {
  const navigate = useNavigate();
  const label = lang === "zh" ? "新对话" : "New Chat";
  const start = useCallback(() => {
    navigate(newChatLocation(lang, "science_agent", crypto.randomUUID()));
    onNavigate();
  }, [lang, navigate, onNavigate]);
  useEffect(() => {
    window.addEventListener(OPEN_NEW_CHAT, start);
    return () => window.removeEventListener(OPEN_NEW_CHAT, start);
  }, [start]);
  return <button type="button" className="app-sidebar-link" aria-label={label} title={label}
    onPointerEnter={() => warmPage("ChatPage")} onFocus={() => warmPage("ChatPage")} onClick={start}>
    <svg className="app-sidebar-icon" aria-hidden="true" viewBox="0 0 24 24"><path d="M13 5H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8M16 3l5 5M9 15l1-5L19 1l5 5-9 9-5 1Z" transform="translate(0 1) scale(.9)"/></svg>
    <span>{label}</span>
  </button>;
}
