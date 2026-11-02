import { useCallback, useRef, useState } from "react";
import { connectionNotice } from "../lib/researchUi";
import "./ChatNotice.css";

/** Repeated background failures stay dismissed until a new action clears them. */
export function useDismissibleError() {
  const [message, setMessage] = useState("");
  const dismissed = useRef("");
  const show = useCallback((next: string) => {
    if (!next) dismissed.current = "";
    if (!next || next !== dismissed.current) setMessage(next);
  }, []);
  const dismiss = useCallback(() => {
    dismissed.current = message;
    setMessage("");
  }, [message]);
  return [message, show, dismiss] as const;
}

export function ChatNotice({ message, onDismiss, lang }: { message: string; onDismiss: () => void; lang: string }) {
  if (!message) return null;
  const text = connectionNotice(message, lang === "zh");
  return <div className="chat-toast">
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="12" r="9"/><path d="M12 7v6m0 3v1"/></svg>
    <span className="chat-toast-message" role="status" aria-live="polite" aria-atomic="true">{text}</span>
    <button type="button" onClick={onDismiss} aria-label={lang === "zh" ? "关闭提示" : "Dismiss notification"}>×</button>
  </div>;
}
