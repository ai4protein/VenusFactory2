import "./HistoryPinButton.css";

export function HistoryPinButton({ pinned, label, lang, onToggle }: {
  pinned: boolean; label: string; lang: string; onToggle: () => void;
}) {
  const action = lang === "zh" ? (pinned ? "取消置顶" : "置顶") : (pinned ? "Unpin" : "Pin");
  return <button type="button" className="history-pin-button" aria-pressed={pinned}
    aria-label={`${action}: ${label}`} title={action}
    onClick={event => { event.stopPropagation(); onToggle(); }}>
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l-1 6 4 4v2H6v-2l4-4-1-6Z"/><path d="M12 15v6"/></svg>
  </button>;
}
