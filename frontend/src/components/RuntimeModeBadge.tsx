import { useLang } from "../lib/i18n";

type RuntimeMode = "unknown" | "local" | "online";

type RuntimeModeBadgeProps = {
  runtimeMode: RuntimeMode;
  placement?: "floating" | "sidebar";
};

const STRINGS = {
  en: {
    online: "Mode: Online",
    local: "Mode: Local",
    unknown: "Mode: Checking..."
  },
  zh: {
    online: "模式：在线",
    local: "模式：本地",
    unknown: "模式：检测中…"
  }
};

export function RuntimeModeBadge({ runtimeMode, placement = "floating" }: RuntimeModeBadgeProps) {
  const t = useLang().t(STRINGS);
  const statusClass: "running" | "stopped" = runtimeMode === "online" ? "running" : "stopped";
  const text = runtimeMode === "online" ? t.online : runtimeMode === "local" ? t.local : t.unknown;
  return (
    <div className={`runtime-mode-badge-wrap runtime-mode-badge-wrap--${placement}`} aria-live="polite">
      <div className={`run-status-bar ${statusClass}`}>
        <span className="run-status-dot" />
        <span className="run-status-text">{text}</span>
      </div>
    </div>
  );
}
