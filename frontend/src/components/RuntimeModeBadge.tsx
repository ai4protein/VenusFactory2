type RuntimeMode = "unknown" | "local" | "online";

type RuntimeModeBadgeProps = {
  runtimeMode: RuntimeMode;
};

function getBadgeMeta(runtimeMode: RuntimeMode): { statusClass: "running" | "stopped"; text: string } {
  if (runtimeMode === "online") {
    return { statusClass: "running", text: "Mode: Online" };
  }
  if (runtimeMode === "local") {
    return { statusClass: "stopped", text: "Mode: Local" };
  }
  return { statusClass: "stopped", text: "Mode: Checking..." };
}

export function RuntimeModeBadge({ runtimeMode }: RuntimeModeBadgeProps) {
  const meta = getBadgeMeta(runtimeMode);
  return (
    <div className="runtime-mode-badge-wrap" aria-live="polite">
      <div className={`run-status-bar ${meta.statusClass}`}>
        <span className="run-status-dot" />
        <span className="run-status-text">{meta.text}</span>
      </div>
    </div>
  );
}
