import { useId, useState } from "react";
import type { AgentContextInfo, KimiContextInfo } from "../lib/api";
import { contextUsage } from "../lib/contextUsage";
import "./ContextUsageRing.css";

type Props = { context?: AgentContextInfo | null; nativeContext?: KimiContextInfo | null; lang: string };

export function ContextUsageRing({ context, nativeContext, lang }: Props) {
  const tooltipId = useId();
  const [dismissed, setDismissed] = useState(false);
  const usage = contextUsage(context, nativeContext);
  const zh = lang === "zh";
  const label = zh ? "上下文用量" : "Context usage";
  const amount = usage?.used !== undefined && usage.limit !== undefined
    ? `${usage.used.toLocaleString()} / ${usage.limit.toLocaleString()} tokens` : "";
  const note = !usage ? (zh ? "暂无用量数据" : "No usage data yet")
    : usage.estimated ? (zh ? "最近请求估算" : "Last request · estimated") : "";
  const compacted = usage?.compacted ? (zh ? "已自动压缩" : "Auto-compacted") : "";
  const detail = [usage ? `${usage.percent}%` : "", amount, note, compacted].filter(Boolean).join(" · ");
  const tone = !usage ? " is-unknown" : usage.percent >= 90 ? " is-high" : usage.percent >= 70 ? " is-warn" : "";
  return <span className={`context-usage${tone}${dismissed ? " is-dismissed" : ""}`}
    onMouseEnter={() => setDismissed(false)} onFocus={() => setDismissed(false)}
    onKeyDown={e => { if (e.key === "Escape") setDismissed(true); }}>
    <span className="context-usage-meter" role={usage ? "meter" : "img"} tabIndex={0} aria-label={label}
      aria-valuemin={usage ? 0 : undefined} aria-valuemax={usage ? 100 : undefined} aria-valuenow={usage?.percent}
      aria-valuetext={detail} aria-describedby={tooltipId}>
      <svg viewBox="0 0 28 28" aria-hidden="true">
        <circle className="context-usage-track" cx="14" cy="14" r="11"/>
        <circle className="context-usage-fill" cx="14" cy="14" r="11" pathLength="100"
          strokeDasharray={`${usage?.percent ?? 0} 100`} transform="rotate(-90 14 14)"/>
      </svg>
      <span className="context-usage-number" aria-hidden="true">{usage ? usage.percent : "–"}</span>
    </span>
    <span id={tooltipId} role="tooltip" className="context-usage-tooltip">
      <strong>{label}{usage ? ` · ${usage.percent}%` : ""}</strong>
      {amount && <span>{amount}</span>}
      {note && <small>{note}</small>}
      {compacted && <small>{compacted}</small>}
    </span>
  </span>;
}
