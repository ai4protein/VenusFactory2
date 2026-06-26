import { useMemo, useState } from "react";
import { MolstarViewer } from "./MolstarViewer";
import {
  extractMediaFromValue,
  fileUrlFor,
  isRemoteUrl,
  parseToolOutput,
  shortToolName,
  type MediaRef,
} from "../lib/richContent";

/**
 * Renders one kimi tool invocation as a card with three visual states:
 *   - running:  spinner + elapsed seconds
 *   - ok:       check + duration, click to expand args+output
 *   - error:    red border + error preview
 *
 * When expanded:
 *   - Input args rendered as pretty JSON
 *   - Output recognizes embedded file paths:
 *       .pdb/.cif → MolstarViewer
 *       .png/.jpg/.svg → inline <img>
 *       .html → sandboxed <iframe>
 *       .fasta/.csv → text preview with a "open" link
 *     plus the raw JSON fallback below.
 */
export interface ToolExecution {
  tool_call_id?: string;
  name?: string;
  tool_name?: string;
  args?: unknown;
  inputs?: unknown;
  output?: unknown;
  outputs?: unknown;
  status?: string;
  is_error?: boolean;
  started_at?: string;
  finished_at?: string;
  timestamp?: string;
  turn_id?: string;
  error?: string;
}

function durationMs(t: ToolExecution): number | null {
  const start = t.started_at || t.timestamp;
  const end = t.finished_at;
  if (!start) return null;
  const startMs = Date.parse(start);
  if (isNaN(startMs)) return null;
  const endMs = end ? Date.parse(end) : Date.now();
  if (isNaN(endMs)) return null;
  return endMs - startMs;
}

function formatDuration(ms: number | null): string {
  if (ms == null) return "";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

function prettyJson(v: unknown, maxLen = 4000): string {
  if (v == null) return "";
  if (typeof v === "string") {
    return v.length > maxLen ? v.slice(0, maxLen) + `\n…(${v.length - maxLen} more chars)` : v;
  }
  try {
    const s = JSON.stringify(v, null, 2);
    return s.length > maxLen ? s.slice(0, maxLen) + `\n…(${s.length - maxLen} more chars)` : s;
  } catch {
    return String(v);
  }
}

function MediaPreview({ m }: { m: MediaRef }) {
  const url = fileUrlFor(m.path);
  const remote = isRemoteUrl(m.path);
  const basename = (m.path.split("?")[0].split("/").pop() || m.path);
  switch (m.kind) {
    case "structure":
      // MolstarViewer fetches via /api/structure/content which only handles
      // local files. Remote PDB/CIF URLs render as a link instead — the user
      // can click through and kimi can `download_rcsb_structure` to make it
      // local if they need 3D rendering.
      if (remote) {
        return (
          <div className="tool-media tool-media-link">
            <span className="tool-media-icon">🧪</span>
            <a href={url} target="_blank" rel="noopener noreferrer">{basename} (remote structure)</a>
          </div>
        );
      }
      return <MolstarViewer filePath={m.path} label={basename} />;
    case "image":
      return (
        <figure className="tool-media tool-media-image">
          <img src={url} alt={basename} loading="lazy" />
          <figcaption>{basename}</figcaption>
        </figure>
      );
    case "html":
      // Remote HTML — open in new tab; embedding remote content via iframe
      // is more risk than reward (some embed-blocking, X-Frame-Options).
      if (remote) {
        return (
          <div className="tool-media tool-media-link">
            <span className="tool-media-icon">📄</span>
            <a href={url} target="_blank" rel="noopener noreferrer">{basename} ↗</a>
          </div>
        );
      }
      return (
        <figure className="tool-media tool-media-html">
          <iframe
            src={url}
            title={basename}
            sandbox=""                /* deny scripts/forms/popups */
            referrerPolicy="no-referrer"
          />
          <figcaption>
            <a href={url} target="_blank" rel="noopener noreferrer">{basename} ↗</a>
          </figcaption>
        </figure>
      );
    case "csv":
    case "fasta":
      return (
        <div className="tool-media tool-media-link">
          <span className="tool-media-icon">{m.kind === "csv" ? "📊" : "🧬"}</span>
          <a href={url} target="_blank" rel="noopener noreferrer">{basename}</a>
        </div>
      );
    default:
      return (
        <div className="tool-media tool-media-link">
          <span className="tool-media-icon">📎</span>
          <a href={url} target="_blank" rel="noopener noreferrer">{basename}</a>
        </div>
      );
  }
}

/**
 * Render a list of tool executions, keeping the chat scannable when a turn
 * calls 10+ tools. Pattern:
 *   - any tool with status="running"   → always shown expanded (in-flight)
 *   - last `recentCount` completed     → shown expanded
 *   - everything older                 → tucked into a collapsible
 *                                        "▸ N earlier tool calls" summary
 */
export function ToolExecutionList({
  tools,
  recentCount = 3,
}: {
  tools: ToolExecution[];
  recentCount?: number;
}) {
  if (tools.length === 0) return null;

  // Split: in-flight always visible; from the rest keep the last `recentCount`.
  const running: ToolExecution[] = [];
  const completed: ToolExecution[] = [];
  for (const t of tools) {
    const st = String(t.status || "").toLowerCase();
    if (st === "running" || (st === "" && !t.finished_at)) running.push(t);
    else completed.push(t);
  }
  const visibleCompleted = completed.slice(-recentCount);
  const hidden = completed.slice(0, Math.max(0, completed.length - recentCount));

  return (
    <>
      {hidden.length > 0 && (
        <details className="tool-history-fold">
          <summary>
            <span className="tool-history-icon">▸</span>
            <span className="tool-history-label">
              {hidden.length} earlier tool {hidden.length === 1 ? "call" : "calls"}
            </span>
            <span className="tool-history-summary">
              {(() => {
                let ok = 0, err = 0;
                for (const t of hidden) {
                  if (String(t.status || "").toLowerCase() === "error" || t.is_error) err++;
                  else ok++;
                }
                const parts: string[] = [];
                if (ok) parts.push(`${ok} ✓`);
                if (err) parts.push(`${err} ✕`);
                return parts.join(" · ");
              })()}
            </span>
          </summary>
          <div className="tool-history-body">
            {hidden.map((t, i) => (
              <ToolExecutionCard
                key={String(t.tool_call_id || `hidden-${i}`)}
                t={t}
              />
            ))}
          </div>
        </details>
      )}
      {visibleCompleted.map((t, i) => (
        <ToolExecutionCard key={String(t.tool_call_id || `recent-${i}`)} t={t} />
      ))}
      {running.map((t, i) => (
        <ToolExecutionCard key={String(t.tool_call_id || `running-${i}`)} t={t} />
      ))}
    </>
  );
}


export function ToolExecutionCard({ t }: { t: ToolExecution }) {
  const [open, setOpen] = useState(false);
  const status = String(t.status || "").toLowerCase();
  const isRunning = status === "running" || (status === "" && !t.finished_at);
  const isError = status === "error" || t.is_error === true;
  const isOk = status === "ok" || (!isRunning && !isError && Boolean(t.finished_at));

  const args = t.args ?? t.inputs;
  const rawOutput = t.output ?? t.outputs;
  const dur = formatDuration(durationMs(t));
  const name = shortToolName(String(t.tool_name || t.name || "tool"));

  // Parse output JSON-string once, then scan for media references.
  const parsedOutput = useMemo(() => parseToolOutput(rawOutput), [rawOutput]);
  const media = useMemo(
    () => (parsedOutput != null ? extractMediaFromValue(parsedOutput, 12) : []),
    [parsedOutput]
  );

  // Media on a successful tool surfaces in the COLLAPSED card too — that's
  // the whole point of "see the structure without clicking". For failed/
  // running ones we keep the chrome minimal until expanded.
  const showMediaCollapsed = !isRunning && !isError && media.length > 0;

  const klass = [
    "tool-card",
    isRunning && "tool-card-running",
    isOk && "tool-card-ok",
    isError && "tool-card-error",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={klass}>
      <button
        type="button"
        className="tool-card-header"
        onClick={() => !isRunning && setOpen((v) => !v)}
        disabled={isRunning}
        title={isRunning ? "" : open ? "Hide details" : "Show details"}
      >
        <span className="tool-card-icon">
          {isRunning ? <span className="chat-tool-spinner" /> : isError ? "✕" : "✓"}
        </span>
        <code className="tool-card-name">{name}</code>
        {showMediaCollapsed && (
          <span className="tool-card-media-count" title={`${media.length} preview(s)`}>
            {media.length} 📎
          </span>
        )}
        {dur && <span className="tool-card-duration">{dur}</span>}
        {!isRunning && (
          <span className="tool-card-chev" aria-hidden="true">
            {open ? "▾" : "▸"}
          </span>
        )}
      </button>

      {showMediaCollapsed && !open && (
        <div className="tool-card-media-strip">
          {media.slice(0, 3).map((m, i) => (
            <MediaPreview key={`${m.kind}-${i}-${m.path}`} m={m} />
          ))}
          {media.length > 3 && (
            <div className="tool-card-media-more" onClick={() => setOpen(true)}>
              + {media.length - 3} more — click to expand
            </div>
          )}
        </div>
      )}

      {open && !isRunning && (
        <div className="tool-card-body">
          {media.length > 0 && (
            <div className="tool-card-section">
              <div className="tool-card-section-label">Outputs ({media.length})</div>
              <div className="tool-card-media-grid">
                {media.map((m, i) => (
                  <MediaPreview key={`${m.kind}-${i}-${m.path}`} m={m} />
                ))}
              </div>
            </div>
          )}
          {args !== undefined && (
            <div className="tool-card-section">
              <div className="tool-card-section-label">Input</div>
              <pre>
                <code>{prettyJson(args)}</code>
              </pre>
            </div>
          )}
          {parsedOutput !== undefined && (
            <div className="tool-card-section">
              <div className="tool-card-section-label">
                {isError ? "Error" : "Raw output"}
              </div>
              <pre>
                <code>{prettyJson(parsedOutput, 6000)}</code>
              </pre>
            </div>
          )}
          {!args && !parsedOutput && t.error && (
            <div className="tool-card-section">
              <div className="tool-card-section-label">Error</div>
              <pre>
                <code>{prettyJson(t.error)}</code>
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
