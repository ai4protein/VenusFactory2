import { useEffect, useMemo, useState } from "react";
import { extractAdvancedDownloadPath, getAdvancedDownloadUrl } from "../../lib/advancedToolsApi";

type ResultTab = "summary" | "table" | "raw" | "heatmap" | "ai";

type AdvancedResultPanelProps = {
  title: string;
  resultPayload: Record<string, unknown> | null;
  aiSummary?: string;
  error: string;
  showSummaryTab?: boolean;
  enableHeatmapTab?: boolean;
  readonly?: boolean;
};

function toPrettyString(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function ResultTextBlock({ text, emptyText = "" }: { text: string; emptyText?: string }) {
  const content = text || emptyText;
  return (
    <div className="copyable-pre-wrap quick-tools-v2-copy-wrap">
      <pre className="copyable-pre report-text quick-tools-v2-text">{content}</pre>
    </div>
  );
}

export function AdvancedResultPanel(props: AdvancedResultPanelProps) {
  const showSummaryTab = props.showSummaryTab !== false;
  const readonly = props.readonly === true;
  const [tab, setTab] = useState<ResultTab>(showSummaryTab ? "summary" : "table");
  const downloadPath = useMemo(() => extractAdvancedDownloadPath(props.resultPayload), [props.resultPayload]);
  const summary = typeof props.resultPayload?.status === "string" ? props.resultPayload.status : "";
  const tableRecords = useMemo(() => toTableRows(props.resultPayload?.table), [props.resultPayload]);
  const tableColumns = useMemo(() => extractTableColumns(tableRecords), [tableRecords]);
  const heatmapPath = useMemo(() => extractHeatmapPath(props.resultPayload), [props.resultPayload]);
  const showHeatmapTab = props.enableHeatmapTab === true;

  function onDownloadTableCsv() {
    if (readonly) return;
    if (tableRecords.length === 0) return;
    const csvText = rowsToCsv(tableRecords);
    const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
    const href = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = href;
    a.download = "table-result.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(href);
  }

  useEffect(() => {
    if (!showSummaryTab && tab === "summary") {
      setTab("table");
    }
  }, [showSummaryTab, tab]);

  useEffect(() => {
    if (!showHeatmapTab && tab === "heatmap") {
      setTab("table");
    }
  }, [showHeatmapTab, tab]);

  return (
    <div className="quick-tools-v2-result-wrap">
      <div className="report-result-header quick-tools-v2-result-header">
        <h3>{props.title}</h3>
        <div className="report-downloads">
          {downloadPath && (
            <a
              className={`chat-header-link ${readonly ? "custom-link-btn" : ""}`}
              href={getAdvancedDownloadUrl(downloadPath)}
              target="_blank"
              rel="noreferrer"
              onClick={
                readonly
                  ? (event) => {
                      event.preventDefault();
                      event.stopPropagation();
                    }
                  : undefined
              }
              aria-disabled={readonly}
            >
              Download Result
            </a>
          )}
          <button
            type="button"
            className="custom-btn-secondary"
            onClick={onDownloadTableCsv}
            disabled={readonly || tableRecords.length === 0}
          >
            Download Table CSV
          </button>
        </div>
      </div>

      <div className="quick-tools-v2-result-tabs">
        {showSummaryTab && (
          <button
            type="button"
            className={tab === "summary" ? "active" : ""}
            onClick={() => setTab("summary")}
            disabled={readonly}
          >
            Summary
          </button>
        )}
        <button type="button" className={tab === "table" ? "active" : ""} onClick={() => setTab("table")} disabled={readonly}>
          Table
        </button>
        <button type="button" className={tab === "raw" ? "active" : ""} onClick={() => setTab("raw")} disabled={readonly}>
          Raw
        </button>
        {showHeatmapTab && (
          <button
            type="button"
            className={tab === "heatmap" ? "active" : ""}
            onClick={() => setTab("heatmap")}
            disabled={readonly}
          >
            Heatmap
          </button>
        )}
        <button type="button" className={tab === "ai" ? "active" : ""} onClick={() => setTab("ai")} disabled={readonly}>
          AI Expert
        </button>
      </div>

      {props.error && <div className="error-box">{props.error}</div>}

      {showSummaryTab && tab === "summary" && (
        <ResultTextBlock text={summary} emptyText="" />
      )}

      {tab === "table" && (
        <div className="report-text quick-tools-v2-text quick-tools-v2-table-view">
          {tableColumns.length > 0 && tableRecords.length > 0 ? (
            <div className="quick-tools-v2-table-wrap">
              <table className="quick-tools-v2-table">
                <thead>
                  <tr>
                    {tableColumns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRecords.slice(0, 200).map((row, idx) => (
                    <tr key={`${idx}-${String(row[tableColumns[0]])}`}>
                      {tableColumns.map((column) => (
                        <td key={`${idx}-${column}`}>{toCellText(row[column])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="quick-tools-v2-empty-note">No tabular rows available for current result.</div>
          )}
        </div>
      )}

      {tab === "raw" && (
        <ResultTextBlock
          text={props.resultPayload ? toPrettyString(props.resultPayload) : ""}
          emptyText="Raw JSON output will appear here..."
        />
      )}

      {showHeatmapTab && tab === "heatmap" && (
        <div className="report-text quick-tools-v2-text quick-tools-v2-heatmap-view">
          {heatmapPath ? (
            <iframe
              className="quick-tools-v2-heatmap-frame"
              src={getAdvancedDownloadUrl(heatmapPath, true)}
              title="Advanced Tools Heatmap"
            />
          ) : (
            <ResultTextBlock
              text=""
              emptyText="No heatmap artifact found in this result. You can still use Download Result to inspect packaged outputs."
            />
          )}
        </div>
      )}

      {tab === "ai" && (
        <ResultTextBlock
          text={props.aiSummary || ""}
          emptyText="Enable AI analysis and run task to generate expert interpretation."
        />
      )}
    </div>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value != null && !Array.isArray(value);
}

function toTableRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.map((item, idx) => {
    if (isRecord(item)) return normalizeSequenceHeaderRow(item);
    return { index: idx + 1, value: item };
  });
}

function normalizeSequenceHeaderRow(row: Record<string, unknown>): Record<string, unknown> {
  const next: Record<string, unknown> = { ...row };
  const canonical = pickCanonicalProteinName(next);
  if (canonical) {
    next["Protein Name"] = canonical;
  }
  delete next.protein_name;
  delete next.header;
  delete next.sequence_header;
  return next;
}

function pickCanonicalProteinName(row: Record<string, unknown>): string {
  const candidates = [row["Protein Name"], row.protein_name, row.header, row.sequence_header];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function extractTableColumns(rows: Record<string, unknown>[]): string[] {
  const keys = new Set<string>();
  for (const row of rows) {
    Object.keys(row).forEach((key) => keys.add(key));
  }
  return Array.from(keys);
}

function toCellText(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function extractHeatmapPath(payload: Record<string, unknown> | null): string {
  if (!payload) return "";
  const data = isRecord(payload.data) ? payload.data : null;
  const fileInfo = isRecord(payload.file_info) ? payload.file_info : null;

  const directCandidates: unknown[] = [
    data?.heatmap_path,
    payload.heatmap_path,
    data?.plot_path,
    payload.plot_path,
    fileInfo?.heatmap_path
  ];
  for (const candidate of directCandidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate;
  }

  const objectCandidates: unknown[] = [data, payload, fileInfo];
  for (const entry of objectCandidates) {
    if (!isRecord(entry)) continue;
    for (const [key, value] of Object.entries(entry)) {
      if (!key.toLowerCase().includes("heatmap")) continue;
      if (typeof value === "string" && value.trim()) return value;
    }
  }

  return findFirstHeatmapPath([data, fileInfo, payload]);
}

function findFirstHeatmapPath(value: unknown, visited = new Set<unknown>()): string {
  if (typeof value === "string") {
    return looksLikeHeatmapPath(value) ? value : "";
  }
  if (value == null || visited.has(value)) return "";

  if (Array.isArray(value)) {
    visited.add(value);
    for (const entry of value) {
      const found = findFirstHeatmapPath(entry, visited);
      if (found) return found;
    }
    return "";
  }

  if (!isRecord(value)) return "";
  visited.add(value);
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry === "string" && key.toLowerCase().includes("heatmap") && entry.trim()) return entry;
    const found = findFirstHeatmapPath(entry, visited);
    if (found) return found;
  }
  return "";
}

function looksLikeHeatmapPath(path: string): boolean {
  const lower = path.trim().toLowerCase();
  if (!lower) return false;
  return (
    lower.includes("heatmap") ||
    lower.includes("mut_map") ||
    lower.includes("prediction_heatmap")
  );
}

function rowsToCsv(rows: Record<string, unknown>[]): string {
  const columns = extractTableColumns(rows);
  if (columns.length === 0) return "";
  const header = columns.map(escapeCsvCell).join(",");
  const body = rows.map((row) => columns.map((col) => escapeCsvCell(toCellText(row[col]))).join(",")).join("\n");
  return `${header}\n${body}\n`;
}

function escapeCsvCell(value: string): string {
  const normalized = value ?? "";
  if (/[",\n\r]/.test(normalized)) {
    return `"${normalized.replace(/"/g, "\"\"")}"`;
  }
  return normalized;
}
