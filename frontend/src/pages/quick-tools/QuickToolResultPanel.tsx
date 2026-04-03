import { useEffect, useMemo, useState } from "react";
import { extractDownloadPath, getDownloadUrl } from "../../lib/quickToolsApi";
import { CopyableTextBlock } from "../../components/CommandPreviewCard";

type ResultTab = "table" | "raw" | "heatmap" | "ai";

type QuickToolResultPanelProps = {
  title: string;
  resultPayload: Record<string, unknown> | null;
  aiSummary: string;
  error: string;
  heatmapHint?: string;
  tableMode?: "default" | "functional-residue";
  enableHeatmapTab?: boolean;
};

export function QuickToolResultPanel(props: QuickToolResultPanelProps) {
  const [tab, setTab] = useState<ResultTab>("table");
  const [labelFilter, setLabelFilter] = useState<"all" | "0" | "1">("all");
  const [csvRows, setCsvRows] = useState<Record<string, unknown>[]>([]);
  const [csvLoading, setCsvLoading] = useState(false);
  const [csvLoadFailed, setCsvLoadFailed] = useState(false);
  const downloadPath = useMemo(() => extractDownloadPath(props.resultPayload), [props.resultPayload]);
  const tableRows = useMemo(() => extractTableRows(props.resultPayload), [props.resultPayload]);
  const csvPath = useMemo(() => extractCsvPath(props.resultPayload), [props.resultPayload]);
  const useCsvAsSource = Boolean(csvPath);
  const fallbackToJsonRows = useCsvAsSource && csvLoadFailed;
  const effectiveTableRows = useCsvAsSource && !fallbackToJsonRows ? csvRows : tableRows;
  const isFunctionalResidueMode = props.tableMode === "functional-residue";
  const transformedFunctionalRows = useMemo(
    () => (isFunctionalResidueMode ? toFunctionalResidueRows(effectiveTableRows) : []),
    [effectiveTableRows, isFunctionalResidueMode]
  );
  const hasFunctionalResidueRows = transformedFunctionalRows.length > 0;
  const tableRowsForDisplay = hasFunctionalResidueRows ? transformedFunctionalRows : effectiveTableRows;
  const tableColumns = useMemo(
    () => (hasFunctionalResidueRows ? ["position", "amino_acid", "label", "probabilities"] : extractTableColumns(tableRowsForDisplay)),
    [hasFunctionalResidueRows, tableRowsForDisplay]
  );
  const filteredRows = useMemo(() => {
    if (!hasFunctionalResidueRows || labelFilter === "all") return tableRowsForDisplay;
    return tableRowsForDisplay.filter((row) => String(row.label ?? "") === labelFilter);
  }, [hasFunctionalResidueRows, labelFilter, tableRowsForDisplay]);
  const hasHeatmapArtifact = useMemo(() => detectHeatmapArtifact(props.resultPayload), [props.resultPayload]);
  const heatmapPath = useMemo(() => extractHeatmapPath(props.resultPayload), [props.resultPayload]);
  const showHeatmapTab = props.enableHeatmapTab === true;
  const MAX_TABLE_ROWS = 200;
  const visibleRows = filteredRows.slice(0, MAX_TABLE_ROWS);
  const rowsTruncated = filteredRows.length > MAX_TABLE_ROWS;

  function onDownloadTableCsv() {
    if (tableRowsForDisplay.length === 0) return;
    const csvText = rowsToCsv(tableRowsForDisplay);
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
    setLabelFilter("all");
  }, [props.resultPayload, props.tableMode]);

  useEffect(() => {
    if (!showHeatmapTab && tab === "heatmap") {
      setTab("table");
    }
  }, [showHeatmapTab, tab]);

  useEffect(() => {
    let cancelled = false;

    if (!csvPath) {
      setCsvRows([]);
      setCsvLoading(false);
      setCsvLoadFailed(false);
      return () => {
        cancelled = true;
      };
    }

    void (async () => {
      try {
        setCsvLoading(true);
        setCsvLoadFailed(false);
        const res = await fetch(getDownloadUrl(csvPath));
        if (!res.ok) throw new Error(`Failed to fetch csv (${res.status})`);
        const text = await res.text();
        const parsedRows = parseCsvToRows(text);
        if (!cancelled) {
          setCsvRows(parsedRows);
          setCsvLoading(false);
        }
      } catch {
        if (!cancelled) {
          setCsvRows([]);
          setCsvLoading(false);
          setCsvLoadFailed(true);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [csvPath]);

  return (
    <div className="quick-tools-v2-result-wrap">
      <div className="report-result-header quick-tools-v2-result-header">
        <h3>{props.title}</h3>
        <div className="report-downloads">
          {downloadPath && (
            <a className="chat-header-link" href={getDownloadUrl(downloadPath)} target="_blank" rel="noreferrer">
              Download Result
            </a>
          )}
          <button
            type="button"
            className="custom-btn-secondary"
            onClick={onDownloadTableCsv}
            disabled={tableRowsForDisplay.length === 0}
          >
            Download Table CSV
          </button>
        </div>
      </div>

      <div className="quick-tools-v2-result-tabs">
        <button type="button" className={tab === "table" ? "active" : ""} onClick={() => setTab("table")}>
          Table
        </button>
        <button type="button" className={tab === "raw" ? "active" : ""} onClick={() => setTab("raw")}>
          Raw
        </button>
        {showHeatmapTab && (
          <button type="button" className={tab === "heatmap" ? "active" : ""} onClick={() => setTab("heatmap")}>
            Heatmap
          </button>
        )}
        <button type="button" className={tab === "ai" ? "active" : ""} onClick={() => setTab("ai")}>
          AI Expert
        </button>
      </div>

      {props.error && <div className="error-box">{props.error}</div>}

      {tab === "table" && (
        <div className="report-text quick-tools-v2-text quick-tools-v2-table-view">
          {useCsvAsSource && csvLoading && <div className="quick-tools-v2-empty-note">Loading CSV result...</div>}
          {useCsvAsSource && !csvLoading && csvLoadFailed && (
            <div className="quick-tools-v2-empty-note">
              Failed to load CSV result. Fallback to JSON preview when available.
            </div>
          )}
          {hasFunctionalResidueRows && (
            <div className="quick-tools-v2-table-filter">
              <span>Label Filter</span>
              <select value={labelFilter} onChange={(e) => setLabelFilter(e.target.value as "all" | "0" | "1")}>
                <option value="all">All</option>
                <option value="0">0</option>
                <option value="1">1</option>
              </select>
            </div>
          )}
          {tableColumns.length > 0 && visibleRows.length > 0 ? (
            <>
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
                    {visibleRows.map((row, idx) => (
                      <tr key={`${idx}-${String(row[tableColumns[0]])}`}>
                        {tableColumns.map((column) => (
                          <td key={`${idx}-${column}`}>{formatCellValue(row[column])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {rowsTruncated && (
                <div className="quick-tools-v2-table-note">
                  Showing first {MAX_TABLE_ROWS} rows out of {filteredRows.length}. Download full result for complete data.
                </div>
              )}
            </>
          ) : !useCsvAsSource && !csvLoading ? (
            <div className="quick-tools-v2-empty-note">No tabular rows available for current result.</div>
          ) : null}
        </div>
      )}

      {tab === "raw" && (
        <CopyableTextBlock
          text={props.resultPayload ? JSON.stringify(props.resultPayload, null, 2) : ""}
          emptyText="Result JSON will appear here..."
          wrapperClassName="quick-tools-v2-copy-wrap"
          preClassName="report-text quick-tools-v2-text"
          ariaLabel="Copy raw result"
        />
      )}

      {showHeatmapTab && tab === "heatmap" && (
        <div className="report-text quick-tools-v2-text quick-tools-v2-heatmap-view">
          {heatmapPath ? (
            <iframe
              className="quick-tools-v2-heatmap-frame"
              src={getDownloadUrl(heatmapPath, true)}
              title="Quick Tools Heatmap"
            />
          ) : (
            <CopyableTextBlock
              text={
                hasHeatmapArtifact
                  ? props.heatmapHint ||
                    "Heatmap artifacts detected in current result. Use Download Result for full interactive plot or file output."
                  : "No heatmap artifact detected in this result. Heatmap is typically available for mutation and residue-related tasks."
              }
              wrapperClassName="quick-tools-v2-copy-wrap"
              preClassName="report-text quick-tools-v2-text quick-tools-v2-heatmap-note"
              ariaLabel="Copy heatmap note"
            />
          )}
        </div>
      )}

      {tab === "ai" && (
        <CopyableTextBlock
          text={props.aiSummary}
          emptyText="Enable AI Summary and run prediction to generate expert analysis."
          wrapperClassName="quick-tools-v2-copy-wrap"
          preClassName="report-text quick-tools-v2-text"
          ariaLabel="Copy AI summary"
        />
      )}
    </div>
  );
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value != null && !Array.isArray(value);
}

function formatCellValue(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function extractTableColumns(rows: Record<string, unknown>[]): string[] {
  const ordered = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      ordered.add(key);
    }
  }
  return Array.from(ordered);
}

function toRowArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.map((item, idx) => {
    if (isPlainRecord(item)) return normalizeSequenceHeaderRow(item);
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

function toKeyValueRows(record: Record<string, unknown>): Record<string, unknown>[] {
  return Object.entries(record).map(([key, value]) => ({
    field: key,
    value: formatCellValue(value)
  }));
}

function extractTableRows(payload: Record<string, unknown> | null): Record<string, unknown>[] {
  if (!payload) return [];

  const data = isPlainRecord(payload.data) ? payload.data : null;
  const candidates: unknown[] = [
    data?.rows,
    payload.rows,
    data?.table,
    payload.table,
    data?.results,
    payload.results,
    data,
    payload
  ];

  for (const candidate of candidates) {
    const rows = toRowArray(candidate);
    if (rows.length > 0) return rows;
  }

  for (const candidate of candidates) {
    if (isPlainRecord(candidate)) {
      const nestedArray = Object.values(candidate).find((entry) => Array.isArray(entry));
      const rows = toRowArray(nestedArray);
      if (rows.length > 0) return rows;
      const keyValueRows = toKeyValueRows(candidate);
      if (keyValueRows.length > 0) return keyValueRows;
    }
  }

  return [];
}

function detectHeatmapArtifact(payload: Record<string, unknown> | null): boolean {
  if (!payload) return false;
  const data = isPlainRecord(payload.data) ? payload.data : null;
  const candidates: unknown[] = [data?.heatmap_path, payload.heatmap_path, data, payload];
  return candidates.some((candidate) => {
    if (typeof candidate === "string") {
      return candidate.toLowerCase().includes("heatmap");
    }
    if (!isPlainRecord(candidate)) return false;
    return Object.entries(candidate).some(([key, value]) => {
      if (key.toLowerCase().includes("heatmap")) return true;
      if (typeof value === "string" && value.toLowerCase().includes("heatmap")) return true;
      return false;
    });
  });
}

function extractHeatmapPath(payload: Record<string, unknown> | null): string {
  if (!payload) return "";
  const data = isPlainRecord(payload.data) ? payload.data : null;
  const fileInfo = isPlainRecord(payload.file_info) ? payload.file_info : null;

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
    if (!isPlainRecord(entry)) continue;
    for (const [key, value] of Object.entries(entry)) {
      if (!key.toLowerCase().includes("heatmap")) continue;
      if (typeof value === "string" && value.trim()) return value;
    }
  }

  return "";
}

function extractCsvPath(payload: Record<string, unknown> | null): string {
  if (!payload) return "";
  const data = isPlainRecord(payload.data) ? payload.data : null;
  const fileInfo = isPlainRecord(payload.file_info) ? payload.file_info : null;
  const directCandidates: unknown[] = [
    data?.csv_path,
    payload.csv_path,
    data?.file_path,
    payload.file_path,
    fileInfo?.csv_path,
    fileInfo?.file_path
  ];

  for (const candidate of directCandidates) {
    if (typeof candidate === "string" && looksLikeCsvPath(candidate)) return candidate;
  }

  return findFirstCsvPath([data, fileInfo, payload]);
}

function looksLikeCsvPath(path: string): boolean {
  return path.trim().toLowerCase().endsWith(".csv");
}

function findFirstCsvPath(value: unknown, visited = new Set<unknown>()): string {
  if (typeof value === "string") {
    return looksLikeCsvPath(value) ? value : "";
  }
  if (value == null || visited.has(value)) return "";

  if (Array.isArray(value)) {
    visited.add(value);
    for (const entry of value) {
      const found = findFirstCsvPath(entry, visited);
      if (found) return found;
    }
    return "";
  }

  if (!isPlainRecord(value)) return "";
  visited.add(value);
  for (const entry of Object.values(value)) {
    const found = findFirstCsvPath(entry, visited);
    if (found) return found;
  }
  return "";
}

function parseCsvToRows(csvText: string): Record<string, unknown>[] {
  const text = csvText.replace(/^\uFEFF/, "");
  const lines = splitCsvLines(text).filter((line) => line.trim().length > 0);
  if (lines.length === 0) return [];

  const headers = parseCsvLine(lines[0]);
  if (headers.length === 0) return [];

  const rows: Record<string, unknown>[] = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cells = parseCsvLine(lines[i]);
    const row: Record<string, unknown> = {};
    headers.forEach((header, idx) => {
      row[header || `column_${idx + 1}`] = cells[idx] ?? "";
    });
    rows.push(normalizeSequenceHeaderRow(row));
  }
  return rows;
}

function splitCsvLines(text: string): string[] {
  const lines: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "\"") {
      if (inQuotes && text[i + 1] === "\"") {
        current += "\"\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
        current += "\"";
      }
      continue;
    }
    if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (current.length > 0) {
        lines.push(current);
        current = "";
      }
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      continue;
    }
    current += ch;
  }
  if (current.length > 0) lines.push(current);
  return lines;
}

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === "\"") {
      if (inQuotes && line[i + 1] === "\"") {
        current += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  out.push(current);
  return out;
}

function toFunctionalResidueRows(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  const expanded: Record<string, unknown>[] = [];
  for (const row of rows) {
    const sequence = pickString(row, ["sequence", "seq"]);
    if (!sequence) continue;
    const labels = extractResidueLabels(row);
    const probabilities = extractResidueProbabilities(row);
    const residueCount = Math.max(sequence.length, labels.length, probabilities.length);
    for (let i = 0; i < residueCount; i += 1) {
      expanded.push({
        position: i + 1,
        amino_acid: sequence[i] || "",
        label: labels[i] ?? "",
        probabilities: formatResidueProbabilities(probabilities[i])
      });
    }
  }
  return expanded;
}

function pickString(row: Record<string, unknown>, keys: string[]): string {
  for (const key of keys) {
    const value = row[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function parseMaybeJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (!text || (!text.startsWith("[") && !text.startsWith("{"))) return value;
  try {
    return JSON.parse(text);
  } catch {
    return value;
  }
}

function unwrapNestedArray(value: unknown): unknown {
  let current = parseMaybeJson(value);
  while (Array.isArray(current) && current.length === 1 && Array.isArray(current[0])) {
    current = current[0];
  }
  return current;
}

function extractResidueLabels(row: Record<string, unknown>): string[] {
  const raw = unwrapNestedArray(row.predicted_class ?? row.label ?? row.labels);
  if (!Array.isArray(raw)) return [];
  return raw.map((item) => {
    const num = Number(item);
    if (!Number.isFinite(num)) return "";
    return num >= 0.5 ? "1" : "0";
  });
}

function extractResidueProbabilities(row: Record<string, unknown>): unknown[] {
  const raw = unwrapNestedArray(row.probabilities ?? row.probability ?? row.probs);
  if (!Array.isArray(raw)) return [];
  return raw;
}

function formatResidueProbabilities(value: unknown): string {
  const parsed = parseMaybeJson(value);
  if (Array.isArray(parsed)) {
    const nums = parsed.map((item) => Number(item)).filter((num) => Number.isFinite(num));
    if (nums.length >= 2) {
      return `0:${nums[0].toFixed(4)} | 1:${nums[1].toFixed(4)}`;
    }
    if (nums.length === 1) {
      return nums[0].toFixed(4);
    }
    return JSON.stringify(parsed);
  }
  const num = Number(parsed);
  if (Number.isFinite(num)) return num.toFixed(4);
  if (parsed == null) return "";
  return String(parsed);
}

function rowsToCsv(rows: Record<string, unknown>[]): string {
  const columns = extractTableColumns(rows);
  if (columns.length === 0) return "";
  const header = columns.map(escapeCsvCell).join(",");
  const body = rows.map((row) => columns.map((col) => escapeCsvCell(formatCellValue(row[col]))).join(",")).join("\n");
  return `${header}\n${body}\n`;
}

function escapeCsvCell(value: string): string {
  const normalized = value ?? "";
  if (/[",\n\r]/.test(normalized)) {
    return `"${normalized.replace(/"/g, "\"\"")}"`;
  }
  return normalized;
}

