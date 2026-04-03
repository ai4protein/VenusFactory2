import { useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { streamSSEFromPost } from "../lib/sse";
import { SegmentedSwitch } from "../components/SegmentedSwitch";
import { PageFooter } from "../components/PageFooter";

type ParsedPayload = {
  sequence_map: Record<string, string>;
  selected_chain: string;
  preview: string;
  current_file: string;
  original_content: string;
};

const ANALYSIS_OPTIONS = [
  { key: "mutation", label: "🧬 Mutation" },
  { key: "function", label: "🔬 Function" },
  { key: "residue", label: "🎯 Residue" },
  { key: "properties", label: "⚗️ Properties" }
];

function renderMarkdown(text: string) {
  const html = marked.parse(text || "", { async: false }) as string;
  return DOMPurify.sanitize(html);
}

export function ReportPage() {
  const [activePhase, setActivePhase] = useState<"idle" | "example" | "upload" | "parse" | "generate">("idle");
  const [inputMode, setInputMode] = useState<"paste" | "upload">("upload");
  const [pasteContent, setPasteContent] = useState("");
  const [uploadedPath, setUploadedPath] = useState("");
  const [parsed, setParsed] = useState<ParsedPayload | null>(null);
  const [selectedChain, setSelectedChain] = useState("Sequence 1");
  const [selectedAnalyses, setSelectedAnalyses] = useState<string[]>([]);
  const [reportText, setReportText] = useState("");
  const [aiReport, setAiReport] = useState("");
  const [htmlUrl, setHtmlUrl] = useState("");
  const [pdfUrl, setPdfUrl] = useState("");
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("Idle");
  const [streamLogs, setStreamLogs] = useState<string[]>([]);
  const [error, setError] = useState("");

  const isExampleLoading = activePhase === "example";
  const isUploadLoading = activePhase === "upload";
  const isParseLoading = activePhase === "parse";
  const isGenerateLoading = activePhase === "generate";
  const hasActiveTask = activePhase !== "idle";

  async function loadDefaultExample() {
    setError("");
    setActivePhase("example");
    try {
      const res = await fetch("/api/report/default-input");
      if (!res.ok) {
        throw new Error(`Load default example failed (${res.status})`);
      }
      const data = (await res.json()) as {
        name: string;
        content: string;
        parse: ParsedPayload;
      };
      setInputMode("paste");
      setUploadedPath("");
      setPasteContent(data.content || "");
      setParsed(data.parse);
      setSelectedChain(data.parse.selected_chain);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load default example.");
    } finally {
      setActivePhase("idle");
    }
  }

  async function parseInput() {
    setError("");
    setActivePhase("parse");
    try {
      const body =
        inputMode === "upload"
          ? { file_path: uploadedPath, content: "" }
          : { content: pasteContent, file_path: "" };
      const res = await fetch("/api/report/parse-input", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!res.ok) {
        throw new Error(`Parse failed (${res.status})`);
      }
      const data = (await res.json()) as ParsedPayload;
      setParsed(data);
      setSelectedChain(data.selected_chain);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to parse input.");
    } finally {
      setActivePhase("idle");
    }
  }

  async function onUploadFile(file: File | null) {
    if (!file) return;
    setError("");
    setActivePhase("upload");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/report/upload", {
        method: "POST",
        body: form
      });
      if (!res.ok) {
        throw new Error(`Upload failed (${res.status})`);
      }
      const data = (await res.json()) as { file_path: string; parse: ParsedPayload };
      setUploadedPath(data.file_path);
      setParsed(data.parse);
      setSelectedChain(data.parse.selected_chain);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload file.");
    } finally {
      setActivePhase("idle");
    }
  }

  async function generateReport() {
    if (!parsed) {
      setError("Please parse input first.");
      return;
    }
    if (!selectedAnalyses.length) {
      setError("Please select at least one analysis type.");
      return;
    }
    setError("");
    setActivePhase("generate");
    setProgress(0);
    setProgressMessage("Starting...");
    setStreamLogs([]);
    setReportText("");
    setAiReport("");
    setHtmlUrl("");
    setPdfUrl("");
    try {
      await streamSSEFromPost(
        "/api/report/generate/stream",
        {
          sequence_map: parsed.sequence_map,
          selected_chain: selectedChain,
          current_file: parsed.current_file,
          original_content: parsed.original_content,
          selected_analyses: selectedAnalyses
        },
        ({ event, data }) => {
          if (!data) return;
          if (event === "progress") {
            const payload = JSON.parse(data) as { progress: number; message: string };
            setProgress(Math.max(0, Math.min(1, payload.progress || 0)));
            setProgressMessage(payload.message || "Running...");
            return;
          }
          if (event === "log") {
            const payload = JSON.parse(data) as { line: string };
            if (payload.line) {
              setStreamLogs((prev) => [...prev, payload.line]);
            }
            return;
          }
          if (event === "error") {
            const payload = JSON.parse(data) as { message: string };
            setError(payload.message || "Stream error.");
            return;
          }
          if (event === "done") {
            const payload = JSON.parse(data) as {
              success: boolean;
              message?: string;
              report_text?: string;
              ai_report?: string;
              html_download_url?: string;
              pdf_download_url?: string;
            };
            if (!payload.success) {
              setError(payload.message || "Failed to generate report.");
              setProgressMessage("Failed");
              return;
            }
            setReportText(payload.report_text || "");
            setAiReport(payload.ai_report || "");
            setHtmlUrl(payload.html_download_url || "");
            setPdfUrl(payload.pdf_download_url || "");
            setProgress(1);
            setProgressMessage("Completed");
          }
        }
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate report.");
    } finally {
      setActivePhase("idle");
    }
  }

  return (
    <div className="report-page">
      <header className="chat-header report-header">
        <div className="report-header-main">
          <h2>Report</h2>
          <div className="chat-header-subrow">
            <p>One-click integrated protein analysis with AI-enhanced narrative report.</p>
          </div>
        </div>
        <div className={`report-status-pill ${hasActiveTask ? "running" : "idle"}`}>
          <span className="report-status-dot" />
          {isGenerateLoading ? "Generating Report" : hasActiveTask ? "Processing Input" : "Ready"}
        </div>
      </header>

      <section className="report-grid">
        <aside className="chat-panel left report-control-panel">
          <h3 className="report-block-title">Input</h3>
          <button
            type="button"
            className="report-btn report-btn-soft"
            onClick={() => void loadDefaultExample()}
            disabled={hasActiveTask}
          >
            {isExampleLoading ? "Loading Example..." : "Use Default Example"}
          </button>
          <div className="report-mode-row">
            <SegmentedSwitch
              value={inputMode}
              onChange={setInputMode}
              ariaLabel="Report input mode"
              className="report-segment-switch"
              options={[
                { value: "paste", label: "Paste" },
                { value: "upload", label: "Upload" }
              ]}
            />
          </div>

          {inputMode === "paste" ? (
            <textarea
              className="report-input-textarea"
              rows={10}
              value={pasteContent}
              onChange={(e) => setPasteContent(e.target.value)}
              placeholder="Paste FASTA/PDB/raw sequence..."
            />
          ) : (
            <input
              className="report-file-input"
              type="file"
              accept=".fasta,.fa,.pdb"
              onChange={(e) => void onUploadFile(e.target.files?.[0] || null)}
            />
          )}

          <button
            type="button"
            className="report-btn report-btn-soft"
            onClick={() => void parseInput()}
            disabled={hasActiveTask}
          >
            {isParseLoading ? "Detecting..." : "Detect Sequence"}
          </button>

          {parsed && (
            <>
              <label className="left-controls">
                Chain/Sequence
                <select value={selectedChain} onChange={(e) => setSelectedChain(e.target.value)}>
                  {Object.keys(parsed.sequence_map).map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </label>
              <div className="report-preview">Preview: {parsed.preview}</div>
            </>
          )}

          <h3 className="report-block-title">Analysis Types</h3>
          <div className="report-options">
            {ANALYSIS_OPTIONS.map((opt) => (
              <label key={opt.key} className="report-option-item">
                <input
                  type="checkbox"
                  checked={selectedAnalyses.includes(opt.key)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedAnalyses((prev) => [...prev, opt.key]);
                    } else {
                      setSelectedAnalyses((prev) => prev.filter((x) => x !== opt.key));
                    }
                  }}
                />
                <span>{opt.label}</span>
              </label>
            ))}
          </div>

          <button
            type="button"
            className="report-btn report-btn-primary"
            onClick={() => void generateReport()}
            disabled={hasActiveTask}
          >
            {isGenerateLoading ? "Generating..." : "Generate Report"}
          </button>
          <div className="report-progress-wrap">
            <div className="report-progress-text">
              <span>{progressMessage}</span>
              <strong>{Math.round(progress * 100)}%</strong>
            </div>
            <div className="report-progress-track">
              <div
                className="report-progress-bar"
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            </div>
          </div>

          {error && <div className="error-box">{error}</div>}
        </aside>

        <section className="chat-panel report-output report-main-panel">
          <div className="report-result-header">
            <h3>AI Expert Analysis</h3>
            <div className="report-downloads">
              {htmlUrl && (
                <a className="chat-header-link" href={htmlUrl} target="_blank" rel="noreferrer">
                  Download HTML
                </a>
              )}
              {pdfUrl && (
                <a className="chat-header-link" href={pdfUrl} target="_blank" rel="noreferrer">
                  Download PDF
                </a>
              )}
            </div>
          </div>
          {aiReport ? (
            <div
              className="report-text report-ai-box report-markdown"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(aiReport) }}
            />
          ) : (
            <div className="report-copy-wrap">
              <pre className="report-text report-ai-box">AI analysis will appear here...</pre>
            </div>
          )}
        </section>

        <aside className="chat-panel right report-right-panel">
          <section className="report-side-card">
            <h3>Streaming Logs</h3>
            <div className="report-copy-wrap report-log-copy-wrap">
              <pre className="report-log-box">{streamLogs.length ? streamLogs.join("\n") : "Task logs will stream here..."}</pre>
            </div>
          </section>
        </aside>
      </section>
      <PageFooter />
    </div>
  );
}
