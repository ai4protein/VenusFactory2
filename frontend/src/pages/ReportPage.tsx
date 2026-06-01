import { useState, useMemo } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { streamSSEFromPost } from "../lib/sse";
import { SegmentedSwitch } from "../components/SegmentedSwitch";
import { PageFooter } from "../components/PageFooter";
import { WorkspaceFilePicker } from "../components/WorkspaceFilePicker";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

type ParsedPayload = {
  sequence_map: Record<string, string>;
  selected_chain: string;
  preview: string;
  current_file: string;
  original_content: string;
};

const STRINGS = {
  en: {
    title: "Report",
    subtitle: "One-click integrated protein analysis with AI-enhanced narrative report.",
    stateGenerating: "Generating Report",
    stateProcessing: "Processing Input",
    stateReady: "Ready",
    input: "Input",
    inputModeAria: "Report input mode",
    paste: "Paste",
    upload: "Upload",
    pastePlaceholder: "Paste FASTA/PDB/raw sequence...",
    fromWorkspace: "From Workspace",
    loadingExample: "Loading Example...",
    useExample: "Use Default Example",
    chain: "Chain/Sequence",
    preview: "Preview:",
    analysisTypes: "Analysis Types",
    optMutation: "🧬 Mutation",
    optFunction: "🔬 Function",
    optResidue: "🎯 Residue",
    optProperties: "⚗️ Properties",
    generating: "Generating...",
    generateBtn: "Generate Report",
    aiAnalysis: "AI Expert Analysis",
    downloadHtml: "Download HTML",
    downloadPdf: "Download PDF",
    aiPlaceholder: "AI analysis will appear here...",
    streamingLogs: "Streaming Logs",
    logsPlaceholder: "Task logs will stream here...",
    idle: "Idle",
    starting: "Starting...",
    running: "Running...",
    completed: "Completed",
    failed: "Failed",
    errParseFirst: "Please parse input first.",
    errSelectAnalysis: "Please select at least one analysis type.",
    errLoadExample: "Failed to load default example.",
    errUpload: "Failed to upload file.",
    errStream: "Stream error.",
    errGenerate: "Failed to generate report.",
    errLoadStatus: "Load default example failed",
    errUploadStatus: "Upload failed"
  },
  zh: {
    title: "报告",
    subtitle: "一键完成蛋白综合分析，并生成 AI 增强的叙述性报告。",
    stateGenerating: "正在生成报告",
    stateProcessing: "处理输入中",
    stateReady: "就绪",
    input: "输入",
    inputModeAria: "报告输入方式",
    paste: "粘贴",
    upload: "上传",
    pastePlaceholder: "粘贴 FASTA / PDB / 原始序列…",
    fromWorkspace: "从工作区",
    loadingExample: "加载示例中…",
    useExample: "使用默认示例",
    chain: "链 / 序列",
    preview: "预览：",
    analysisTypes: "分析类型",
    optMutation: "🧬 突变",
    optFunction: "🔬 功能",
    optResidue: "🎯 残基",
    optProperties: "⚗️ 性质",
    generating: "生成中…",
    generateBtn: "生成报告",
    aiAnalysis: "AI 专家分析",
    downloadHtml: "下载 HTML",
    downloadPdf: "下载 PDF",
    aiPlaceholder: "AI 分析结果将显示在此处…",
    streamingLogs: "流式日志",
    logsPlaceholder: "任务日志将在此处流式输出…",
    idle: "空闲",
    starting: "启动中…",
    running: "运行中…",
    completed: "已完成",
    failed: "失败",
    errParseFirst: "请先解析输入。",
    errSelectAnalysis: "请至少选择一种分析类型。",
    errLoadExample: "加载默认示例失败。",
    errUpload: "上传文件失败。",
    errStream: "流式传输错误。",
    errGenerate: "生成报告失败。",
    errLoadStatus: "加载默认示例失败",
    errUploadStatus: "上传失败"
  }
};

function renderMarkdown(text: string) {
  const html = marked.parse(text || "", { async: false }) as string;
  return DOMPurify.sanitize(html);
}

type ReportPageProps = {
  workspaceEnabled: boolean;
};

export function ReportPage({ workspaceEnabled }: ReportPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
  const analysisOptions = useMemo(
    () => [
      { key: "mutation", label: t.optMutation },
      { key: "function", label: t.optFunction },
      { key: "residue", label: t.optResidue },
      { key: "properties", label: t.optProperties }
    ],
    [t]
  );
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
  const [progressMessage, setProgressMessage] = useState("");
  const [streamLogs, setStreamLogs] = useState<string[]>([]);
  const [error, setError] = useState("");

  const isExampleLoading = activePhase === "example";
  const isUploadLoading = activePhase === "upload";
  const isGenerateLoading = activePhase === "generate";
  const hasActiveTask = activePhase !== "idle";

  async function loadDefaultExample() {
    setError("");
    setActivePhase("example");
    try {
      const res = await fetch("/api/report/default-input");
      if (!res.ok) {
        throw new Error(`${t.errLoadStatus} (${res.status})`);
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
      setError(err instanceof Error ? err.message : t.errLoadExample);
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
        throw new Error(`${t.errUploadStatus} (${res.status})`);
      }
      const data = (await res.json()) as { file_path: string; parse: ParsedPayload };
      setUploadedPath(data.file_path);
      setParsed(data.parse);
      setSelectedChain(data.parse.selected_chain);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errUpload);
    } finally {
      setActivePhase("idle");
    }
  }

  async function generateReport() {
    if (!parsed) {
      setError(t.errParseFirst);
      return;
    }
    if (!selectedAnalyses.length) {
      setError(t.errSelectAnalysis);
      return;
    }
    setError("");
    setActivePhase("generate");
    setProgress(0);
    setProgressMessage(t.starting);
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
            setProgressMessage(payload.message || t.running);
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
            setError(payload.message || t.errStream);
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
              setError(payload.message || t.errGenerate);
              setProgressMessage(t.failed);
              return;
            }
            setReportText(payload.report_text || "");
            setAiReport(payload.ai_report || "");
            setHtmlUrl(payload.html_download_url || "");
            setPdfUrl(payload.pdf_download_url || "");
            setProgress(1);
            setProgressMessage(t.completed);
          }
        }
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errGenerate);
    } finally {
      setActivePhase("idle");
    }
  }

  return (
    <div className="report-page">
      <header className="chat-header report-header">
        <div className="report-header-main">
          <h2>{t.title}</h2>
          <div className="chat-header-subrow">
            <p>{t.subtitle}</p>
          </div>
        </div>
        <div className={`report-status-pill ${hasActiveTask ? "running" : "idle"}`}>
          <span className="report-status-dot" />
          {isGenerateLoading ? t.stateGenerating : hasActiveTask ? t.stateProcessing : t.stateReady}
        </div>
      </header>

      <section className="report-grid">
        <aside className="chat-panel left report-control-panel">
          <section className="custom-section-card">
            <h3 className="report-block-title">{t.input}</h3>
            <div className="report-mode-row">
              <SegmentedSwitch
                value={inputMode}
                onChange={setInputMode}
                ariaLabel={t.inputModeAria}
                className="report-segment-switch"
                options={[
                  { value: "paste", label: t.paste },
                  { value: "upload", label: t.upload }
                ]}
              />
            </div>

            {inputMode === "paste" ? (
              <textarea
                className="report-input-textarea"
                rows={10}
                value={pasteContent}
                onChange={(e) => setPasteContent(e.target.value)}
                placeholder={t.pastePlaceholder}
              />
            ) : (
              <div className="upload-source-stack">
                <div className="file-source-inline">
                  <input
                    className="report-file-input"
                    type="file"
                    accept=".fasta,.fa,.pdb"
                    onChange={(e) => void onUploadFile(e.target.files?.[0] || null)}
                  />
                  <WorkspaceFilePicker
                    workspaceEnabled={workspaceEnabled}
                    disabled={hasActiveTask}
                    acceptedCategories={["sequence", "structure"]}
                    buttonLabel={t.fromWorkspace}
                    onPick={(picked) => {
                      const selected = picked[0];
                      if (!selected) return;
                      setUploadedPath(selected.storage_path);
                    }}
                  />
                </div>
                <button
                  type="button"
                  className="custom-btn-secondary"
                  onClick={() => void loadDefaultExample()}
                  disabled={hasActiveTask}
                >
                  {isExampleLoading ? t.loadingExample : t.useExample}
                </button>
              </div>
            )}

            {parsed && (
              <>
                <label className="left-controls">
                  {t.chain}
                  <select value={selectedChain} onChange={(e) => setSelectedChain(e.target.value)}>
                    {Object.keys(parsed.sequence_map).map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="report-preview">{t.preview} {parsed.preview}</div>
              </>
            )}
          </section>

          <section className="custom-section-card">
            <h3 className="report-block-title">{t.analysisTypes}</h3>
            <div className="report-options">
              {analysisOptions.map((opt) => (
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
              {isGenerateLoading ? t.generating : t.generateBtn}
            </button>
            <div className="report-progress-wrap">
              <div className="report-progress-text">
                <span>{progressMessage || t.idle}</span>
                <strong>{Math.round(progress * 100)}%</strong>
              </div>
              <div className="report-progress-track">
                <div
                  className="report-progress-bar"
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
            </div>
          </section>

          {error && <div className="error-box">{error}</div>}
        </aside>

        <section className="chat-panel report-output report-main-panel">
          <div className="report-result-header">
            <h3>{t.aiAnalysis}</h3>
            <div className="report-downloads">
              {htmlUrl && (
                <a className="chat-header-link" href={htmlUrl} target="_blank" rel="noreferrer">
                  {t.downloadHtml}
                </a>
              )}
              {pdfUrl && (
                <a className="chat-header-link" href={pdfUrl} target="_blank" rel="noreferrer">
                  {t.downloadPdf}
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
              <pre className="report-text report-ai-box">{t.aiPlaceholder}</pre>
            </div>
          )}
        </section>

        <aside className="chat-panel right report-right-panel">
          <section className="report-side-card">
            <h3>{t.streamingLogs}</h3>
            <div className="report-copy-wrap report-log-copy-wrap">
              <pre className="report-log-box">{streamLogs.length ? streamLogs.join("\n") : t.logsPlaceholder}</pre>
            </div>
          </section>
        </aside>
      </section>
      <PageFooter />
    </div>
  );
}
