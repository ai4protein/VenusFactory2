import { useMemo, useState } from "react";
import { buildArchiveDownloadUrl, runDownloadTask, type DownloadMethod, type DownloadTaskResponse } from "../../lib/downloadApi";
import { DownloadLayout } from "./DownloadLayout";
import { CopyableTextBlock } from "../../components/CommandPreviewCard";
import { SegmentedSwitch } from "../../components/SegmentedSwitch";

type DownloadTaskConfig = {
  title: string;
  subtitle: string;
  endpoint: "uniprot" | "ncbi" | "rcsb-structure" | "alphafold-structure" | "rcsb-metadata" | "interpro-metadata";
  idLabel: string;
  idPlaceholder: string;
  defaultId: string;
  supportsMerge?: boolean;
  supportsFileType?: boolean;
  showVisualization?: boolean;
  fileHint: string;
};

type DownloadTaskPageProps = {
  config: DownloadTaskConfig;
};

export function DownloadTaskPage({ config }: DownloadTaskPageProps) {
  const [method, setMethod] = useState<DownloadMethod>("Single ID");
  const [idValue, setIdValue] = useState(config.defaultId);
  const [fileContent, setFileContent] = useState("");
  const [filePreview, setFilePreview] = useState("");
  const [merge, setMerge] = useState(false);
  const [saveErrorFile, setSaveErrorFile] = useState(true);
  const [fileType, setFileType] = useState<"pdb" | "cif">("pdb");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<DownloadTaskResponse | null>(null);

  const archiveUrl = useMemo(() => {
    if (!result?.archive_relative_path) return "";
    return buildArchiveDownloadUrl(result.archive_relative_path);
  }, [result]);

  const visualizationStatus = useMemo(() => {
    const raw = result?.details?.visualization_status;
    return typeof raw === "string" ? raw : "";
  }, [result]);

  const statusTone = running
    ? "running"
    : error || (result && !result.success)
      ? "failed"
      : result?.success
        ? "success"
        : "idle";
  const statusText = running
    ? "Download in progress..."
    : error
      ? error
      : result
        ? result.success
          ? "Download completed successfully."
          : "Download finished with errors."
        : "Ready to start a download task.";

  async function onUpload(file: File | null) {
    if (!file) {
      setFileContent("");
      setFilePreview("");
      return;
    }
    const text = await file.text();
    setFileContent(text);
    const lines = text.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
    const previewLines = lines.slice(0, 20);
    let preview = previewLines.join("\n");
    if (lines.length > 20) {
      preview += `\n... and ${lines.length - 20} more entries (showing first 20)`;
    }
    setFilePreview(preview);
  }

  async function onRun() {
    setError("");
    setRunning(true);
    try {
      const payload = await runDownloadTask(config.endpoint, {
        method,
        id_value: idValue.trim(),
        file_content: fileContent,
        save_error_file: saveErrorFile,
        merge: config.supportsMerge ? merge : undefined,
        file_type: config.supportsFileType ? fileType : undefined,
        unzip: config.supportsFileType ? true : undefined
      });
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <DownloadLayout
      title={config.title}
      subtitle={config.subtitle}
      running={running}
      left={
        <>
          <section className="custom-section-card">
            <h3>Download Method</h3>
            <div className="custom-row">
              <SegmentedSwitch
                value={method}
                onChange={setMethod}
                ariaLabel="Download method"
                className="download-segment-switch"
                options={[
                  { value: "Single ID", label: "Single ID" },
                  { value: "From File", label: "From File" }
                ]}
              />
            </div>

            {method === "Single ID" ? (
              <label className="left-controls download-field">
                {config.idLabel}
                <input
                  className="download-input-field"
                  value={idValue}
                  onChange={(e) => setIdValue(e.target.value)}
                  placeholder={config.idPlaceholder}
                />
              </label>
            ) : (
              <div className="left-controls download-field">
                <input
                  className="download-file-input"
                  type="file"
                  accept=".txt"
                  onChange={(e) => void onUpload(e.target.files?.[0] || null)}
                />
                <small>{config.fileHint}</small>
                <pre className="download-file-preview">{filePreview || "Upload a .txt file with one ID per line."}</pre>
              </div>
            )}
          </section>

          <section className="custom-section-card">
            <h3>Task Options</h3>
            {config.supportsFileType && (
              <label className="left-controls download-field">
                Structure File Type
                <select className="download-input-field" value={fileType} onChange={(e) => setFileType(e.target.value as "pdb" | "cif")}>
                  <option value="pdb">pdb</option>
                  <option value="cif">cif</option>
                </select>
              </label>
            )}

            {config.supportsMerge && (
              <label className="download-option-item">
                <input type="checkbox" checked={merge} onChange={(e) => setMerge(e.target.checked)} />
                <span>
                  Merge outputs into one FASTA
                </span>
              </label>
            )}

            <label className="download-option-item">
              <input type="checkbox" checked={saveErrorFile} onChange={(e) => setSaveErrorFile(e.target.checked)} />
              <span>
                Save error file
              </span>
            </label>
          </section>

          <button type="button" className="download-action-btn" onClick={() => void onRun()} disabled={running}>
            {running ? "Running..." : "Start Download"}
          </button>
        </>
      }
      right={
        <div className="download-result-wrap">
          <div className="report-result-header quick-tools-v2-result-header">
            <h3>Download Result</h3>
            <div className="report-downloads">
              {archiveUrl && (
                <a className="download-archive-btn" href={archiveUrl} target="_blank" rel="noreferrer">
                  Save Downloaded Data
                </a>
              )}
            </div>
          </div>

          <div className={`download-status-banner ${statusTone}`}>
            <strong>Status:</strong> {statusText}
          </div>

          {config.showVisualization && (
            <div className="download-viz-status">
              <strong>Visualization:</strong> {visualizationStatus || "No structure preview available yet."}
            </div>
          )}

          <div className="download-result-grid">
            <section className={`download-result-card ${statusTone === "failed" ? "failed" : ""}`}>
              <h4>Output Preview</h4>
              <CopyableTextBlock
                text={result?.preview || ""}
                emptyText="Preview will appear here after download."
                wrapperClassName="quick-tools-v2-copy-wrap"
                preClassName="report-text quick-tools-v2-text"
                ariaLabel="Copy output preview"
              />
            </section>
            <section className={`download-result-card ${statusTone === "failed" ? "failed" : statusTone === "success" ? "success" : ""}`}>
              <h4>Download Status</h4>
              <CopyableTextBlock
                text={result?.message || ""}
                emptyText="Status logs will appear here."
                wrapperClassName="quick-tools-v2-copy-wrap"
                preClassName="report-text quick-tools-v2-text"
                ariaLabel="Copy download status"
              />
            </section>
          </div>
        </div>
      }
    />
  );
}
