import { useState } from "react";
import {
  loadAdvancedDefaultExample,
  runAdvancedProteinDiscoveryStream,
  uploadAdvancedToolFile
} from "../../lib/advancedToolsApi";
import { AdvancedToolsLayout } from "./AdvancedToolsLayout";
import { AdvancedResultPanel } from "./AdvancedResultPanel";

type AdvancedProteinDiscoveryPageProps = {
  readonly?: boolean;
};

export function AdvancedProteinDiscoveryPage({ readonly = false }: AdvancedProteinDiscoveryPageProps) {
  const [uploadedPath, setUploadedPath] = useState("");
  const [protectStart, setProtectStart] = useState(1);
  const [protectEnd, setProtectEnd] = useState(100);
  const [mmseqsThreads, setMmseqsThreads] = useState(96);
  const [mmseqsIterations, setMmseqsIterations] = useState(3);
  const [mmseqsMaxSeqs, setMmseqsMaxSeqs] = useState(100);
  const [clusterMinSeqId, setClusterMinSeqId] = useState(0.5);
  const [clusterThreads, setClusterThreads] = useState(96);
  const [topNThreshold, setTopNThreshold] = useState(10);
  const [evalueThreshold, setEvalueThreshold] = useState(1e-5);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("Idle");

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const data = await uploadAdvancedToolFile(file);
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  async function onRun() {
    setError("");
    setProgress(0);
    setProgressMessage("Preparing task...");
    setRunning(true);
    try {
      const payload = await runAdvancedProteinDiscoveryStream({
        pdb_file: uploadedPath,
        protect_start: protectStart,
        protect_end: protectEnd,
        mmseqs_threads: mmseqsThreads,
        mmseqs_iterations: mmseqsIterations,
        mmseqs_max_seqs: mmseqsMaxSeqs,
        cluster_min_seq_id: clusterMinSeqId,
        cluster_threads: clusterThreads,
        top_n_threshold: topNThreshold,
        evalue_threshold: evalueThreshold
      }, (evt) => {
        setProgress(evt.progress);
        setProgressMessage(evt.message);
      });
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage("Prediction completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run failed.");
    } finally {
      setRunning(false);
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadAdvancedDefaultExample("pdb");
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load example.");
    }
  }

  return (
    <AdvancedToolsLayout
      title="Protein Discovery (VenusMine)"
      subtitle="Search and cluster structural homologs with FoldSeek and MMseqs."
      running={running}
      progress={progress}
      progressMessage={progressMessage}
      left={
        <div className={`advanced-discovery-form ${readonly ? "readonly-mode" : ""}`}>
          {readonly && (
            <div className="readonly-banner" role="status" aria-live="polite">
              Online mode: protein discovery controls are view-only in this deployment.
            </div>
          )}
          <fieldset className="readonly-fieldset advanced-discovery-fieldset" disabled={readonly}>
            <section className="custom-section-card">
              <h3>PDB Input</h3>
              <div className="custom-file-example-row">
                <label className="left-controls custom-file-picker-field">
                  Select File
                  <input type="file" accept=".pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                  Use Example PDB
                </button>
              </div>
              {uploadedPath && <div className="report-preview">Uploaded: {uploadedPath}</div>}
            </section>

            <section className="custom-section-card">
              <h3>Advanced Parameters</h3>
              <label className="left-controls">
                Protected Region Start
                <input type="number" value={protectStart} onChange={(e) => setProtectStart(Number(e.target.value) || 1)} />
              </label>
              <label className="left-controls">
                Protected Region End
                <input type="number" value={protectEnd} onChange={(e) => setProtectEnd(Number(e.target.value) || 100)} />
              </label>
              <label className="left-controls">
                MMseqs Threads
                <input type="number" value={mmseqsThreads} onChange={(e) => setMmseqsThreads(Number(e.target.value) || 1)} />
              </label>
              <label className="left-controls">
                MMseqs Iterations
                <input
                  type="number"
                  value={mmseqsIterations}
                  onChange={(e) => setMmseqsIterations(Number(e.target.value) || 1)}
                />
              </label>
              <label className="left-controls">
                MMseqs Max Sequences
                <input type="number" value={mmseqsMaxSeqs} onChange={(e) => setMmseqsMaxSeqs(Number(e.target.value) || 1)} />
              </label>
              <label className="left-controls">
                Cluster Min Seq Identity
                <input
                  type="number"
                  step="0.01"
                  value={clusterMinSeqId}
                  onChange={(e) => setClusterMinSeqId(Number(e.target.value) || 0.5)}
                />
              </label>
              <label className="left-controls">
                Cluster Threads
                <input type="number" value={clusterThreads} onChange={(e) => setClusterThreads(Number(e.target.value) || 1)} />
              </label>
              <label className="left-controls">
                Tree Top-N Threshold
                <input type="number" value={topNThreshold} onChange={(e) => setTopNThreshold(Number(e.target.value) || 10)} />
              </label>
              <label className="left-controls">
                E-value Threshold
                <input
                  type="number"
                  step="0.000001"
                  value={evalueThreshold}
                  onChange={(e) => setEvalueThreshold(Number(e.target.value) || 0.00001)}
                />
              </label>
            </section>

            <button type="button" className="custom-btn-primary advanced-discovery-submit" onClick={() => void onRun()} disabled={running}>
              {running ? "Running..." : "Start VenusMine Discovery"}
            </button>
          </fieldset>
        </div>
      }
      right={
        <AdvancedResultPanel
          title="Protein Discovery Result"
          resultPayload={resultPayload}
          aiSummary=""
          error={error}
          showSummaryTab={false}
          enableHeatmapTab={false}
          readonly={readonly}
        />
      }
    />
  );
}
