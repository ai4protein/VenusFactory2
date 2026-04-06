import { useState } from "react";
import {
  loadQuickToolDefaultExample,
  runProteinDiscoveryToolStream,
  uploadQuickToolFile
} from "../../lib/quickToolsApi";
import { QuickToolsLayout } from "./QuickToolsLayout";
import { QuickToolResultPanel } from "./QuickToolResultPanel";
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";

type ProteinDiscoveryPageProps = {
  readonly?: boolean;
  workspaceEnabled?: boolean;
};

export function ProteinDiscoveryPage({ readonly = false, workspaceEnabled = false }: ProteinDiscoveryPageProps) {
  const [uploadedPath, setUploadedPath] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("Idle");

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const data = await uploadQuickToolFile(file);
      if (data.suffix !== ".pdb") {
        throw new Error("Protein Discovery only supports PDB structure input.");
      }
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadQuickToolDefaultExample("pdb");
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load example.");
    }
  }

  async function onRun() {
    setError("");
    setProgress(0);
    setProgressMessage("Preparing task...");
    setRunning(true);
    try {
      if (!uploadedPath) {
        throw new Error("Please upload or pick a PDB file first.");
      }
      const payload = await runProteinDiscoveryToolStream(
        { pdbFile: uploadedPath },
        (evt) => {
          setProgress(evt.progress);
          setProgressMessage(evt.message);
        }
      );
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage("Prediction completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <QuickToolsLayout
      title="Protein Discovery (VenusMine)"
      subtitle="Quick mode keeps only PDB input. Advanced discovery parameters use backend defaults."
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
              <div className="custom-file-example-row upload-source-stack">
                <div className="file-source-inline">
                  <label className="left-controls custom-file-picker-field">
                    Select File
                    <input type="file" accept=".pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                  </label>
                  <WorkspaceFilePicker
                    workspaceEnabled={workspaceEnabled}
                    disabled={running || readonly}
                    acceptedCategories={["structure"]}
                    buttonLabel="From Workspace"
                    onPick={(picked) => {
                      const selected = picked[0];
                      if (!selected || selected.suffix !== ".pdb") return;
                      setUploadedPath(selected.storage_path);
                    }}
                  />
                </div>
                <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                  Use Example PDB
                </button>
              </div>
              {uploadedPath && <div className="report-preview">Selected: {uploadedPath}</div>}
            </section>

            <button
              type="button"
              className="custom-btn-primary advanced-discovery-submit"
              onClick={() => void onRun()}
              disabled={running || !uploadedPath}
            >
              {running ? "Running..." : "Start VenusMine Discovery"}
            </button>
          </fieldset>
        </div>
      }
      right={
        <QuickToolResultPanel
          title="Protein Discovery Result"
          resultPayload={resultPayload}
          aiSummary=""
          error={error}
        />
      }
    />
  );
}
