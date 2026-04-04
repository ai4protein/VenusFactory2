import { useEffect, useMemo, useState } from "react";
import {
  fetchAdvancedToolsMeta,
  loadAdvancedDefaultExample,
  runAdvancedProteinFunctionStream,
  type AdvancedToolsMeta,
  uploadAdvancedToolFile
} from "../../lib/advancedToolsApi";
import { AdvancedToolsLayout } from "./AdvancedToolsLayout";
import { AdvancedResultPanel } from "./AdvancedResultPanel";
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";

const DEFAULT_META: AdvancedToolsMeta = {
  dataset_mapping_zero_shot: [],
  sequence_model_options: [],
  structure_model_options: [],
  model_mapping_function: ["ESM2-650M"],
  residue_model_mapping_function: ["ESM2-650M"],
  dataset_mapping_function: { Solubility: ["DeepSol"] },
  residue_mapping_function: { "Activity Site": ["Protein_Mutation"] },
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"]
};

type AdvancedProteinFunctionPageProps = {
  workspaceEnabled: boolean;
};

export function AdvancedProteinFunctionPage({ workspaceEnabled }: AdvancedProteinFunctionPageProps) {
  const [meta, setMeta] = useState<AdvancedToolsMeta>(DEFAULT_META);
  const [task, setTask] = useState("Solubility");
  const [modelName, setModelName] = useState("ESM2-650M");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [sequence, setSequence] = useState("");
  const [uploadedPath, setUploadedPath] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null);
  const [enableAi, setEnableAi] = useState(false);
  const [llmProvider, setLlmProvider] = useState(DEFAULT_META.llm_models[0]);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("Idle");

  useEffect(() => {
    void (async () => {
      const loaded = await fetchAdvancedToolsMeta();
      setMeta(loaded);
      const firstTask = Object.keys(loaded.dataset_mapping_function)[0];
      if (firstTask) {
        setTask(firstTask);
        setSelectedDatasets(loaded.dataset_mapping_function[firstTask] || []);
      }
      if (loaded.model_mapping_function.length > 0) setModelName(loaded.model_mapping_function[0]);
      if (loaded.llm_models.length > 0) setLlmProvider(loaded.llm_models[0]);
    })();
  }, []);

  const datasetOptions = useMemo(
    () => meta.dataset_mapping_function[task] || [],
    [meta.dataset_mapping_function, task]
  );

  useEffect(() => {
    if (datasetOptions.length > 0) {
      setSelectedDatasets(datasetOptions);
    } else {
      setSelectedDatasets([]);
    }
  }, [datasetOptions]);

  function toggleDataset(dataset: string) {
    setSelectedDatasets((prev) =>
      prev.includes(dataset) ? prev.filter((item) => item !== dataset) : [...prev, dataset]
    );
  }

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const data = await uploadAdvancedToolFile(file);
      setUploadedPath(data.file_path);
      if (data.suffix === ".fasta" || data.suffix === ".fa" || data.suffix === ".txt") {
        setSequence(await file.text());
      } else {
        setSequence("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  async function onRun() {
    setError("");
    setResultPayload(null);
    setProgress(0);
    setProgressMessage("Preparing task...");
    setRunning(true);
    try {
      const payload = await runAdvancedProteinFunctionStream({
        task,
        file_path: uploadedPath || undefined,
        sequence: sequence.trim() || undefined,
        model_name: modelName,
        datasets: selectedDatasets,
        enable_ai: enableAi,
        llm_provider: llmProvider,
        user_api_key: ""
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
      const data = await loadAdvancedDefaultExample("fasta");
      setUploadedPath(data.file_path);
      setSequence(data.content || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load example.");
    }
  }

  return (
    <AdvancedToolsLayout
      title="Protein Function"
      subtitle="Predict protein functions across selected datasets."
      running={running}
      progress={progress}
      progressMessage={progressMessage}
      left={
        <>
          <section className="custom-section-card">
            <h3>Model and Task</h3>
            <label className="left-controls">
              Model
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {meta.model_mapping_function.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label className="left-controls">
              Task
              <select value={task} onChange={(e) => setTask(e.target.value)}>
                {Object.keys(meta.dataset_mapping_function).map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <div className="left-controls">
              <span>Datasets (Multi-select)</span>
              <div className="advanced-dataset-toolbar">
                <span className="advanced-dataset-count">{selectedDatasets.length} selected</span>
                <div className="advanced-dataset-actions">
                  <button
                    type="button"
                    className="custom-btn-secondary"
                    onClick={() => setSelectedDatasets(datasetOptions)}
                    disabled={datasetOptions.length === 0}
                  >
                    Select All
                  </button>
                  <button
                    type="button"
                    className="custom-btn-secondary"
                    onClick={() => setSelectedDatasets([])}
                    disabled={selectedDatasets.length === 0}
                  >
                    Clear
                  </button>
                </div>
              </div>
              <div className="advanced-dataset-grid">
                {datasetOptions.map((item) => {
                  const checked = selectedDatasets.includes(item);
                  return (
                    <button
                      key={item}
                      type="button"
                      className={`advanced-dataset-item ${checked ? "active" : ""}`}
                      aria-pressed={checked}
                      onClick={() => toggleDataset(item)}
                    >
                      {checked && <span className="advanced-dataset-item-status">selected</span>}
                      <span className="advanced-dataset-item-label">{item}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </section>

          <section className="custom-section-card">
            <h3>Input</h3>
            <label className="left-controls">
              Paste FASTA / sequence
              <textarea
                rows={6}
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="Paste sequence or FASTA content..."
              />
            </label>
            {meta.online_limit_enabled && (
              <p className="advanced-ai-note">
                Online mode supports up to {meta.online_fasta_limit ?? 50} FASTA sequences per run.
              </p>
            )}
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  Select File
                  <input type="file" accept=".fasta,.fa,.txt" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
                  acceptedCategories={["sequence"]}
                  buttonLabel="From Workspace"
                  onPick={(picked) => {
                    const selected = picked[0];
                    if (!selected) return;
                    setUploadedPath(selected.storage_path);
                    setSequence("");
                  }}
                />
              </div>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                Use Example FASTA
              </button>
            </div>
            {uploadedPath && <div className="report-preview">Uploaded: {uploadedPath}</div>}
          </section>

          <section className="custom-section-card advanced-ai-section">
            <h3>AI Expert (Optional)</h3>
            <label className="advanced-ai-toggle">
              <input type="checkbox" checked={enableAi} onChange={(e) => setEnableAi(e.target.checked)} />
              <span className="advanced-ai-toggle-box" />
              <span className="advanced-ai-toggle-text">Enable AI analysis and expert summary</span>
              <span className={`advanced-ai-pill ${enableAi ? "active" : ""}`}>{enableAi ? "Enabled" : "Disabled"}</span>
            </label>
            <p className="advanced-ai-note">
              {enableAi
                ? "AI insight will be generated and attached to the result panel."
                : "Enable this to generate expert interpretation after prediction."}
            </p>
            {enableAi && (
              <div className="advanced-ai-fields">
                <label className="left-controls">
                  LLM Provider
                  <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)}>
                    {meta.llm_models.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}
          </section>

          <button type="button" className="custom-btn-primary" onClick={() => void onRun()} disabled={running}>
            {running ? "Running..." : "Start Function Prediction"}
          </button>
        </>
      }
      right={
        <AdvancedResultPanel
          title="Protein Function Result"
          resultPayload={resultPayload}
          aiSummary={(resultPayload?.ai_summary as string) || ""}
          error={error}
          showSummaryTab={false}
          enableHeatmapTab={false}
        />
      }
    />
  );
}
