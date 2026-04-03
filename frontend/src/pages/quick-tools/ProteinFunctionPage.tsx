import { useEffect, useState } from "react";
import {
  fetchQuickToolsMeta,
  loadQuickToolDefaultExample,
  normalizePastedFastaForDisplay,
  requestQuickToolAiSummary,
  runProteinFunctionToolStream,
  type QuickToolsMeta,
  validateFastaWithHeader,
  uploadQuickToolFile,
  uploadSequenceAsFasta
} from "../../lib/quickToolsApi";
import { QuickToolsLayout } from "./QuickToolsLayout";
import { QuickToolResultPanel } from "./QuickToolResultPanel";

const DEFAULT_META: QuickToolsMeta = {
  dataset_mapping_zero_shot: [],
  model_mapping_zero_shot: [],
  dataset_mapping_function: ["Solubility"],
  residue_mapping_function: [],
  protein_properties_function: [],
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"]
};

export function ProteinFunctionPage() {
  const [meta, setMeta] = useState<QuickToolsMeta>(DEFAULT_META);
  const [task, setTask] = useState(DEFAULT_META.dataset_mapping_function[0]);
  const [sequence, setSequence] = useState("");
  const [uploadedPath, setUploadedPath] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null);
  const [aiSummary, setAiSummary] = useState("");
  const [enableAi, setEnableAi] = useState(false);
  const [llmProvider, setLlmProvider] = useState(DEFAULT_META.llm_models[0]);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("Idle");
  useEffect(() => {
    void (async () => {
      const loaded = await fetchQuickToolsMeta();
      setMeta(loaded);
      if (loaded.dataset_mapping_function.length > 0) setTask(loaded.dataset_mapping_function[0]);
      if (loaded.llm_models.length > 0) setLlmProvider(loaded.llm_models[0]);
    })();
  }, []);

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      validateFastaWithHeader(await file.text());
      const data = await uploadQuickToolFile(file);
      setUploadedPath(data.file_path);
      const content = await file.text();
      setSequence(normalizePastedFastaForDisplay(content));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadQuickToolDefaultExample("fasta");
      setUploadedPath(data.file_path);
      setSequence(data.content || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load example.");
    }
  }

  async function resolveFastaFile(): Promise<string> {
    if (uploadedPath) return uploadedPath;
    if (sequence.trim()) {
      const uploaded = await uploadSequenceAsFasta(sequence);
      setUploadedPath(uploaded.file_path);
      return uploaded.file_path;
    }
    throw new Error("Please upload a FASTA file or paste sequence.");
  }

  async function onRun() {
    setError("");
    setAiSummary("");
    setRunning(true);
    setProgress(0);
    setProgressMessage("Preparing task...");
    try {
      const fastaPath = await resolveFastaFile();
      const payload = await runProteinFunctionToolStream({ fastaFile: fastaPath, task }, (evt) => {
        setProgress(evt.progress);
        setProgressMessage(evt.message);
      });
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage("Prediction completed");
      if (enableAi) {
        const ai = await requestQuickToolAiSummary({
          tool: "function",
          task,
          provider: llmProvider,
          userApiKey: "",
          resultPayload: payload
        });
        setAiSummary(ai.summary);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <QuickToolsLayout
      title="Protein Function"
      subtitle="Predict protein-level function from FASTA sequences."
      running={running}
      progress={progress}
      progressMessage={progressMessage}
      left={
        <>
          <section className="custom-section-card">
            <h3>Task Configuration</h3>
            <label className="left-controls">
              Select Task
              <select value={task} onChange={(e) => setTask(e.target.value)}>
                {meta.dataset_mapping_function.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section className="custom-section-card">
            <h3>Data Input</h3>
            <label className="left-controls">
              Paste FASTA/sequence
              <textarea
                rows={7}
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="Paste FASTA content (must include >header)..."
              />
            </label>
            {meta.online_limit_enabled && (
              <p className="quick-ai-note">
                Online mode supports up to {meta.online_fasta_limit ?? 50} FASTA sequences per run.
              </p>
            )}
            <div className="custom-file-example-row">
              <label className="left-controls custom-file-picker-field">
                Select File
                <input type="file" accept=".fasta,.fa" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
              </label>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                Use Example FASTA
              </button>
            </div>
            {uploadedPath && <div className="report-preview">Uploaded: {uploadedPath}</div>}
          </section>

          <section className="custom-section-card quick-ai-section">
            <h3>AI Expert (Optional)</h3>
            <label className="quick-ai-toggle">
              <input type="checkbox" checked={enableAi} onChange={(e) => setEnableAi(e.target.checked)} />
              <span className="quick-ai-toggle-box" />
              <span className="quick-ai-toggle-text">Enable AI analysis and expert summary</span>
              <span className={`quick-ai-pill ${enableAi ? "active" : ""}`}>{enableAi ? "Enabled" : "Disabled"}</span>
            </label>
            <p className="quick-ai-note">
              {enableAi
                ? "AI expert interpretation will be generated together with prediction output."
                : "Turn on to generate an expert summary after prediction finishes."}
            </p>
            {enableAi && (
              <div className="quick-ai-fields">
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
            {running ? "Running..." : "Start Prediction"}
          </button>
        </>
      }
      right={
        <QuickToolResultPanel
          title="Protein Function Result"
          resultPayload={resultPayload}
          aiSummary={aiSummary}
          error={error}
          enableHeatmapTab={false}
        />
      }
    />
  );
}
