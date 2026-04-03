import { useEffect, useMemo, useState } from "react";
import {
  fetchAdvancedToolsMeta,
  loadAdvancedDefaultExample,
  runAdvancedDirectedEvolutionStream,
  type AdvancedToolsMeta,
  uploadAdvancedToolFile
} from "../../lib/advancedToolsApi";
import { AdvancedToolsLayout } from "./AdvancedToolsLayout";
import { AdvancedResultPanel } from "./AdvancedResultPanel";
import { SegmentedSwitch } from "../../components/SegmentedSwitch";

const DEFAULT_META: AdvancedToolsMeta = {
  dataset_mapping_zero_shot: ["Activity", "Binding", "Expression", "Organismal Fitness", "Stability"],
  sequence_model_options: ["VenusPLM", "ESM2-650M", "ESM-1v", "ESM-1b"],
  structure_model_options: ["VenusREM (foldseek-based)", "ProSST-2048", "ProtSSN", "ESM-IF1", "SaProt", "MIF-ST"],
  model_mapping_function: ["ESM2-650M"],
  residue_model_mapping_function: ["ESM2-650M"],
  dataset_mapping_function: { Solubility: ["DeepSol"] },
  residue_mapping_function: { "Activity Site": ["Protein_Mutation"] },
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"]
};

export function AdvancedDirectedEvolutionPage() {
  const [meta, setMeta] = useState<AdvancedToolsMeta>(DEFAULT_META);
  const [inputMode, setInputMode] = useState<"sequence" | "structure">("sequence");
  const [modelName, setModelName] = useState(DEFAULT_META.sequence_model_options[0]);
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
      if (loaded.sequence_model_options.length > 0) setModelName(loaded.sequence_model_options[0]);
      if (loaded.llm_models.length > 0) setLlmProvider(loaded.llm_models[0]);
    })();
  }, []);

  useEffect(() => {
    if (inputMode === "sequence" && meta.sequence_model_options.length > 0) {
      setModelName(meta.sequence_model_options[0]);
    }
    if (inputMode === "structure" && meta.structure_model_options.length > 0) {
      setModelName(meta.structure_model_options[0]);
    }
  }, [inputMode, meta.sequence_model_options, meta.structure_model_options]);

  const modelOptions = useMemo(
    () => (inputMode === "sequence" ? meta.sequence_model_options : meta.structure_model_options),
    [inputMode, meta.sequence_model_options, meta.structure_model_options]
  );

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
      const payload = await runAdvancedDirectedEvolutionStream({
        input_mode: inputMode,
        file_path: uploadedPath || undefined,
        sequence: sequence.trim() || undefined,
        model_name: modelName,
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
      const kind = inputMode === "structure" ? "pdb" : "fasta";
      const data = await loadAdvancedDefaultExample(kind);
      setUploadedPath(data.file_path);
      if (kind === "fasta") {
        setSequence(data.content || "");
      } else {
        setSequence("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load example.");
    }
  }

  return (
    <AdvancedToolsLayout
      title="Directed Evolution"
      subtitle="Run saturation mutagenesis scoring with sequence or structure models."
      running={running}
      progress={progress}
      progressMessage={progressMessage}
      left={
        <>
          <section className="custom-section-card">
            <div className="advanced-section-caption">Prediction Mode</div>
            <div className="advanced-mode-row">
              <SegmentedSwitch
                value={inputMode}
                onChange={setInputMode}
                ariaLabel="Prediction mode switch"
                className="advanced-mode-segment-switch"
                options={[
                  { value: "sequence", label: "Sequence Model" },
                  { value: "structure", label: "Structure Model" }
                ]}
              />
            </div>
            <div className="advanced-mode-hint">
              {inputMode === "sequence"
                ? "Sequence mode: paste FASTA/sequence or upload .fasta/.fa"
                : "Structure mode: upload .pdb for structure-based scoring"}
            </div>
            <label className="left-controls advanced-full-row">
              Select Model
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {modelOptions.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section className="custom-section-card">
            <h3>Input</h3>
            {inputMode === "sequence" && (
              <label className="left-controls">
                Paste Sequence / FASTA
                <textarea
                  rows={6}
                  value={sequence}
                  onChange={(e) => setSequence(e.target.value)}
                  placeholder="Paste raw sequence or FASTA content..."
                />
              </label>
            )}
            <p className="advanced-ai-note">Directed Evolution supports one protein per run (sequence or structure).</p>
            <div className="custom-file-example-row">
              <label className="left-controls custom-file-picker-field">
                Select File
                <input
                  type="file"
                  accept={inputMode === "sequence" ? ".fasta,.fa,.txt" : ".pdb"}
                  onChange={(e) => void onUpload(e.target.files?.[0] || null)}
                />
              </label>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                {inputMode === "structure" ? "Use Example PDB" : "Use Example FASTA"}
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
            {running ? "Running..." : "Start Directed Evolution"}
          </button>
        </>
      }
      right={
        <AdvancedResultPanel
          title="Directed Evolution Result"
          resultPayload={resultPayload}
          aiSummary={(resultPayload?.ai_summary as string) || ""}
          error={error}
          showSummaryTab={false}
          enableHeatmapTab
        />
      }
    />
  );
}
