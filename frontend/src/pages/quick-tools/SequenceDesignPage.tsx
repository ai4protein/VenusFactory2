import { useEffect, useState } from "react";
import {
  fetchQuickToolsMeta,
  loadQuickToolDefaultExample,
  requestQuickToolAiSummary,
  runSequenceDesignToolStream,
  type QuickToolsMeta,
  uploadQuickToolFile
} from "../../lib/quickToolsApi";
import { QuickToolsLayout } from "./QuickToolsLayout";
import { QuickToolResultPanel } from "./QuickToolResultPanel";
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";
type ModelFamily = "soluble" | "vanilla" | "ca";

const DEFAULT_MODEL_NAME = "v_48_020";
const DEFAULT_BACKBONE_NOISE = 0.2;

const DEFAULT_META: QuickToolsMeta = {
  dataset_mapping_zero_shot: [],
  model_mapping_zero_shot: [],
  dataset_mapping_function: [],
  residue_mapping_function: [],
  protein_properties_function: [],
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"]
};

type SequenceDesignPageProps = {
  workspaceEnabled?: boolean;
};

function parseChainList(input: string): string[] {
  return input
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter((item) => /^[A-Z0-9]$/.test(item));
}

function diversityToTemperatures(level: "low" | "medium" | "high"): number[] {
  if (level === "high") return [0.3];
  if (level === "medium") return [0.2];
  return [0.1];
}

export function SequenceDesignPage({ workspaceEnabled = false }: SequenceDesignPageProps) {
  const [meta, setMeta] = useState<QuickToolsMeta>(DEFAULT_META);
  const [uploadedPath, setUploadedPath] = useState("");
  const [designedChainsText, setDesignedChainsText] = useState("");
  const [fixedResiduesText, setFixedResiduesText] = useState("");
  const [modelFamily, setModelFamily] = useState<ModelFamily>("soluble");
  const [numSequences, setNumSequences] = useState(8);
  const [diversity, setDiversity] = useState<"low" | "medium" | "high">("medium");
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
      if (loaded.llm_models.length > 0) setLlmProvider(loaded.llm_models[0]);
    })();
  }, []);

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const data = await uploadQuickToolFile(file);
      if (data.suffix !== ".pdb") {
        throw new Error("Sequence Design only supports PDB structure input.");
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
    setAiSummary("");
    setRunning(true);
    setProgress(0);
    setProgressMessage("Preparing sequence design...");
    try {
      if (!uploadedPath) {
        throw new Error("Please upload or pick a PDB file first.");
      }
      const payload = await runSequenceDesignToolStream(
        {
          structureFile: uploadedPath,
          modelFamily,
          designedChains: parseChainList(designedChainsText),
          fixedResiduesText: fixedResiduesText.trim(),
          numSequences,
          modelName: DEFAULT_MODEL_NAME,
          backboneNoise: DEFAULT_BACKBONE_NOISE,
          useSolubleModel: modelFamily === "soluble",
          caOnly: modelFamily === "ca",
          temperatures: diversityToTemperatures(diversity)
        },
        (evt) => {
          setProgress(evt.progress);
          setProgressMessage(evt.message);
        }
      );
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage("Sequence design completed");
      if (enableAi) {
        const ai = await requestQuickToolAiSummary({
          tool: "sequence-design",
          task: "ProteinMPNN",
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
      title="Sequence Design"
      subtitle="Design protein sequences from a structure with simple, biology-friendly controls."
      running={running}
      progress={progress}
      progressMessage={progressMessage}
      left={
        <>
          <section className="custom-section-card">
            <h3>Task Configuration</h3>
            <label className="left-controls">
              Model Family
              <select value={modelFamily} onChange={(e) => setModelFamily(e.target.value as ModelFamily)}>
                <option value="soluble">Soluble (recommended for protein discovery and design)</option>
                <option value="vanilla">Vanilla (recommended for membrane proteins)</option>
                <option value="ca">CA (only for C-alpha coarse-grained coordinates)</option>
              </select>
            </label>
            <p className="quick-ai-note">
              Quick mode uses <code>{DEFAULT_MODEL_NAME}</code> by default (backbone_noise{" "}
              {DEFAULT_BACKBONE_NOISE.toFixed(2)}A).
            </p>
            <label className="left-controls quick-seq-match-input">
              Designed Chains (optional)
              <input
                value={designedChainsText}
                onChange={(e) => setDesignedChainsText(e.target.value)}
                placeholder="A,B (empty means all chains)"
              />
            </label>
            <label className="left-controls quick-seq-match-input">
              Fixed Residues (optional)
              <input
                value={fixedResiduesText}
                onChange={(e) => setFixedResiduesText(e.target.value)}
                placeholder="A12,C13 or A:12,13;B:5-8"
              />
            </label>
            <div style={{ display: "flex", gap: "12px", alignItems: "stretch" }}>
              <label className="left-controls" style={{ flex: 1, minWidth: 0 }}>
                Number of sequences
                <select value={numSequences} onChange={(e) => setNumSequences(Number(e.target.value))}>
                  {[4, 8, 16, 32].map((count) => (
                    <option key={count} value={count}>
                      {count}
                    </option>
                  ))}
                </select>
              </label>
              <label className="left-controls" style={{ flex: 1, minWidth: 0 }}>
                Design Diversity
                <select value={diversity} onChange={(e) => setDiversity(e.target.value as "low" | "medium" | "high")}>
                  <option value="low">Low (conservative)</option>
                  <option value="medium">Medium (balanced)</option>
                  <option value="high">High (exploratory)</option>
                </select>
              </label>
            </div>
          </section>

          <section className="custom-section-card">
            <h3>Structure Input</h3>
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  Select PDB File
                  <input type="file" accept=".pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
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

          <section className="custom-section-card quick-ai-section">
            <h3>AI Expert (Optional)</h3>
            <label className="quick-ai-toggle">
              <input type="checkbox" checked={enableAi} onChange={(e) => setEnableAi(e.target.checked)} />
              <span className="quick-ai-toggle-box" />
              <span className="quick-ai-toggle-text">Enable AI analysis and expert summary</span>
              <span className={`quick-ai-pill ${enableAi ? "active" : ""}`}>{enableAi ? "Enabled" : "Disabled"}</span>
            </label>
            {enableAi && (
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
            )}
          </section>

          <button type="button" className="custom-btn-primary" onClick={() => void onRun()} disabled={running}>
            {running ? "Running..." : "Start Sequence Design"}
          </button>
        </>
      }
      right={
        <QuickToolResultPanel
          title="Sequence Design Result"
          resultPayload={resultPayload}
          aiSummary={aiSummary}
          error={error}
        />
      }
    />
  );
}
