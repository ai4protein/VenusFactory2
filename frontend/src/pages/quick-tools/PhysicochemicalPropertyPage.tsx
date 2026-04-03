import { useEffect, useState } from "react";
import {
  fetchQuickToolsMeta,
  getDownloadUrl,
  loadQuickToolDefaultExample,
  normalizePastedFastaForDisplay,
  requestQuickToolAiSummary,
  runPropertiesToolStream,
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
  dataset_mapping_function: [],
  residue_mapping_function: [],
  protein_properties_function: [
    "Physical and chemical properties",
    "Relative solvent accessible surface area (PDB only)",
    "SASA value (PDB only)",
    "Secondary structure (PDB only)"
  ],
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"]
};

export function PhysicochemicalPropertyPage() {
  const [meta, setMeta] = useState<QuickToolsMeta>(DEFAULT_META);
  const [task, setTask] = useState(DEFAULT_META.protein_properties_function[0]);
  const [chainId, setChainId] = useState("A");
  const [chainOptions, setChainOptions] = useState<string[]>(["A"]);
  const [pasteSequence, setPasteSequence] = useState("");
  const [uploadedPath, setUploadedPath] = useState("");
  const [uploadedSuffix, setUploadedSuffix] = useState("");
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
      if (loaded.protein_properties_function.length > 0) setTask(loaded.protein_properties_function[0]);
      if (loaded.llm_models.length > 0) setLlmProvider(loaded.llm_models[0]);
    })();
  }, []);
  const requiresPdb = task.includes("(PDB only)");

  useEffect(() => {
    if (!requiresPdb) {
      setChainOptions(["A"]);
      setChainId("A");
    }
  }, [requiresPdb]);

  useEffect(() => {
    let cancelled = false;
    if (!requiresPdb || uploadedSuffix !== ".pdb" || !uploadedPath) {
      return () => {
        cancelled = true;
      };
    }

    void (async () => {
      try {
        const res = await fetch(getDownloadUrl(uploadedPath));
        if (!res.ok) throw new Error(`Failed to read PDB (${res.status})`);
        const text = await res.text();
        const parsedChains = extractPdbChains(text);
        if (cancelled) return;
        setChainOptions(parsedChains);
        setChainId((prev) => (parsedChains.includes(prev) ? prev : parsedChains[0]));
      } catch {
        if (cancelled) return;
        setChainOptions(["A"]);
        setChainId("A");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [requiresPdb, uploadedPath, uploadedSuffix]);

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const lowerName = file.name.toLowerCase();
      if (lowerName.endsWith(".fasta") || lowerName.endsWith(".fa")) {
        validateFastaWithHeader(await file.text());
      }
      const data = await uploadQuickToolFile(file);
      setUploadedPath(data.file_path);
      setUploadedSuffix(data.suffix);
      if (data.suffix === ".fasta" || data.suffix === ".fa") {
        const content = await file.text();
        setPasteSequence(normalizePastedFastaForDisplay(content));
      } else {
        setPasteSequence("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const kind = task.includes("(PDB only)") ? "pdb" : "fasta";
      const data = await loadQuickToolDefaultExample(kind);
      setUploadedPath(data.file_path);
      setUploadedSuffix(data.suffix);
      if (data.suffix === ".fasta" || data.suffix === ".fa") {
        setPasteSequence(data.content || "");
      } else {
        setPasteSequence("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load example.");
    }
  }

  async function resolveInputFile(): Promise<{ filePath: string; suffix: string }> {
    if (uploadedPath) {
      return { filePath: uploadedPath, suffix: uploadedSuffix };
    }

    if (task.includes("(PDB only)")) {
      throw new Error("Please upload a PDB file for this task.");
    }

    if (!pasteSequence.trim()) {
      throw new Error("Please upload a file or paste sequence.");
    }

    const uploaded = await uploadSequenceAsFasta(pasteSequence);
    setUploadedPath(uploaded.file_path);
    setUploadedSuffix(uploaded.suffix);
    return { filePath: uploaded.file_path, suffix: uploaded.suffix };
  }

  function validateInput(suffix: string) {
    if (task.includes("(PDB only)") && suffix !== ".pdb") {
      throw new Error("Current task requires .pdb file.");
    }
    if (!task.includes("(PDB only)") && suffix === ".pdb") {
      throw new Error("Physical and chemical properties expects FASTA input.");
    }
  }

  async function onRun() {
    setError("");
    setAiSummary("");
    setRunning(true);
    setProgress(0);
    setProgressMessage("Preparing task...");
    try {
      const { filePath, suffix } = await resolveInputFile();
      validateInput(suffix);
      const payload = await runPropertiesToolStream({
        task,
        uploadedPath: filePath,
        chainId
      }, (evt) => {
        setProgress(evt.progress);
        setProgressMessage(evt.message);
      });
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage("Prediction completed");
      if (enableAi) {
        const ai = await requestQuickToolAiSummary({
          tool: "properties",
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
      title="Physicochemical Property"
      subtitle="Calculate protein properties from FASTA or PDB inputs."
      running={running}
      progress={progress}
      progressMessage={progressMessage}
      left={
        <>
          <section className="custom-section-card">
            <h3>Task Configuration</h3>
            <label className="left-controls">
              Select Properties of Protein
              <select value={task} onChange={(e) => setTask(e.target.value)}>
                {meta.protein_properties_function.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            {requiresPdb && uploadedSuffix === ".pdb" && (
              <label className="left-controls">
                PDB Chain
                <select value={chainId} onChange={(e) => setChainId(e.target.value)}>
                  {chainOptions.map((chain) => (
                    <option key={chain} value={chain}>
                      {chain}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </section>

          <section className="custom-section-card">
            <h3>Data Input</h3>
            <label className="left-controls">
              Paste Sequence
              <textarea
                rows={6}
                value={pasteSequence}
                onChange={(e) => setPasteSequence(e.target.value)}
                placeholder="Paste FASTA content with >header for non-PDB tasks..."
                disabled={task.includes("(PDB only)")}
              />
            </label>
            {meta.online_limit_enabled && !task.includes("(PDB only)") && (
              <p className="quick-ai-note">
                Online mode supports up to {meta.online_fasta_limit ?? 50} FASTA sequences per run.
              </p>
            )}
            <div className="custom-file-example-row">
              <label className="left-controls custom-file-picker-field">
                Select File
                <input type="file" accept=".fasta,.fa,.pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
              </label>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                {task.includes("(PDB only)") ? "Use Example PDB" : "Use Example FASTA"}
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
          title="Physicochemical Property Result"
          resultPayload={resultPayload}
          aiSummary={aiSummary}
          error={error}
          enableHeatmapTab={false}
        />
      }
    />
  );
}

function extractPdbChains(content: string): string[] {
  const chains = new Set<string>();
  const lines = content.split(/\r?\n/);
  for (const line of lines) {
    if (!(line.startsWith("ATOM") || line.startsWith("HETATM"))) continue;
    const rawChain = line.length > 21 ? line[21].trim() : "";
    chains.add(rawChain || "A");
  }
  return chains.size > 0 ? Array.from(chains) : ["A"];
}
