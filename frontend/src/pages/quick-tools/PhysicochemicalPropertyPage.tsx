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
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";
import { COMMON_STRINGS } from "../../lib/commonStrings";

const STRINGS = {
  en: {
    ...COMMON_STRINGS.en,
    title: "Physicochemical Property",
    subtitle: "Calculate protein properties from FASTA or PDB inputs.",
    selectProperties: "Select Properties of Protein",
    pdbChain: "PDB Chain",
    pasteSeqPlaceholderPdb: "Paste FASTA content with >header for non-PDB tasks...",
    onlineFastaLimit: (n: number) => `Online mode supports up to ${n} FASTA sequences per run.`,
    useExamplePdb: "Use Example PDB",
    pleaseUploadPdb: "Please upload a PDB file for this task.",
    pleaseUploadOrPaste: "Please upload a file or paste sequence.",
    pdbRequiredError: "Current task requires .pdb file.",
    fastaRequiredError: "Physical and chemical properties expects FASTA input.",
    resultTitle: "Physicochemical Property Result"
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "理化性质",
    subtitle: "基于 FASTA 或 PDB 输入计算蛋白质理化性质。",
    selectProperties: "选择蛋白性质",
    pdbChain: "PDB 链",
    pasteSeqPlaceholderPdb: "对于非 PDB 任务，请粘贴包含 > 开头 header 的 FASTA 内容…",
    onlineFastaLimit: (n: number) => `在线模式每次运行最多支持 ${n} 条 FASTA 序列。`,
    useExamplePdb: "使用示例 PDB",
    pleaseUploadPdb: "当前任务需要上传 PDB 文件。",
    pleaseUploadOrPaste: "请上传文件或粘贴序列。",
    pdbRequiredError: "当前任务需要 .pdb 文件。",
    fastaRequiredError: "理化性质计算需要 FASTA 输入。",
    resultTitle: "理化性质计算结果"
  }
};

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

type PhysicochemicalPropertyPageProps = {
  workspaceEnabled?: boolean;
};

export function PhysicochemicalPropertyPage({ workspaceEnabled = false }: PhysicochemicalPropertyPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
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
  const [progressMessage, setProgressMessage] = useState("");

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
      setError(err instanceof Error ? err.message : t.uploadFailed);
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
      setError(err instanceof Error ? err.message : t.loadExampleFailed);
    }
  }

  async function resolveInputFile(): Promise<{ filePath: string; suffix: string }> {
    if (uploadedPath) {
      return { filePath: uploadedPath, suffix: uploadedSuffix };
    }

    if (task.includes("(PDB only)")) {
      throw new Error(t.pleaseUploadPdb);
    }

    if (!pasteSequence.trim()) {
      throw new Error(t.pleaseUploadOrPaste);
    }

    const uploaded = await uploadSequenceAsFasta(pasteSequence);
    setUploadedPath(uploaded.file_path);
    setUploadedSuffix(uploaded.suffix);
    return { filePath: uploaded.file_path, suffix: uploaded.suffix };
  }

  function validateInput(suffix: string) {
    if (task.includes("(PDB only)") && suffix !== ".pdb") {
      throw new Error(t.pdbRequiredError);
    }
    if (!task.includes("(PDB only)") && suffix === ".pdb") {
      throw new Error(t.fastaRequiredError);
    }
  }

  async function onRun() {
    setError("");
    setAiSummary("");
    setRunning(true);
    setProgress(0);
    setProgressMessage(t.preparingTask);
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
      setProgressMessage(t.predictionDone);
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
      setError(err instanceof Error ? err.message : t.runFailed);
    } finally {
      setRunning(false);
    }
  }

  return (
    <QuickToolsLayout
      title={t.title}
      subtitle={t.subtitle}
      running={running}
      progress={progress}
      progressMessage={progressMessage || t.idle}
      left={
        <>
          <section className="custom-section-card">
            <h3>{t.taskConfig}</h3>
            <label className="left-controls">
              {t.selectProperties}
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
                {t.pdbChain}
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
            <h3>{t.dataInput}</h3>
            <label className="left-controls">
              {t.pasteSequence}
              <textarea
                rows={6}
                value={pasteSequence}
                onChange={(e) => setPasteSequence(e.target.value)}
                placeholder={t.pasteSeqPlaceholderPdb}
                disabled={task.includes("(PDB only)")}
              />
            </label>
            {meta.online_limit_enabled && !task.includes("(PDB only)") && (
              <p className="quick-ai-note">
                {t.onlineFastaLimit(meta.online_fasta_limit ?? 50)}
              </p>
            )}
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  {t.selectFile}
                  <input type="file" accept=".fasta,.fa,.pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
                  acceptedCategories={task.includes("(PDB only)") ? ["structure"] : ["sequence"]}
                  buttonLabel={t.fromWorkspace}
                  onPick={(picked) => {
                    const selected = picked[0];
                    if (!selected) return;
                    setUploadedPath(selected.storage_path);
                    setUploadedSuffix(selected.suffix);
                    setPasteSequence("");
                  }}
                />
              </div>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                {task.includes("(PDB only)") ? t.useExamplePdb : t.useExampleFasta}
              </button>
            </div>
            {uploadedPath && <div className="report-preview">{t.uploaded} {uploadedPath}</div>}
          </section>

          <section className="custom-section-card quick-ai-section">
            <h3>{t.aiExpert}</h3>
            <label className="quick-ai-toggle">
              <input type="checkbox" checked={enableAi} onChange={(e) => setEnableAi(e.target.checked)} />
              <span className="quick-ai-toggle-box" />
              <span className="quick-ai-toggle-text">{t.enableAi}</span>
              <span className={`quick-ai-pill ${enableAi ? "active" : ""}`}>{enableAi ? t.enabled : t.disabled}</span>
            </label>
            <p className="quick-ai-note">{enableAi ? t.aiOn : t.aiOff}</p>
            {enableAi && (
              <div className="quick-ai-fields">
                <label className="left-controls">
                  {t.llmProvider}
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
            {running ? t.runningBtn : t.startPrediction}
          </button>
        </>
      }
      right={
        <QuickToolResultPanel
          title={t.resultTitle}
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
