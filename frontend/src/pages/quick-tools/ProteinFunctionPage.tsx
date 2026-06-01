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
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";
import { COMMON_STRINGS } from "../../lib/commonStrings";

const STRINGS = {
  en: {
    ...COMMON_STRINGS.en,
    title: "Protein Function",
    subtitle: "Predict protein-level function from FASTA sequences.",
    selectTask: "Select Task",
    pasteFasta: "Paste FASTA/sequence",
    onlineFastaLimit: (n: number) => `Online mode supports up to ${n} FASTA sequences per run.`,
    pleaseProvideFasta: "Please upload a FASTA file or paste sequence.",
    resultTitle: "Protein Function Result"
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "蛋白质功能",
    subtitle: "基于 FASTA 序列预测蛋白质层级的功能。",
    selectTask: "选择任务",
    pasteFasta: "粘贴 FASTA / 序列",
    onlineFastaLimit: (n: number) => `在线模式每次运行最多支持 ${n} 条 FASTA 序列。`,
    pleaseProvideFasta: "请上传 FASTA 文件或粘贴序列。",
    resultTitle: "蛋白质功能预测结果"
  }
};

const DEFAULT_META: QuickToolsMeta = {
  dataset_mapping_zero_shot: [],
  model_mapping_zero_shot: [],
  dataset_mapping_function: ["Solubility"],
  residue_mapping_function: [],
  protein_properties_function: [],
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"]
};

type ProteinFunctionPageProps = {
  workspaceEnabled?: boolean;
};

export function ProteinFunctionPage({ workspaceEnabled = false }: ProteinFunctionPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
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
  const [progressMessage, setProgressMessage] = useState("");
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
      setError(err instanceof Error ? err.message : t.uploadFailed);
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadQuickToolDefaultExample("fasta");
      setUploadedPath(data.file_path);
      setSequence(data.content || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : t.loadExampleFailed);
    }
  }

  async function resolveFastaFile(): Promise<string> {
    if (uploadedPath) return uploadedPath;
    if (sequence.trim()) {
      const uploaded = await uploadSequenceAsFasta(sequence);
      setUploadedPath(uploaded.file_path);
      return uploaded.file_path;
    }
    throw new Error(t.pleaseProvideFasta);
  }

  async function onRun() {
    setError("");
    setAiSummary("");
    setRunning(true);
    setProgress(0);
    setProgressMessage(t.preparingTask);
    try {
      const fastaPath = await resolveFastaFile();
      const payload = await runProteinFunctionToolStream({ fastaFile: fastaPath, task }, (evt) => {
        setProgress(evt.progress);
        setProgressMessage(evt.message);
      });
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage(t.predictionDone);
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
              {t.selectTask}
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
            <h3>{t.dataInput}</h3>
            <label className="left-controls">
              {t.pasteFasta}
              <textarea
                rows={7}
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder={t.pasteSeqPlaceholder}
              />
            </label>
            {meta.online_limit_enabled && (
              <p className="quick-ai-note">
                {t.onlineFastaLimit(meta.online_fasta_limit ?? 50)}
              </p>
            )}
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  {t.selectFile}
                  <input type="file" accept=".fasta,.fa" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
                  acceptedCategories={["sequence"]}
                  buttonLabel={t.fromWorkspace}
                  onPick={(picked) => {
                    const selected = picked[0];
                    if (!selected) return;
                    setUploadedPath(selected.storage_path);
                    setSequence("");
                  }}
                />
              </div>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                {t.useExampleFasta}
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
