import { useEffect, useState } from "react";
import {
  fetchQuickToolsMeta,
  loadQuickToolDefaultExample,
  normalizePastedFastaForDisplay,
  requestQuickToolAiSummary,
  runMutationToolStream,
  type QuickToolsMeta,
  validateFastaWithHeader,
  uploadQuickToolFile
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
    title: "Directed Evolution",
    subtitle: "Score single-point mutations from FASTA or PDB inputs.",
    deNote: "Directed Evolution supports one protein per run (sequence or structure).",
    resultTitle: "Directed Evolution Result",
    heatmapHint: "Mutation heatmap files are returned as artifacts. Download the result package for full interactive plot."
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "定向进化",
    subtitle: "基于 FASTA 或 PDB 输入对单点突变进行评分。",
    deNote: "定向进化每次运行支持一条蛋白（序列或结构）。",
    resultTitle: "定向进化结果",
    heatmapHint: "突变热图以文件形式返回，下载结果包可获得完整可交互图。"
  }
};

const DEFAULT_META: QuickToolsMeta = {
  dataset_mapping_zero_shot: ["Activity", "Binding", "Expression", "Organismal Fitness", "Stability"],
  model_mapping_zero_shot: ["ESM2-650M", "ESM-IF1"],
  dataset_mapping_function: [],
  residue_mapping_function: [],
  protein_properties_function: [],
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"]
};

type DirectedEvolutionPageProps = {
  workspaceEnabled?: boolean;
};

export function DirectedEvolutionPage({ workspaceEnabled = false }: DirectedEvolutionPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
  const [meta, setMeta] = useState<QuickToolsMeta>(DEFAULT_META);
  const [functionTask, setFunctionTask] = useState(DEFAULT_META.dataset_mapping_zero_shot[0]);
  const [sequence, setSequence] = useState("");
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
      if (loaded.dataset_mapping_zero_shot.length > 0) setFunctionTask(loaded.dataset_mapping_zero_shot[0]);
      if (loaded.llm_models.length > 0) setLlmProvider(loaded.llm_models[0]);
    })();
  }, []);

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
        setSequence(normalizePastedFastaForDisplay(content));
      } else {
        setSequence("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.uploadFailed);
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadQuickToolDefaultExample("fasta");
      setUploadedPath(data.file_path);
      setUploadedSuffix(data.suffix);
      setSequence(data.content || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : t.loadExampleFailed);
    }
  }

  async function onRun() {
    setError("");
    setAiSummary("");
    setRunning(true);
    setProgress(0);
    setProgressMessage(t.preparingTask);
    try {
      if (!uploadedPath && !sequence.trim()) {
        throw new Error(t.pleaseProvideInput);
      }
      const runArgs = {
        uploadedPath,
        uploadedSuffix,
        sequence,
        modelName: uploadedSuffix === ".pdb" ? "ESM-IF1" : "ESM2-650M"
      };
      const payload = await runMutationToolStream(runArgs, (evt) => {
        setProgress(evt.progress);
        setProgressMessage(evt.message);
      });
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage(t.predictionDone);
      if (enableAi) {
        const ai = await requestQuickToolAiSummary({
          tool: "mutation",
          task: functionTask,
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
              {t.selectProteinFunction}
              <select value={functionTask} onChange={(e) => setFunctionTask(e.target.value)}>
                {meta.dataset_mapping_zero_shot.map((item) => (
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
              {t.pasteSequence}
              <textarea
                rows={7}
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder={t.pasteSeqPlaceholder}
              />
            </label>
            <p className="quick-ai-note">{t.deNote}</p>
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  {t.selectFile}
                  <input type="file" accept=".fasta,.fa,.pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
                  acceptedCategories={["sequence", "structure"]}
                  buttonLabel={t.fromWorkspace}
                  onPick={(picked) => {
                    const selected = picked[0];
                    if (!selected) return;
                    setUploadedPath(selected.storage_path);
                    setUploadedSuffix(selected.suffix);
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
          enableHeatmapTab
          heatmapHint={t.heatmapHint}
        />
      }
    />
  );
}
