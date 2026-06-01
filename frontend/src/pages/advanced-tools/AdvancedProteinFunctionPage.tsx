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
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";
import { COMMON_STRINGS } from "../../lib/commonStrings";

const STRINGS = {
  en: {
    ...COMMON_STRINGS.en,
    title: "Protein Function",
    subtitle: "Predict protein functions across selected datasets.",
    modelTaskSection: "Model and Task",
    modelLabel: "Model",
    taskLabel: "Task",
    datasetsLabel: "Datasets (Multi-select)",
    selectedCount: (n: number) => `${n} selected`,
    selectAll: "Select All",
    clear: "Clear",
    datasetSelected: "selected",
    inputSection: "Input",
    pasteFastaSequence: "Paste FASTA / sequence",
    pasteFastaPlaceholder: "Paste sequence or FASTA content...",
    onlineLimitNote: (n: number) => `Online mode supports up to ${n} FASTA sequences per run.`,
    aiOn: "AI insight will be generated and attached to the result panel.",
    aiOff: "Enable this to generate expert interpretation after prediction.",
    startBtn: "Start Function Prediction",
    resultTitle: "Protein Function Result"
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "蛋白功能",
    subtitle: "在选定的数据集上进行蛋白功能预测。",
    modelTaskSection: "模型与任务",
    modelLabel: "模型",
    taskLabel: "任务",
    datasetsLabel: "数据集（多选）",
    selectedCount: (n: number) => `已选 ${n} 项`,
    selectAll: "全选",
    clear: "清空",
    datasetSelected: "已选",
    inputSection: "输入",
    pasteFastaSequence: "粘贴 FASTA / 序列",
    pasteFastaPlaceholder: "粘贴序列或 FASTA 内容…",
    onlineLimitNote: (n: number) => `在线模式每次运行最多支持 ${n} 条 FASTA 序列。`,
    aiOn: "AI 专家解读将生成并附加在结果面板中。",
    aiOff: "开启后，预测完成会自动生成专家解读。",
    startBtn: "开始功能预测",
    resultTitle: "蛋白功能预测结果"
  }
};

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
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
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
  const [progressMessage, setProgressMessage] = useState("");

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
      setError(err instanceof Error ? err.message : t.uploadFailed);
    }
  }

  async function onRun() {
    setError("");
    setResultPayload(null);
    setProgress(0);
    setProgressMessage(t.preparingTask);
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
      setProgressMessage(t.predictionDone);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.runFailed);
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
      setError(err instanceof Error ? err.message : t.loadExampleFailed);
    }
  }

  return (
    <AdvancedToolsLayout
      title={t.title}
      subtitle={t.subtitle}
      running={running}
      progress={progress}
      progressMessage={progressMessage || t.idle}
      left={
        <>
          <section className="custom-section-card">
            <h3>{t.modelTaskSection}</h3>
            <label className="left-controls">
              {t.modelLabel}
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {meta.model_mapping_function.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label className="left-controls">
              {t.taskLabel}
              <select value={task} onChange={(e) => setTask(e.target.value)}>
                {Object.keys(meta.dataset_mapping_function).map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <div className="left-controls">
              <span>{t.datasetsLabel}</span>
              <div className="advanced-dataset-toolbar">
                <span className="advanced-dataset-count">{t.selectedCount(selectedDatasets.length)}</span>
                <div className="advanced-dataset-actions">
                  <button
                    type="button"
                    className="custom-btn-secondary"
                    onClick={() => setSelectedDatasets(datasetOptions)}
                    disabled={datasetOptions.length === 0}
                  >
                    {t.selectAll}
                  </button>
                  <button
                    type="button"
                    className="custom-btn-secondary"
                    onClick={() => setSelectedDatasets([])}
                    disabled={selectedDatasets.length === 0}
                  >
                    {t.clear}
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
                      {checked && <span className="advanced-dataset-item-status">{t.datasetSelected}</span>}
                      <span className="advanced-dataset-item-label">{item}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </section>

          <section className="custom-section-card">
            <h3>{t.inputSection}</h3>
            <label className="left-controls">
              {t.pasteFastaSequence}
              <textarea
                rows={6}
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder={t.pasteFastaPlaceholder}
              />
            </label>
            {meta.online_limit_enabled && (
              <p className="advanced-ai-note">
                {t.onlineLimitNote(meta.online_fasta_limit ?? 50)}
              </p>
            )}
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  {t.selectFile}
                  <input type="file" accept=".fasta,.fa,.txt" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
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

          <section className="custom-section-card advanced-ai-section">
            <h3>{t.aiExpert}</h3>
            <label className="advanced-ai-toggle">
              <input type="checkbox" checked={enableAi} onChange={(e) => setEnableAi(e.target.checked)} />
              <span className="advanced-ai-toggle-box" />
              <span className="advanced-ai-toggle-text">{t.enableAi}</span>
              <span className={`advanced-ai-pill ${enableAi ? "active" : ""}`}>{enableAi ? t.enabled : t.disabled}</span>
            </label>
            <p className="advanced-ai-note">
              {enableAi ? t.aiOn : t.aiOff}
            </p>
            {enableAi && (
              <div className="advanced-ai-fields">
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
            {running ? t.runningBtn : t.startBtn}
          </button>
        </>
      }
      right={
        <AdvancedResultPanel
          title={t.resultTitle}
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
