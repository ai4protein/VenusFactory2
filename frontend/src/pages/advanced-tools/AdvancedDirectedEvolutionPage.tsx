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
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";
import { COMMON_STRINGS } from "../../lib/commonStrings";

const STRINGS = {
  en: {
    ...COMMON_STRINGS.en,
    title: "Directed Evolution",
    subtitle: "Run saturation mutagenesis scoring with sequence or structure models.",
    predictionMode: "Prediction Mode",
    predictionModeSwitchAria: "Prediction mode switch",
    sequenceModel: "Sequence Model",
    structureModel: "Structure Model",
    sequenceModeHint: "Sequence mode: paste FASTA/sequence or upload .fasta/.fa",
    structureModeHint: "Structure mode: upload .pdb for structure-based scoring",
    selectModel: "Select Model",
    inputSection: "Input",
    pasteSequenceFasta: "Paste Sequence / FASTA",
    pasteSequenceFastaPlaceholder: "Paste raw sequence or FASTA content...",
    deNote: "Directed Evolution supports one protein per run (sequence or structure).",
    useExamplePdb: "Use Example PDB",
    aiOn: "AI insight will be generated and attached to the result panel.",
    aiOff: "Enable this to generate expert interpretation after prediction.",
    startBtn: "Start Directed Evolution",
    resultTitle: "Directed Evolution Result"
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "定向进化",
    subtitle: "使用序列或结构模型进行饱和突变评分。",
    predictionMode: "预测模式",
    predictionModeSwitchAria: "预测模式切换",
    sequenceModel: "序列模型",
    structureModel: "结构模型",
    sequenceModeHint: "序列模式：粘贴 FASTA / 序列，或上传 .fasta / .fa 文件",
    structureModeHint: "结构模式：上传 .pdb 文件，基于结构进行评分",
    selectModel: "选择模型",
    inputSection: "输入",
    pasteSequenceFasta: "粘贴序列 / FASTA",
    pasteSequenceFastaPlaceholder: "粘贴原始序列或 FASTA 内容…",
    deNote: "定向进化每次运行支持一条蛋白（序列或结构）。",
    useExamplePdb: "使用示例 PDB",
    aiOn: "AI 专家解读将生成并附加在结果面板中。",
    aiOff: "开启后，预测完成会自动生成专家解读。",
    startBtn: "开始定向进化",
    resultTitle: "定向进化结果"
  }
};

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

type AdvancedDirectedEvolutionPageProps = {
  workspaceEnabled: boolean;
};

export function AdvancedDirectedEvolutionPage({ workspaceEnabled }: AdvancedDirectedEvolutionPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
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
  const [progressMessage, setProgressMessage] = useState("");

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
      setError(err instanceof Error ? err.message : t.uploadFailed);
    }
  }

  async function onRun() {
    setError("");
    setProgress(0);
    setProgressMessage(t.preparingTask);
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
      const kind = inputMode === "structure" ? "pdb" : "fasta";
      const data = await loadAdvancedDefaultExample(kind);
      setUploadedPath(data.file_path);
      if (kind === "fasta") {
        setSequence(data.content || "");
      } else {
        setSequence("");
      }
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
            <div className="advanced-section-caption">{t.predictionMode}</div>
            <div className="advanced-mode-row">
              <SegmentedSwitch
                value={inputMode}
                onChange={setInputMode}
                ariaLabel={t.predictionModeSwitchAria}
                className="advanced-mode-segment-switch"
                options={[
                  { value: "sequence", label: t.sequenceModel },
                  { value: "structure", label: t.structureModel }
                ]}
              />
            </div>
            <div className="advanced-mode-hint">
              {inputMode === "sequence" ? t.sequenceModeHint : t.structureModeHint}
            </div>
            <label className="left-controls advanced-full-row">
              {t.selectModel}
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
            <h3>{t.inputSection}</h3>
            {inputMode === "sequence" && (
              <label className="left-controls">
                {t.pasteSequenceFasta}
                <textarea
                  rows={6}
                  value={sequence}
                  onChange={(e) => setSequence(e.target.value)}
                  placeholder={t.pasteSequenceFastaPlaceholder}
                />
              </label>
            )}
            <p className="advanced-ai-note">{t.deNote}</p>
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  {t.selectFile}
                  <input
                    type="file"
                    accept={inputMode === "sequence" ? ".fasta,.fa,.txt" : ".pdb"}
                    onChange={(e) => void onUpload(e.target.files?.[0] || null)}
                  />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
                  acceptedCategories={inputMode === "sequence" ? ["sequence"] : ["structure"]}
                  buttonLabel={t.fromWorkspace}
                  onPick={(picked) => {
                    const selected = picked[0];
                    if (!selected) return;
                    setUploadedPath(selected.storage_path);
                  }}
                />
              </div>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                {inputMode === "structure" ? t.useExamplePdb : t.useExampleFasta}
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
          enableHeatmapTab
        />
      }
    />
  );
}
