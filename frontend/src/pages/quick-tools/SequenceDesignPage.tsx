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
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";
import { COMMON_STRINGS } from "../../lib/commonStrings";
type ModelFamily = "soluble" | "vanilla" | "ca";

const DEFAULT_MODEL_NAME = "v_48_020";
const DEFAULT_BACKBONE_NOISE = 0.2;

const STRINGS = {
  en: {
    ...COMMON_STRINGS.en,
    title: "Sequence Design",
    subtitle: "Design protein sequences from a structure with simple, biology-friendly controls.",
    modelFamily: "Model Family",
    modelSoluble: "Soluble (recommended for protein discovery and design)",
    modelVanilla: "Vanilla (recommended for membrane proteins)",
    modelCa: "CA (only for C-alpha coarse-grained coordinates)",
    quickModeBefore: "Quick mode uses ",
    quickModeAfter: (noise: string) => ` by default (backbone_noise ${noise}A).`,
    designedChains: "Designed Chains (optional)",
    designedChainsPlaceholder: "A,B (empty means all chains)",
    fixedResidues: "Fixed Residues (optional)",
    fixedResiduesPlaceholder: "A12,C13 or A:12,13;B:5-8",
    numSequences: "Number of sequences",
    designDiversity: "Design Diversity",
    diversityLow: "Low (conservative)",
    diversityMedium: "Medium (balanced)",
    diversityHigh: "High (exploratory)",
    onlineLimitNote: (n: number) => `Online mode supports up to ${n} designed sequences per run.`,
    structureInput: "Structure Input",
    selectPdb: "Select PDB File",
    useExamplePdb: "Use Example PDB",
    selected: "Selected:",
    startSeqDesign: "Start Sequence Design",
    preparingSeqDesign: "Preparing sequence design...",
    seqDesignDone: "Sequence design completed",
    pdbOnlyError: "Sequence Design only supports PDB structure input.",
    pleaseUploadPdb: "Please upload or pick a PDB file first.",
    onlineLimitExceeded: (n: number) => `Online mode supports up to ${n} designed sequences per run.`,
    resultTitle: "Sequence Design Result"
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "序列设计",
    subtitle: "基于蛋白结构进行序列设计，提供贴近生物学家习惯的简洁参数。",
    modelFamily: "模型家族",
    modelSoluble: "Soluble（推荐用于蛋白挖掘与设计）",
    modelVanilla: "Vanilla（推荐用于膜蛋白）",
    modelCa: "CA（仅适用于 C-alpha 粗粒度坐标）",
    quickModeBefore: "快速模式默认使用 ",
    quickModeAfter: (noise: string) => `（backbone_noise ${noise}A）。`,
    designedChains: "待设计链（可选）",
    designedChainsPlaceholder: "A,B（留空表示所有链）",
    fixedResidues: "固定残基（可选）",
    fixedResiduesPlaceholder: "A12,C13 或 A:12,13;B:5-8",
    numSequences: "生成序列数",
    designDiversity: "设计多样性",
    diversityLow: "低（保守）",
    diversityMedium: "中（平衡）",
    diversityHigh: "高（探索）",
    onlineLimitNote: (n: number) => `在线模式每次运行最多支持设计 ${n} 条序列。`,
    structureInput: "结构输入",
    selectPdb: "选择 PDB 文件",
    useExamplePdb: "使用示例 PDB",
    selected: "已选择：",
    startSeqDesign: "开始序列设计",
    preparingSeqDesign: "准备序列设计中…",
    seqDesignDone: "序列设计完成",
    pdbOnlyError: "序列设计仅支持 PDB 结构输入。",
    pleaseUploadPdb: "请先上传或选择一个 PDB 文件。",
    onlineLimitExceeded: (n: number) => `在线模式每次运行最多支持设计 ${n} 条序列。`,
    resultTitle: "序列设计结果"
  }
};

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
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
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
  const [progressMessage, setProgressMessage] = useState("");
  const onlineSequenceDesignLimit = meta.online_limit_enabled ? Math.max(1, meta.online_sequence_design_limit ?? 50) : 512;
  const baseSequenceOptions = [4, 8, 16, 32];
  const sequenceOptions = (
    meta.online_limit_enabled
      ? Array.from(
          new Set([...baseSequenceOptions.filter((count) => count <= onlineSequenceDesignLimit), onlineSequenceDesignLimit])
        )
      : baseSequenceOptions
  ).sort((a, b) => a - b);

  useEffect(() => {
    void (async () => {
      const loaded = await fetchQuickToolsMeta();
      setMeta(loaded);
      if (loaded.llm_models.length > 0) setLlmProvider(loaded.llm_models[0]);
    })();
  }, []);

  useEffect(() => {
    if (numSequences > onlineSequenceDesignLimit) {
      setNumSequences(onlineSequenceDesignLimit);
    }
  }, [numSequences, onlineSequenceDesignLimit]);

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const data = await uploadQuickToolFile(file);
      if (data.suffix !== ".pdb") {
        throw new Error(t.pdbOnlyError);
      }
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.uploadFailed);
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadQuickToolDefaultExample("pdb");
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.loadExampleFailed);
    }
  }

  async function onRun() {
    setError("");
    setAiSummary("");
    setRunning(true);
    setProgress(0);
    setProgressMessage(t.preparingSeqDesign);
    try {
      if (!uploadedPath) {
        throw new Error(t.pleaseUploadPdb);
      }
      if (meta.online_limit_enabled && numSequences > onlineSequenceDesignLimit) {
        throw new Error(t.onlineLimitExceeded(onlineSequenceDesignLimit));
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
      setProgressMessage(t.seqDesignDone);
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
              {t.modelFamily}
              <select value={modelFamily} onChange={(e) => setModelFamily(e.target.value as ModelFamily)}>
                <option value="soluble">{t.modelSoluble}</option>
                <option value="vanilla">{t.modelVanilla}</option>
                <option value="ca">{t.modelCa}</option>
              </select>
            </label>
            <p className="quick-ai-note">
              {t.quickModeBefore}
              <code>{DEFAULT_MODEL_NAME}</code>
              {t.quickModeAfter(DEFAULT_BACKBONE_NOISE.toFixed(2))}
            </p>
            <label className="left-controls quick-seq-match-input">
              {t.designedChains}
              <input
                value={designedChainsText}
                onChange={(e) => setDesignedChainsText(e.target.value)}
                placeholder={t.designedChainsPlaceholder}
              />
            </label>
            <label className="left-controls quick-seq-match-input">
              {t.fixedResidues}
              <input
                value={fixedResiduesText}
                onChange={(e) => setFixedResiduesText(e.target.value)}
                placeholder={t.fixedResiduesPlaceholder}
              />
            </label>
            <div style={{ display: "flex", gap: "12px", alignItems: "stretch" }}>
              <label className="left-controls" style={{ flex: 1, minWidth: 0 }}>
                {t.numSequences}
                <select value={numSequences} onChange={(e) => setNumSequences(Number(e.target.value))}>
                  {sequenceOptions.map((count) => (
                    <option key={count} value={count}>
                      {count}
                    </option>
                  ))}
                </select>
              </label>
              <label className="left-controls" style={{ flex: 1, minWidth: 0 }}>
                {t.designDiversity}
                <select value={diversity} onChange={(e) => setDiversity(e.target.value as "low" | "medium" | "high")}>
                  <option value="low">{t.diversityLow}</option>
                  <option value="medium">{t.diversityMedium}</option>
                  <option value="high">{t.diversityHigh}</option>
                </select>
              </label>
            </div>
            {meta.online_limit_enabled && (
              <p className="quick-ai-note">{t.onlineLimitNote(onlineSequenceDesignLimit)}</p>
            )}
          </section>

          <section className="custom-section-card">
            <h3>{t.structureInput}</h3>
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  {t.selectPdb}
                  <input type="file" accept=".pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
                  acceptedCategories={["structure"]}
                  buttonLabel={t.fromWorkspace}
                  onPick={(picked) => {
                    const selected = picked[0];
                    if (!selected || selected.suffix !== ".pdb") return;
                    setUploadedPath(selected.storage_path);
                  }}
                />
              </div>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                {t.useExamplePdb}
              </button>
            </div>
            {uploadedPath && <div className="report-preview">{t.selected} {uploadedPath}</div>}
          </section>

          <section className="custom-section-card quick-ai-section">
            <h3>{t.aiExpert}</h3>
            <label className="quick-ai-toggle">
              <input type="checkbox" checked={enableAi} onChange={(e) => setEnableAi(e.target.checked)} />
              <span className="quick-ai-toggle-box" />
              <span className="quick-ai-toggle-text">{t.enableAi}</span>
              <span className={`quick-ai-pill ${enableAi ? "active" : ""}`}>{enableAi ? t.enabled : t.disabled}</span>
            </label>
            {enableAi && (
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
            )}
          </section>

          <button type="button" className="custom-btn-primary" onClick={() => void onRun()} disabled={running}>
            {running ? t.runningBtn : t.startSeqDesign}
          </button>
        </>
      }
      right={
        <QuickToolResultPanel
          title={t.resultTitle}
          resultPayload={resultPayload}
          aiSummary={aiSummary}
          error={error}
        />
      }
    />
  );
}
