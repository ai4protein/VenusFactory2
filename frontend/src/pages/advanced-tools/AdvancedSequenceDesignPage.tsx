import { useEffect, useMemo, useState } from "react";
import {
  fetchAdvancedToolsMeta,
  runAdvancedSequenceDesignStream,
  uploadAdvancedToolFile,
  loadAdvancedDefaultExample,
  type AdvancedSequenceDesignRequest,
  type AdvancedToolsMeta
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
    title: "Sequence Design (ProteinMPNN)",
    subtitle: "Configure full ProteinMPNN inference options for structure-conditioned sequence design.",
    coreSection: "ProteinMPNN Core Design",
    coreHint:
      "Choose model family by data type first: Soluble for discovery/design, Vanilla for membrane proteins, CA only when you only have C-alpha coordinates.",
    modelFamily: "Model Family",
    modelFamilySoluble: "Soluble (recommended for protein discovery and design)",
    modelFamilyVanilla: "Vanilla (recommended for membrane proteins)",
    modelFamilyCa: "CA (only for C-alpha coarse-grained coordinates)",
    designedChains: "Designed Chains (optional)",
    designedChainsPlaceholder: "A,B (empty means all chains)",
    fixedChains: "Fixed Chains",
    fixedChainsPlaceholder: "e.g. A",
    temperatures: "Temperatures",
    temperaturesPlaceholder: "0.1 or 0.1,0.2",
    numSequences: "Number of sequences",
    onlineLimitNote: (n: number) => `Online mode supports up to ${n} designed sequences per run.`,
    fixedResidues: "Fixed Residues (optional)",
    fixedResiduesPlaceholder: "A12,C13 or A:12,13;B:5-8",
    enableHomomer: "Enable homomer tying",
    modelRuntimeSection: "Model and Runtime",
    modelNameLabel: "Model Name",
    omitAas: "Omit AAs",
    modelNoiseHint:
      "`v_48_020` (0.20A) is the default for most structures (AI-generated backbones, AlphaFold, routine redesign). Use `v_48_002` (0.02A) only for very high-resolution native structures.",
    seed: "Seed",
    batchSize: "Batch Size",
    maxLength: "Max Length",
    advancedRulesSection: "Optional Advanced Rules",
    advancedRulesHint: "Use readable text rules. Backend converts them to JSON/JSONL automatically.",
    tiedPositions: "Tied Positions",
    tiedPositionsPlaceholder: "A12=B12;A13=B13",
    omitAaRules: "Omit AA Rules",
    omitAaRulesPlaceholder: "A12:WY;B5:AP",
    aaBias: "AA Bias",
    aaBiasPlaceholder: "A:-1.1,F:0.7",
    biasByResidue: "Bias By Residue",
    biasByResiduePlaceholder: "A12:F=1.0|W=-0.3;B5:A=0.2",
    pssmRules: "PSSM Rules",
    pssmRulesPlaceholder: "Optional JSON or compact rules",
    structureInputSection: "Structure Input",
    selectPdbFile: "Select PDB File",
    useExamplePdb: "Use Example PDB",
    selected: "Selected:",
    startBtn: "Start Sequence Design",
    resetDefaults: "Reset to Defaults",
    pdbRequired: "ProteinMPNN Sequence Design requires .pdb input.",
    pleaseUploadPdb: "Please upload a PDB file first.",
    tempsRequired: "Temperatures must include at least one numeric value.",
    onlineLimitExceeded: (n: number) => `Online mode supports up to ${n} designed sequences per run.`,
    preparingTask: "Preparing ProteinMPNN task...",
    mpnnDone: "ProteinMPNN design completed",
    resultTitle: "ProteinMPNN Sequence Design Result"
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "序列设计（ProteinMPNN）",
    subtitle: "配置完整的 ProteinMPNN 推理参数，进行基于结构的序列设计。",
    coreSection: "ProteinMPNN 核心设计",
    coreHint:
      "请先根据数据类型选择模型族：Soluble 适用于蛋白挖掘与设计，Vanilla 适用于膜蛋白，CA 仅在仅有 Cα 坐标时使用。",
    modelFamily: "模型族",
    modelFamilySoluble: "Soluble（推荐用于蛋白挖掘与设计）",
    modelFamilyVanilla: "Vanilla（推荐用于膜蛋白）",
    modelFamilyCa: "CA（仅适用于 Cα 粗粒度坐标）",
    designedChains: "设计链（可选）",
    designedChainsPlaceholder: "A,B（留空表示所有链）",
    fixedChains: "固定链",
    fixedChainsPlaceholder: "例如 A",
    temperatures: "温度",
    temperaturesPlaceholder: "0.1 或 0.1,0.2",
    numSequences: "生成序列数",
    onlineLimitNote: (n: number) => `在线模式每次运行最多支持 ${n} 条设计序列。`,
    fixedResidues: "固定残基（可选）",
    fixedResiduesPlaceholder: "A12,C13 或 A:12,13;B:5-8",
    enableHomomer: "启用同源寡聚体绑定",
    modelRuntimeSection: "模型与运行参数",
    modelNameLabel: "模型名称",
    omitAas: "排除的氨基酸",
    modelNoiseHint:
      "`v_48_020`（0.20Å）适用于多数结构（AI 生成骨架、AlphaFold、常规重设计）的默认选择。`v_48_002`（0.02Å）仅推荐用于高分辨率天然结构。",
    seed: "随机种子",
    batchSize: "批大小",
    maxLength: "最大长度",
    advancedRulesSection: "高级规则（可选）",
    advancedRulesHint: "使用可读的文本规则，后端会自动转换为 JSON/JSONL。",
    tiedPositions: "绑定位点",
    tiedPositionsPlaceholder: "A12=B12;A13=B13",
    omitAaRules: "排除氨基酸规则",
    omitAaRulesPlaceholder: "A12:WY;B5:AP",
    aaBias: "氨基酸偏置",
    aaBiasPlaceholder: "A:-1.1,F:0.7",
    biasByResidue: "按残基偏置",
    biasByResiduePlaceholder: "A12:F=1.0|W=-0.3;B5:A=0.2",
    pssmRules: "PSSM 规则",
    pssmRulesPlaceholder: "可选的 JSON 或紧凑规则",
    structureInputSection: "结构输入",
    selectPdbFile: "选择 PDB 文件",
    useExamplePdb: "使用示例 PDB",
    selected: "已选择：",
    startBtn: "开始序列设计",
    resetDefaults: "恢复默认设置",
    pdbRequired: "ProteinMPNN 序列设计需要 .pdb 输入。",
    pleaseUploadPdb: "请先上传 PDB 文件。",
    tempsRequired: "温度必须至少包含一个数值。",
    onlineLimitExceeded: (n: number) => `在线模式每次运行最多支持 ${n} 条设计序列。`,
    preparingTask: "正在准备 ProteinMPNN 任务…",
    mpnnDone: "ProteinMPNN 序列设计完成",
    resultTitle: "ProteinMPNN 序列设计结果"
  }
};

type ModelFamily = "soluble" | "vanilla" | "ca";
const DEFAULT_MPNN_OPTIONS: Required<AdvancedToolsMeta>["proteinmpnn_model_options"] = {
  vanilla: ["v_48_020", "v_48_002"],
  soluble: ["v_48_020", "v_48_002"],
  ca: ["v_48_020", "v_48_002"]
};
const MODEL_NOISE_DEFAULTS: Record<string, string> = {
  v_48_002: "0.02",
  v_48_010: "0.10",
  v_48_020: "0.20",
  v_48_030: "0.30"
};

function parseChainList(input: string): string[] {
  return input
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter((item) => /^[A-Z0-9]$/.test(item));
}

function parseNumberList(input: string): number[] {
  return input
    .split(/[,\s]+/)
    .map((item) => Number(item.trim()))
    .filter((value) => Number.isFinite(value));
}

type AdvancedSequenceDesignPageProps = {
  workspaceEnabled: boolean;
};

export function AdvancedSequenceDesignPage({ workspaceEnabled }: AdvancedSequenceDesignPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
  const [toolsMeta, setToolsMeta] = useState<AdvancedToolsMeta | null>(null);
  const [modelOptionsByFamily, setModelOptionsByFamily] = useState(DEFAULT_MPNN_OPTIONS);
  const [uploadedPath, setUploadedPath] = useState("");
  const [designedChainsText, setDesignedChainsText] = useState("");
  const [fixedChainsText, setFixedChainsText] = useState("");
  const [fixedResiduesText, setFixedResiduesText] = useState("");
  const [homomer, setHomomer] = useState(false);
  const [numSequences, setNumSequences] = useState(8);
  const [temperaturesText, setTemperaturesText] = useState("0.1");
  const [modelFamily, setModelFamily] = useState<ModelFamily>("soluble");
  const [modelName, setModelName] = useState("v_48_020");
  const [omitAas, setOmitAas] = useState("X");
  const [backboneNoise, setBackboneNoise] = useState("0.20");
  const [seed, setSeed] = useState("0");
  const [batchSize, setBatchSize] = useState("1");
  const [maxLength, setMaxLength] = useState("200000");
  const [tiedPositionsText, setTiedPositionsText] = useState("");
  const [omitAaRulesText, setOmitAaRulesText] = useState("");
  const [aaBiasText, setAaBiasText] = useState("");
  const [biasByResidueText, setBiasByResidueText] = useState("");
  const [pssmRulesText, setPssmRulesText] = useState("");
  const [pssmMulti, setPssmMulti] = useState("0.0");
  const [pssmThreshold, setPssmThreshold] = useState("0.0");
  const [pssmLogOddsFlag, setPssmLogOddsFlag] = useState("0");
  const [pssmBiasFlag, setPssmBiasFlag] = useState("0");

  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const onlineSequenceDesignLimit =
    toolsMeta?.online_limit_enabled ? Math.max(1, toolsMeta.online_sequence_design_limit ?? 50) : 512;

  useEffect(() => {
    setError("");
  }, [uploadedPath]);

  useEffect(() => {
    void (async () => {
      try {
        const meta = await fetchAdvancedToolsMeta();
        setToolsMeta(meta);
        if (meta.proteinmpnn_model_options) {
          setModelOptionsByFamily(meta.proteinmpnn_model_options);
        }
      } catch {
        // keep fallback options
      }
    })();
  }, []);

  const parsedTemperatures = useMemo(() => parseNumberList(temperaturesText), [temperaturesText]);
  const modelOptions = useMemo(
    () => modelOptionsByFamily[modelFamily] ?? modelOptionsByFamily.vanilla,
    [modelFamily, modelOptionsByFamily]
  );

  useEffect(() => {
    setBackboneNoise(MODEL_NOISE_DEFAULTS[modelName] || "0.20");
  }, [modelName]);

  useEffect(() => {
    if (!modelOptions.includes(modelName)) {
      setModelName(modelOptions[0] || "v_48_020");
    }
  }, [modelFamily, modelOptions, modelName]);

  useEffect(() => {
    if (numSequences > onlineSequenceDesignLimit) {
      setNumSequences(onlineSequenceDesignLimit);
    }
  }, [numSequences, onlineSequenceDesignLimit]);

  function resetToDefaults() {
    setDesignedChainsText("");
    setFixedChainsText("");
    setFixedResiduesText("");
    setHomomer(false);
    setNumSequences(8);
    setTemperaturesText("0.1");
    setModelFamily("soluble");
    setModelName("v_48_020");
    setOmitAas("X");
    setBackboneNoise("0.20");
    setSeed("0");
    setBatchSize("1");
    setMaxLength("200000");
    setTiedPositionsText("");
    setOmitAaRulesText("");
    setAaBiasText("");
    setBiasByResidueText("");
    setPssmRulesText("");
    setPssmMulti("0.0");
    setPssmThreshold("0.0");
    setPssmLogOddsFlag("0");
    setPssmBiasFlag("0");
  }

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const data = await uploadAdvancedToolFile(file);
      if (data.suffix !== ".pdb") {
        throw new Error(t.pdbRequired);
      }
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.uploadFailed);
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadAdvancedDefaultExample("pdb");
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.loadExampleFailed);
    }
  }

  async function onRun() {
    setError("");
    setRunning(true);
    setProgress(0);
    setProgressMessage(t.preparingTask);
    try {
      if (!uploadedPath) throw new Error(t.pleaseUploadPdb);
      if (parsedTemperatures.length === 0) throw new Error(t.tempsRequired);
      if (numSequences > onlineSequenceDesignLimit) {
        throw new Error(t.onlineLimitExceeded(onlineSequenceDesignLimit));
      }

      const body: AdvancedSequenceDesignRequest = {
        structure_file: uploadedPath,
        model_family: modelFamily,
        designed_chains: parseChainList(designedChainsText),
        fixed_chains: parseChainList(fixedChainsText),
        fixed_residues_text: fixedResiduesText.trim(),
        homomer,
        num_sequences: Number(numSequences),
        temperatures: parsedTemperatures,
        omit_aas: omitAas || "X",
        model_name: modelName || "v_48_020",
        backbone_noise: Number(backboneNoise || "0"),
        ca_only: modelFamily === "ca",
        use_soluble_model: modelFamily === "soluble",
        seed: Number(seed || "0"),
        batch_size: Number(batchSize || "1"),
        max_length: Number(maxLength || "200000"),
        tied_positions_text: tiedPositionsText.trim() || undefined,
        omit_aa_rules_text: omitAaRulesText.trim() || undefined,
        aa_bias_text: aaBiasText.trim() || undefined,
        bias_by_residue_text: biasByResidueText.trim() || undefined,
        pssm_rules_text: pssmRulesText.trim() || undefined,
        pssm_multi: Number(pssmMulti || "0"),
        pssm_threshold: Number(pssmThreshold || "0"),
        pssm_log_odds_flag: Number(pssmLogOddsFlag || "0"),
        pssm_bias_flag: Number(pssmBiasFlag || "0")
      };

      const payload = await runAdvancedSequenceDesignStream(body, (evt) => {
        setProgress(evt.progress);
        setProgressMessage(evt.message);
      });
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage(t.mpnnDone);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.runFailed);
    } finally {
      setRunning(false);
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
            <h3>{t.coreSection}</h3>
            <p className="advanced-ai-note">{t.coreHint}</p>
            <label className="left-controls">
              {t.modelFamily}
              <select value={modelFamily} onChange={(e) => setModelFamily(e.target.value as ModelFamily)}>
                <option value="soluble">{t.modelFamilySoluble}</option>
                <option value="vanilla">{t.modelFamilyVanilla}</option>
                <option value="ca">{t.modelFamilyCa}</option>
              </select>
            </label>
            <label className="left-controls">
              {t.designedChains}
              <input
                value={designedChainsText}
                onChange={(e) => setDesignedChainsText(e.target.value)}
                placeholder={t.designedChainsPlaceholder}
              />
            </label>
            <label className="left-controls">
              {t.fixedChains}
              <input value={fixedChainsText} onChange={(e) => setFixedChainsText(e.target.value)} placeholder={t.fixedChainsPlaceholder} />
            </label>
            <label className="left-controls">
              {t.temperatures}
              <input
                value={temperaturesText}
                onChange={(e) => setTemperaturesText(e.target.value)}
                placeholder={t.temperaturesPlaceholder}
              />
            </label>
            <label className="left-controls">
              {t.numSequences}
              <input
                type="number"
                min={1}
                max={onlineSequenceDesignLimit}
                value={numSequences}
                onChange={(e) =>
                  setNumSequences(
                    Math.max(1, Math.min(onlineSequenceDesignLimit, Number(e.target.value) || 1))
                  )
                }
              />
            </label>
            {toolsMeta?.online_limit_enabled && (
              <p className="advanced-ai-note">
                {t.onlineLimitNote(onlineSequenceDesignLimit)}
              </p>
            )}
            <label className="left-controls">
              {t.fixedResidues}
              <textarea
                rows={2}
                className="advanced-two-line-text"
                value={fixedResiduesText}
                onChange={(e) => setFixedResiduesText(e.target.value)}
                placeholder={t.fixedResiduesPlaceholder}
              />
            </label>
            <label className="quick-ai-toggle advanced-homomer-row">
              <input type="checkbox" checked={homomer} onChange={(e) => setHomomer(e.target.checked)} />
              <span className="quick-ai-toggle-box" />
              <span className="quick-ai-toggle-text">{t.enableHomomer}</span>
            </label>
          </section>

          <section className="custom-section-card">
            <h3>{t.modelRuntimeSection}</h3>
            <label className="left-controls">
              {t.modelNameLabel}
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {modelOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label className="left-controls">
              {t.omitAas}
              <input value={omitAas} onChange={(e) => setOmitAas(e.target.value)} placeholder="X" />
            </label>
            <p className="advanced-ai-note">
              {t.modelNoiseHint}
            </p>
            <label className="left-controls">
              {t.seed}
              <input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="0" />
            </label>
            <label className="left-controls">
              {t.batchSize}
              <input value={batchSize} onChange={(e) => setBatchSize(e.target.value)} placeholder="1" />
            </label>
            <label className="left-controls">
              {t.maxLength}
              <input value={maxLength} onChange={(e) => setMaxLength(e.target.value)} placeholder="200000" />
            </label>
          </section>

          <section className="custom-section-card">
            <h3>{t.advancedRulesSection}</h3>
            <p className="advanced-ai-note">{t.advancedRulesHint}</p>
            <label className="left-controls">
              {t.tiedPositions}
              <input
                value={tiedPositionsText}
                onChange={(e) => setTiedPositionsText(e.target.value)}
                placeholder={t.tiedPositionsPlaceholder}
              />
            </label>
            <label className="left-controls">
              {t.omitAaRules}
              <input
                value={omitAaRulesText}
                onChange={(e) => setOmitAaRulesText(e.target.value)}
                placeholder={t.omitAaRulesPlaceholder}
              />
            </label>
            <label className="left-controls">
              {t.aaBias}
              <input
                value={aaBiasText}
                onChange={(e) => setAaBiasText(e.target.value)}
                placeholder={t.aaBiasPlaceholder}
              />
            </label>
            <label className="left-controls">
              {t.biasByResidue}
              <input
                value={biasByResidueText}
                onChange={(e) => setBiasByResidueText(e.target.value)}
                placeholder={t.biasByResiduePlaceholder}
              />
            </label>
            <label className="left-controls">
              {t.pssmRules}
              <input
                value={pssmRulesText}
                onChange={(e) => setPssmRulesText(e.target.value)}
                placeholder={t.pssmRulesPlaceholder}
              />
            </label>
            <label className="left-controls">
              pssm_multi
              <input value={pssmMulti} onChange={(e) => setPssmMulti(e.target.value)} />
            </label>
            <label className="left-controls">
              pssm_threshold
              <input value={pssmThreshold} onChange={(e) => setPssmThreshold(e.target.value)} />
            </label>
            <label className="left-controls">
              pssm_log_odds_flag
              <input value={pssmLogOddsFlag} onChange={(e) => setPssmLogOddsFlag(e.target.value)} />
            </label>
            <label className="left-controls">
              pssm_bias_flag
              <input value={pssmBiasFlag} onChange={(e) => setPssmBiasFlag(e.target.value)} />
            </label>
          </section>

          <section className="custom-section-card">
            <h3>{t.structureInputSection}</h3>
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  {t.selectPdbFile}
                  <input type="file" accept=".pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
                  acceptedCategories={["structure"]}
                  buttonLabel={t.fromWorkspace}
                  onPick={(picked) => {
                    const selected = picked[0];
                    if (!selected) return;
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

          <button type="button" className="custom-btn-primary" onClick={() => void onRun()} disabled={running}>
            {running ? t.runningBtn : t.startBtn}
          </button>
          <button type="button" className="custom-btn-secondary" onClick={resetToDefaults} disabled={running}>
            {t.resetDefaults}
          </button>
        </>
      }
      right={
        <AdvancedResultPanel
          title={t.resultTitle}
          resultPayload={resultPayload}
          aiSummary={(resultPayload?.ai_summary as string) || ""}
          error={error}
          showSummaryTab
        />
      }
    />
  );
}
