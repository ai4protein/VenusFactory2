import { useEffect, useMemo, useState } from "react";
import {
  abortPredict,
  fetchCustomModelMeta,
  fetchModelConfig,
  fetchModelFolders,
  fetchModelsInFolder,
  previewPredict,
  startPredictStream,
  uploadCustomModelPredictBatchFile,
  uploadCustomModelPredictBatchText,
  type CustomModelMeta,
  type ModelOption
} from "../lib/customModelApi";
import { SegmentedSwitch } from "../components/SegmentedSwitch";
import { PageFooter } from "../components/PageFooter";
import { WorkspaceFilePicker } from "../components/WorkspaceFilePicker";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const STRUCTURE_MODELS = ["protssn", "prosst", "saprot"];
const SES_STRUCTURE_COLUMNS = ["foldseek_seq", "ss8_seq"];

const STRINGS = {
  en: {
    docTitle: "Predict — VenusFactory2",
    docDescription: "Run predictions with your custom-trained models.",
    headerTitle: "Custom Model Predict",
    headerSubtitle: "Run single or batch inference with your selected custom model.",
    readonlyBanner: "Online mode: custom model controls are view-only in this deployment.",
    onlineUnavailable: "Online mode: unavailable",
    noOptions: "No options available",
    failedLoadMeta: "Failed to load metadata.",
    modelFolder: "Model Folder",
    modelPath: "Model Path",
    selectModel: "Select model",
    predictParams: "Predict Parameters",
    modelSettings: "Model Settings",
    plm: "PLM",
    evalMethod: "Eval Method",
    pooling: "Pooling",
    problemType: "Problem Type",
    numLabels: "Num Labels",
    predictionMode: "Prediction Mode",
    predictionModeSwitch: "Prediction mode switch",
    single: "Single",
    batch: "Batch",
    batchSize: "Batch Size",
    inputSettings: "Input Settings",
    structureModelNote: "Selected structure model uses PDB Folder based input.",
    aaSequence: "AA Sequence",
    aaSeqPlaceholder: "Paste protein sequence...",
    batchSource: "Batch Source",
    batchSourceSwitch: "Batch input source switch",
    upload: "Upload",
    pasteFastaTab: "Paste FASTA",
    pathTab: "Path",
    fileLabel: "File",
    chooseFile: "Choose File",
    fromWorkspace: "From Workspace",
    useExample: "Use Example",
    clear: "Clear",
    pasteFasta: "Paste FASTA",
    inputFilePath: "Input File Path",
    inputFilePathPlaceholder: "e.g. data/test.csv or data/test.fasta",
    structureInputs: "Structure Inputs",
    structureSeq: "Structure Seq",
    foldseekSequence: "Foldseek Sequence",
    ss8Sequence: "SS8 Sequence",
    pdbDir: "PDB Dir",
    structureRequirePdb: "Selected structure model requires PDB Folder.",
    previewCommand: "Preview Command",
    startPredict: "Start Predict",
    abort: "Abort",
    outputPanel: "Output Panel",
    commandPreview: "Command Preview",
    clickPreview: "Click Preview Command to generate CLI command.",
    progress: "Progress",
    noLogs: "No logs yet.",
    noFileSelected: "No file selected",
    statusIdle: "Idle",
    statusStarting: "Starting...",
    statusRunning: "Running",
    statusCompleted: "Completed",
    statusFailed: "Failed",
    statusAborted: "Aborted",
    plmCfgFallback: "PLM missing in ckpt config, fallback to current/default value.",
    trainMethodCfgFallback: "training_method missing in ckpt config, fallback to default eval method.",
    poolingCfgFallback: "pooling_method missing in ckpt config, fallback to default pooling.",
    problemTypeCfgFallback: "problem_type missing in ckpt config, fallback to default problem type.",
    numLabelsCfgFallback: "num_labels missing in ckpt config, fallback to default label count.",
    valuesAutoFilled: "Values auto-filled from selected model.",
    valuesAutoFilledLocked: "Values auto-filled from selected model and locked for consistency.",
    modelConfigNotFound: "Model config not found. Parameters remain editable with current/default values.",
    errStructurePdb: "Structure PLM (ProSST/ProtSSN/SaProt) requires PDB Folder.",
    errBatchFastaEmpty: "Batch FASTA text cannot be empty.",
    errBatchInputRequired: "Batch prediction requires an input file.",
    errSesNeedsCols: "ses-adapter requires selecting foldseek_seq and/or ss8_seq.",
    errSingleSesFoldseek: "Single predict with ses-adapter requires Foldseek Sequence or PDB Folder.",
    errSingleSesSs8: "Single predict with ses-adapter requires SS8 Sequence or PDB Folder.",
    errBatchSesCols: "Batch ses-adapter requires selected structure columns in input file, or provide PDB Folder.",
    errPreviewFailed: "Preview failed.",
    errPredictFailed: "Predict failed.",
    errPredictStartFailed: "Predict start failed.",
    errFileUploadFailed: "File upload failed."
  },
  zh: {
    docTitle: "预测 — VenusFactory2",
    docDescription: "使用您自定义训练的模型进行预测。",
    headerTitle: "自定义模型预测",
    headerSubtitle: "使用您选择的自定义模型进行单条或批量推理。",
    readonlyBanner: "在线模式：当前部署下自定义模型控件仅供查看。",
    onlineUnavailable: "在线模式：不可用",
    noOptions: "暂无可选项",
    failedLoadMeta: "加载元数据失败。",
    modelFolder: "模型文件夹",
    modelPath: "模型路径",
    selectModel: "请选择模型",
    predictParams: "预测参数",
    modelSettings: "模型设置",
    plm: "PLM",
    evalMethod: "评估方法",
    pooling: "池化方式",
    problemType: "任务类型",
    numLabels: "类别数",
    predictionMode: "预测模式",
    predictionModeSwitch: "预测模式切换",
    single: "单条",
    batch: "批量",
    batchSize: "批次大小",
    inputSettings: "输入设置",
    structureModelNote: "所选结构模型基于 PDB 文件夹输入。",
    aaSequence: "氨基酸序列",
    aaSeqPlaceholder: "请粘贴蛋白序列…",
    batchSource: "批量来源",
    batchSourceSwitch: "批量输入来源切换",
    upload: "上传",
    pasteFastaTab: "粘贴 FASTA",
    pathTab: "路径",
    fileLabel: "文件",
    chooseFile: "选择文件",
    fromWorkspace: "从工作区",
    useExample: "使用示例",
    clear: "清除",
    pasteFasta: "粘贴 FASTA",
    inputFilePath: "输入文件路径",
    inputFilePathPlaceholder: "例如 data/test.csv 或 data/test.fasta",
    structureInputs: "结构输入",
    structureSeq: "结构序列",
    foldseekSequence: "Foldseek 序列",
    ss8Sequence: "SS8 序列",
    pdbDir: "PDB 目录",
    structureRequirePdb: "所选结构模型需要提供 PDB 文件夹。",
    previewCommand: "预览命令",
    startPredict: "开始预测",
    abort: "中止",
    outputPanel: "输出面板",
    commandPreview: "命令预览",
    clickPreview: "点击「预览命令」生成 CLI 命令。",
    progress: "进度",
    noLogs: "暂无日志。",
    noFileSelected: "未选择文件",
    statusIdle: "空闲",
    statusStarting: "启动中…",
    statusRunning: "运行中",
    statusCompleted: "已完成",
    statusFailed: "失败",
    statusAborted: "已中止",
    plmCfgFallback: "ckpt 配置缺少 PLM，回退至当前/默认值。",
    trainMethodCfgFallback: "ckpt 配置缺少 training_method，回退至默认评估方法。",
    poolingCfgFallback: "ckpt 配置缺少 pooling_method，回退至默认池化方式。",
    problemTypeCfgFallback: "ckpt 配置缺少 problem_type，回退至默认任务类型。",
    numLabelsCfgFallback: "ckpt 配置缺少 num_labels，回退至默认类别数。",
    valuesAutoFilled: "已根据所选模型自动填充。",
    valuesAutoFilledLocked: "已根据所选模型自动填充并锁定以保证一致性。",
    modelConfigNotFound: "未找到模型配置。参数仍可编辑，使用当前/默认值。",
    errStructurePdb: "结构 PLM（ProSST/ProtSSN/SaProt）需要提供 PDB 文件夹。",
    errBatchFastaEmpty: "批量 FASTA 文本不能为空。",
    errBatchInputRequired: "批量预测需要提供输入文件。",
    errSesNeedsCols: "ses-adapter 需要勾选 foldseek_seq 和/或 ss8_seq。",
    errSingleSesFoldseek: "使用 ses-adapter 的单条预测需要提供 Foldseek 序列或 PDB 文件夹。",
    errSingleSesSs8: "使用 ses-adapter 的单条预测需要提供 SS8 序列或 PDB 文件夹。",
    errBatchSesCols: "批量 ses-adapter 需要在输入文件中包含所选结构列，或提供 PDB 文件夹。",
    errPreviewFailed: "预览失败。",
    errPredictFailed: "预测失败。",
    errPredictStartFailed: "预测启动失败。",
    errFileUploadFailed: "文件上传失败。"
  }
};

function isNotFoundLikeError(message: string): boolean {
  const text = String(message || "").toLowerCase();
  return text.includes("404") || text.includes("not found") || text.includes('{"detail":"not found"}');
}

type CustomModelPredictPageProps = {
  readonly?: boolean;
  workspaceEnabled?: boolean;
};

export function CustomModelPredictPage({ readonly = false, workspaceEnabled = false }: CustomModelPredictPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: t.docTitle, description: t.docDescription });
  const [meta, setMeta] = useState<CustomModelMeta | null>(null);
  const [predictionMode, setPredictionMode] = useState<"single" | "batch">("single");
  const [batchInputSource, setBatchInputSource] = useState<"upload" | "paste" | "path">("upload");
  const [folderOptions, setFolderOptions] = useState<string[]>(["ckpt"]);
  const [selectedFolder, setSelectedFolder] = useState("ckpt");
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [modelPath, setModelPath] = useState("");
  const [plmModel, setPlmModel] = useState("");
  const [evalMethod, setEvalMethod] = useState("full");
  const [poolingMethod, setPoolingMethod] = useState("mean");
  const [problemType, setProblemType] = useState("single_label_classification");
  const [numLabels, setNumLabels] = useState(2);
  const [aaSeq, setAaSeq] = useState("");
  const [inputFile, setInputFile] = useState("");
  const [batchFastaText, setBatchFastaText] = useState("");
  const [batchColumns, setBatchColumns] = useState<string[]>([]);
  const [batchSize, setBatchSize] = useState(1);
  const [structureSeq, setStructureSeq] = useState<string[]>([]);
  const [foldseekSeq, setFoldseekSeq] = useState("");
  const [ss8Seq, setSs8Seq] = useState("");
  const [pdbDir, setPdbDir] = useState("");

  const [running, setRunning] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [commandPreview, setCommandPreview] = useState("");
  const [error, setError] = useState("");
  const [ckptLocked, setCkptLocked] = useState(false);
  const [ckptConfigNote, setCkptConfigNote] = useState("");
  const emptySelectLabel = readonly ? t.onlineUnavailable : t.noOptions;

  useEffect(() => {
    if (readonly) return;
    void (async () => {
      try {
        const data = await fetchCustomModelMeta();
        setMeta(data);
        setPlmModel(Object.keys(data.plm_models)[0] || "");
      } catch (err) {
        setError(err instanceof Error ? err.message : t.failedLoadMeta);
      }
    })();
  }, [readonly]);

  useEffect(() => {
    if (readonly) return;
    void (async () => {
      const folders = await fetchModelFolders("ckpt");
      setFolderOptions(folders.folders.length ? folders.folders : ["ckpt"]);
    })();
  }, [readonly]);

  useEffect(() => {
    if (readonly) return;
    if (!selectedFolder) return;
    void (async () => {
      const result = await fetchModelsInFolder(selectedFolder);
      setModelOptions(result.models);
      setModelPath(result.models[0]?.path || "");
    })();
  }, [selectedFolder, readonly]);

  useEffect(() => {
    if (!modelPath) {
      setCkptLocked(false);
      setCkptConfigNote("");
    }
  }, [modelPath]);

  useEffect(() => {
    if (readonly) return;
    if (!modelPath || !meta) return;
    void (async () => {
      try {
        const cfg = (await fetchModelConfig(modelPath)).config;
        const notes: string[] = [];
        const display = Object.entries(meta.plm_models).find(([, v]) => v === cfg.plm_model)?.[0];
        if (display) {
          setPlmModel(display);
        } else {
          notes.push(t.plmCfgFallback);
        }

        if (typeof cfg.training_method === "string") {
          setEvalMethod(cfg.training_method);
        } else {
          notes.push(t.trainMethodCfgFallback);
        }

        if (typeof cfg.pooling_method === "string") {
          setPoolingMethod(cfg.pooling_method);
        } else {
          notes.push(t.poolingCfgFallback);
        }

        if (typeof cfg.problem_type === "string") {
          setProblemType(cfg.problem_type);
        } else {
          notes.push(t.problemTypeCfgFallback);
        }

        if (typeof cfg.num_labels === "number") {
          setNumLabels(cfg.num_labels);
        } else {
          notes.push(t.numLabelsCfgFallback);
        }

        const structureFromCfg = cfg.structure_seq;
        if (Array.isArray(structureFromCfg)) {
          setStructureSeq(structureFromCfg.filter((x): x is string => typeof x === "string"));
        } else if (typeof structureFromCfg === "string") {
          setStructureSeq(
            structureFromCfg
              .split(",")
              .map((x) => x.trim())
              .filter(Boolean)
          );
        }

        setCkptLocked(true);
        setCkptConfigNote(
          notes.length
            ? `${t.valuesAutoFilled} ${notes.join(" ")}`
            : t.valuesAutoFilledLocked
        );
      } catch {
        setCkptLocked(false);
        setCkptConfigNote(t.modelConfigNotFound);
      }
    })();
  }, [modelPath, meta, readonly]);

  const modelHint = useMemo(() => `${plmModel} ${modelPath}`.toLowerCase(), [plmModel, modelPath]);
  const plmModelKeys = useMemo(() => Object.keys(meta?.plm_models || {}), [meta?.plm_models]);
  const isStructurePlm = useMemo(() => STRUCTURE_MODELS.some((key) => modelHint.includes(key)), [modelHint]);
  const isSesAdapter = useMemo(() => {
    const method = String(evalMethod || "").toLowerCase();
    return method === "ses-adapter" || method === "ses_adapter";
  }, [evalMethod]);
  const structureSeqRequired = useMemo(
    () => structureSeq.filter((item) => SES_STRUCTURE_COLUMNS.includes(item)),
    [structureSeq]
  );
  const knownBatchColumns = useMemo(
    () => new Set<string>(batchColumns.map((x) => String(x || "").trim()).filter(Boolean)),
    [batchColumns]
  );
  const showStructureInputs = isStructurePlm || isSesAdapter;
  const showPdbDir = isStructurePlm || isSesAdapter;
  const showStructureSeq = isSesAdapter;
  const showFoldseekInput = isSesAdapter && structureSeq.includes("foldseek_seq");
  const showSs8Input = isSesAdapter && structureSeq.includes("ss8_seq");
  const predictRuleError = useMemo(() => {
    if (isStructurePlm && !pdbDir.trim()) {
      return t.errStructurePdb;
    }
    if (predictionMode === "batch") {
      if (batchInputSource === "paste" && !batchFastaText.trim()) {
        return t.errBatchFastaEmpty;
      }
      if (batchInputSource !== "paste" && !inputFile.trim()) {
        return t.errBatchInputRequired;
      }
    }
    if (isSesAdapter) {
      if (!structureSeqRequired.length) {
        return t.errSesNeedsCols;
      }
      if (predictionMode === "single" && !pdbDir.trim()) {
        if (structureSeqRequired.includes("foldseek_seq") && !foldseekSeq.trim()) {
          return t.errSingleSesFoldseek;
        }
        if (structureSeqRequired.includes("ss8_seq") && !ss8Seq.trim()) {
          return t.errSingleSesSs8;
        }
      }
      if (predictionMode === "batch" && !pdbDir.trim() && structureSeqRequired.some((col) => !knownBatchColumns.has(col))) {
        return t.errBatchSesCols;
      }
    }
    return "";
  }, [
    isStructurePlm,
    pdbDir,
    predictionMode,
    batchInputSource,
    batchFastaText,
    inputFile,
    isSesAdapter,
    structureSeqRequired,
    foldseekSeq,
    ss8Seq,
    knownBatchColumns,
    t
  ]);

  const predictArgs = useMemo(
    () => ({
      prediction_mode: predictionMode,
      plm_model: plmModel,
      model_path: modelPath,
      eval_method: evalMethod,
      pooling_method: poolingMethod,
      problem_type: problemType,
      num_labels: numLabels,
      aa_seq: aaSeq,
      input_file: inputFile,
      batch_size: batchSize,
      structure_seq: structureSeq,
      foldseek_seq: foldseekSeq,
      ss8_seq: ss8Seq,
      pdb_dir: pdbDir
    }),
    [
      predictionMode,
      plmModel,
      modelPath,
      evalMethod,
      poolingMethod,
      problemType,
      numLabels,
      aaSeq,
      inputFile,
      batchSize,
      structureSeq,
      foldseekSeq,
      ss8Seq,
      pdbDir
    ]
  );

  async function onPreviewCommand() {
    if (readonly) return;
    if (predictRuleError) {
      setError(predictRuleError);
      return;
    }
    setError("");
    try {
      const resolvedInputFile = await resolveBatchInputFile();
      const result = await previewPredict({ ...predictArgs, input_file: resolvedInputFile });
      setCommandPreview(result.command);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errPreviewFailed);
    }
  }

  async function onStart() {
    if (readonly) return;
    if (predictRuleError) {
      setError(predictRuleError);
      return;
    }
    setError("");
    setRunning(true);
    setLogs([]);
    setProgress(0);
    setStatusText(t.statusStarting);
    try {
      const resolvedInputFile = await resolveBatchInputFile();
      await startPredictStream({ ...predictArgs, input_file: resolvedInputFile }, (evt) => {
        if (evt.type === "start") {
          setCommandPreview(evt.data.command || "");
          setStatusText(t.statusRunning);
          return;
        }
        if (evt.type === "progress") {
          const nextProgress = Number.isFinite(evt.data.progress) ? evt.data.progress : 0;
          const nextMessage = evt.data.message || t.statusRunning;
          setProgress((prev) => Math.max(prev, nextProgress));
          setStatusText((prev) => {
            if (nextMessage.startsWith("Epoch ")) return nextMessage;
            if (prev.startsWith("Epoch ") && nextProgress < 0.999) return prev;
            return nextMessage;
          });
        }
        if (evt.type === "log" && evt.data.line) setLogs((prev) => [...prev, evt.data.line]);
        if (evt.type === "error") setError(evt.data.message || t.errPredictFailed);
        if (evt.type === "done") {
          const finalProgress = evt.data.final_progress;
          if (typeof finalProgress === "number") {
            setProgress((prev) => Math.max(prev, finalProgress));
          }
          setStatusText(evt.data.message || (evt.data.success ? t.statusCompleted : t.statusFailed));
          setProgress((prev) => (evt.data.success ? 1 : prev));
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errPredictStartFailed);
      setStatusText(t.statusFailed);
    } finally {
      setRunning(false);
    }
  }

  async function onAbort() {
    if (readonly) return;
    await abortPredict();
    setStatusText(t.statusAborted);
    setRunning(false);
  }

  function toggleStructureSeqOption(option: string) {
    if (readonly || ckptLocked) return;
    setStructureSeq((prev) => (prev.includes(option) ? prev.filter((x) => x !== option) : [...prev, option]));
  }

  async function onUploadBatchFile(file: File | null) {
    if (readonly) return;
    if (!file) return;
    setError("");
    try {
      const result = await uploadCustomModelPredictBatchFile(file);
      setInputFile(result.file_path);
      if (Array.isArray(result.columns)) {
        setBatchColumns(result.columns.filter(Boolean));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errFileUploadFailed);
    }
  }

  async function onUseBatchExample() {
    if (readonly) return;
    const content = ">seq1\nMKTAYIAKQRQISFVKSHFSRQ\n>seq2\nGAVLILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSR\n";
    const file = new File([content], "predict_example.fasta", { type: "text/plain" });
    await onUploadBatchFile(file);
  }

  async function resolveBatchInputFile() {
    if (readonly) return inputFile.trim();
    if (predictionMode !== "batch") return inputFile;
    if (batchInputSource !== "paste") return inputFile.trim();
    const text = batchFastaText.trim();
    if (!text) return "";
    const uploaded = await uploadCustomModelPredictBatchText(text);
    setInputFile(uploaded.file_path);
    setBatchColumns(Array.isArray(uploaded.columns) ? uploaded.columns.filter(Boolean) : []);
    return uploaded.file_path;
  }

  function clearBatchInputFile() {
    if (readonly) return;
    setInputFile("");
    setBatchColumns([]);
  }

  function displayUploadedName(pathValue: string) {
    const normalized = String(pathValue || "").trim();
    if (!normalized) return t.noFileSelected;
    return normalized.split("/").pop() || normalized;
  }
  const visibleError = readonly && isNotFoundLikeError(error) ? "" : error;

  return (
    <div className={`custom-model-page ${readonly ? "readonly-mode" : ""}`}>
      <header className="chat-header">
        <div>
          <h2>{t.headerTitle}</h2>
          <p>{t.headerSubtitle}</p>
        </div>
        <div className={`run-status-bar ${running ? "running" : "stopped"}`}>
          <span className="run-status-dot" />
          <span className="run-status-text">{statusText || t.statusIdle}</span>
        </div>
      </header>
      {readonly && (
        <div className="readonly-banner" role="status" aria-live="polite">
          {t.readonlyBanner}
        </div>
      )}

      <section className="custom-model-grid">
        <aside className="chat-panel left custom-model-controls">
          <fieldset className="readonly-fieldset" disabled={readonly}>
          <section className="custom-section-card custom-section-wide">
            <label className="left-controls custom-field-short">{t.modelFolder}
              <select value={selectedFolder} onChange={(e) => setSelectedFolder(e.target.value)}>
                {folderOptions.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-medium">{t.modelPath}
              <select value={modelPath} onChange={(e) => setModelPath(e.target.value)}>
                <option value="">{t.selectModel}</option>
                {modelOptions.map((m) => <option key={m.path} value={m.path}>{m.label}</option>)}
              </select>
            </label>
          </section>
          </fieldset>
        </aside>

        <section className="custom-bottom-split">
          <aside className="chat-panel left custom-bottom-left custom-model-controls">
            <fieldset className="readonly-fieldset" disabled={readonly}>
            <section className="custom-section-card custom-section-wide">
              <h3 className="custom-panel-title">{t.predictParams}</h3>
              <div className="custom-advanced-grid custom-advanced-panel custom-predict-params">
                {ckptConfigNote && <div className="custom-readonly-note custom-field-span-full">{ckptConfigNote}</div>}
                <div className="custom-param-group-title custom-field-span-full">{t.modelSettings}</div>
            <label className="left-controls custom-field-short">{t.plm}
              <select value={plmModel} onChange={(e) => setPlmModel(e.target.value)} disabled={ckptLocked}>
                {plmModelKeys.length === 0 && <option value="">{emptySelectLabel}</option>}
                {plmModelKeys.map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-short">{t.evalMethod}
              <select value={evalMethod} onChange={(e) => setEvalMethod(e.target.value)} disabled={ckptLocked}>
                {(meta?.training_methods || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                {(meta?.training_methods || []).map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-short">{t.pooling}
              <select value={poolingMethod} onChange={(e) => setPoolingMethod(e.target.value)} disabled={ckptLocked}>
                {(meta?.pooling_methods || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                {(meta?.pooling_methods || []).map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-short">{t.problemType}
              <select value={problemType} onChange={(e) => setProblemType(e.target.value)} disabled={ckptLocked}>
                {(meta?.problem_types || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                {(meta?.problem_types || []).map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-short">{t.numLabels}<input type="number" value={numLabels} onChange={(e) => setNumLabels(Number(e.target.value) || 1)} disabled={ckptLocked} /></label>
                <div className="custom-field-span-full predict-mode-row">
                  <div className="custom-mode-inline" role="group" aria-label={t.predictionMode}>
                    <span className="custom-mode-inline-label">{t.predictionMode}</span>
                    <SegmentedSwitch
                      value={predictionMode}
                      onChange={setPredictionMode}
                      ariaLabel={t.predictionModeSwitch}
                      className="custom-segment-switch-compact"
                      options={[
                        { value: "single", label: t.single },
                        { value: "batch", label: t.batch }
                      ]}
                    />
                  </div>
                  {predictionMode === "batch" && (
                    <label className="left-controls predict-batch-size-field">
                      {t.batchSize}
                      <input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value) || 1)} />
                    </label>
                  )}
                </div>
                <div className="custom-param-group-title custom-field-span-full">{t.inputSettings}</div>
                {predictionMode === "single" ? (
                  isStructurePlm ? (
                    <p className="custom-readonly-note custom-field-span-full">{t.structureModelNote}</p>
                  ) : (
                    <label className="left-controls custom-field-span-full">{t.aaSequence}
                      <textarea rows={6} value={aaSeq} onChange={(e) => setAaSeq(e.target.value)} placeholder={t.aaSeqPlaceholder} />
                    </label>
                  )
                ) : (
                  <>
                    <div className="custom-batch-source-row custom-field-span-full predict-batch-source-row">
                      <span className="custom-mode-inline-label">{t.batchSource}</span>
                      <SegmentedSwitch
                        value={batchInputSource}
                        onChange={(value) => setBatchInputSource(value as "upload" | "paste" | "path")}
                        ariaLabel={t.batchSourceSwitch}
                        className="custom-segment-switch-compact"
                        options={[
                          { value: "upload", label: t.upload },
                          { value: "paste", label: t.pasteFastaTab },
                          { value: "path", label: t.pathTab }
                        ]}
                      />
                    </div>
                    {batchInputSource === "upload" ? (
                      <div className="custom-upload-dropzone-wrap custom-field-span-2 predict-batch-input-row">
                        <div className="custom-upload-dropzone-grid">
                          <div className="custom-upload-item upload-source-stack">
                            <span className="custom-upload-item-label">{t.fileLabel}</span>
                            <label className="custom-upload-trigger">
                              <input
                                type="file"
                                accept=".csv,.tsv,.xlsx,.xls,.fasta,.fa,.txt"
                                onChange={(e) => void onUploadBatchFile(e.target.files?.[0] || null)}
                              />
                              {t.chooseFile}
                            </label>
                            <WorkspaceFilePicker
                              workspaceEnabled={workspaceEnabled}
                              disabled={readonly || running}
                              acceptedCategories={["table_or_text", "sequence"]}
                              buttonLabel={t.fromWorkspace}
                              onPick={(picked) => {
                                const selected = picked[0];
                                if (!selected) return;
                                setInputFile(selected.storage_path);
                                setBatchColumns([]);
                              }}
                            />
                            <button
                              type="button"
                              className="custom-btn-secondary"
                              onClick={() => void onUseBatchExample()}
                              disabled={readonly || running}
                            >
                              {t.useExample}
                            </button>
                            <span className="custom-upload-file-chip">{displayUploadedName(inputFile)}</span>
                            <button type="button" className="custom-upload-clear-btn" onClick={clearBatchInputFile}>
                              {t.clear}
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : batchInputSource === "paste" ? (
                      <label className="left-controls custom-field-span-2 predict-batch-input-row">{t.pasteFasta}
                        <textarea
                          rows={5}
                          value={batchFastaText}
                          onChange={(e) => setBatchFastaText(e.target.value)}
                          placeholder=">seq1&#10;MKT...&#10;>seq2&#10;GAV..."
                        />
                      </label>
                    ) : (
                      <label className="left-controls custom-field-span-2 predict-batch-input-row">{t.inputFilePath}
                        <input value={inputFile} onChange={(e) => setInputFile(e.target.value)} placeholder={t.inputFilePathPlaceholder} />
                      </label>
                    )}
                  </>
                )}

                {showStructureInputs && (
                  <>
                    <div className="custom-param-group-title custom-field-span-full">{t.structureInputs}</div>
                    {showStructureSeq && (
                      <div className="left-controls custom-field-span-2">
                        <span>{t.structureSeq}</span>
                        <div className="custom-multi-grid">
                          {(meta?.structure_seq_options || []).map((item) => {
                            const checked = structureSeq.includes(item);
                            return (
                              <button
                                key={item}
                                type="button"
                                className={`custom-multi-item ${checked ? "active" : ""}`}
                                aria-pressed={checked}
                                onClick={() => toggleStructureSeqOption(item)}
                                disabled={ckptLocked}
                              >
                                <span className="custom-multi-item-label">{item}</span>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    {showFoldseekInput && (
                      <label className="left-controls custom-field-span-2">{t.foldseekSequence}
                        <textarea rows={3} value={foldseekSeq} onChange={(e) => setFoldseekSeq(e.target.value)} />
                      </label>
                    )}
                    {showSs8Input && (
                      <label className="left-controls custom-field-span-2">{t.ss8Sequence}
                        <textarea rows={3} value={ss8Seq} onChange={(e) => setSs8Seq(e.target.value)} />
                      </label>
                    )}
                    {showPdbDir && <label className="left-controls custom-field-short">{t.pdbDir}<input value={pdbDir} onChange={(e) => setPdbDir(e.target.value)} /></label>}
                  </>
                )}
                {isStructurePlm && (
                  <div className="custom-readonly-note custom-field-span-full">
                    {t.structureRequirePdb}
                  </div>
                )}
                {predictRuleError && <div className="custom-readonly-note custom-field-span-full">{predictRuleError}</div>}
              </div>

              <div className="custom-actions-inline-status">
                <div className="custom-row custom-actions">
                  <button type="button" className="custom-btn-secondary" disabled={readonly} onClick={() => void onPreviewCommand()}>{t.previewCommand}</button>
                  <button type="button" className="custom-btn-primary" disabled={readonly || running} onClick={() => void onStart()}>{t.startPredict}</button>
                  <button type="button" className="custom-btn-danger" disabled={readonly || !running} onClick={() => void onAbort()}>{t.abort}</button>
                </div>
              </div>
            </section>
            </fieldset>
          </aside>

          <div className="chat-panel right custom-model-output custom-bottom-right">
            <h3 className="custom-output-panel-title custom-panel-title">{t.outputPanel}</h3>
            {visibleError && <div className="error-box">{visibleError}</div>}
            <div className="custom-inline-status-panel custom-inline-status-panel-compact">
              <details className="custom-preview-collapse custom-command-preview-collapse">
                <summary>{t.commandPreview}</summary>
                <pre className="custom-command custom-command-plain">
                  <code>{commandPreview || t.clickPreview}</code>
                </pre>
              </details>
              <div className="custom-inline-status-title">{t.progress}</div>
              <div className="custom-progress-wrap">
                <div className="custom-progress-meta">
                  <span>{statusText || t.statusIdle}</span>
                  <span>{Math.round(progress * 100)}%</span>
                </div>
                <div className="custom-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
                  <div className="custom-progress-fill" style={{ width: `${Math.max(0, Math.min(100, progress * 100))}%` }} />
                </div>
              </div>
            </div>
            <div className="custom-log-box">
              {logs.length ? logs.map((line, idx) => <div key={`${idx}-${line.slice(0, 20)}`}>{line}</div>) : t.noLogs}
            </div>
          </div>
        </section>
      </section>
      <PageFooter />
    </div>
  );
}
