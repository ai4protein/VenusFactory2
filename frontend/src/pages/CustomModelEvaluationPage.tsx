import { useEffect, useMemo, useState } from "react";
import {
  abortEvaluation,
  fetchDatasetConfigDefaults,
  fetchCustomModelMeta,
  fetchModelConfig,
  fetchModelFolders,
  fetchModelsInFolder,
  previewDataset,
  previewEvaluation,
  startEvaluationStream,
  uploadCustomModelDatasetFile,
  type CustomModelMeta,
  type DatasetConfigDefaults,
  type DatasetPreviewResult,
  type ModelOption
} from "../lib/customModelApi";
import { SegmentedSwitch } from "../components/SegmentedSwitch";
import { PageFooter } from "../components/PageFooter";
import { WorkspaceFilePicker } from "../components/WorkspaceFilePicker";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const DEFAULT_METRICS = ["accuracy", "mcc", "f1", "precision", "recall", "auroc"];
const STRUCTURE_MODELS = ["protssn", "prosst", "saprot"];
const SES_STRUCTURE_COLUMNS = ["foldseek_seq", "ss8_seq"];

const STRINGS = {
  en: {
    docTitle: "Evaluate — VenusFactory2",
    docDescription: "Evaluate trained checkpoints on protein datasets.",
    headerTitle: "Custom Model Evaluation",
    headerSubtitle: "Evaluate trained models with configurable metrics and transparent commands.",
    readonlyBanner: "Online mode: custom model controls are view-only in this deployment.",
    onlineUnavailable: "Online mode: unavailable",
    noOptions: "No options available",
    failedLoadMeta: "Failed to load metadata.",
    modelFolder: "Model Folder",
    modelPath: "Model Path",
    selectModel: "Select model",
    dataset: "Dataset",
    datasetMode: "Dataset mode switch",
    custom: "Custom",
    defaultDataset: "default",
    datasetPath: "Dataset Path",
    source: "Source",
    sourceMode: "Custom dataset source mode",
    hfPath: "HF Path",
    upload: "Upload",
    testFileHfId: "Test File / HF id",
    hfPlaceholder: "hf dataset id or local dataset path",
    testLabel: "Test",
    chooseFile: "Choose File",
    fromWorkspace: "From Workspace",
    useExample: "Use Example",
    previewDatasetBtn: "Preview Dataset",
    loading: "Loading...",
    datasetPreview: "Dataset Preview",
    sampleRows: "Sample Rows",
    noSampleRows: "No sample rows available.",
    noPreviewYet: "No preview yet.",
    evaluationParams: "Evaluation Params",
    datasetPresetsLocked: "Default dataset presets are loaded from dataset JSON and locked.",
    taskSettings: "Task Settings",
    plm: "PLM",
    evalMethod: "Eval Method",
    pooling: "Pooling",
    structureSeq: "Structure Seq",
    pdbFolder: "PDB Folder",
    structureRequirePdb: "Selected structure model requires PDB Folder.",
    problemType: "Problem Type",
    numLabels: "Num Labels",
    labelColumn: "Label Column",
    sequenceColumn: "Sequence Column",
    metrics: "Metrics",
    optimization: "Optimization",
    batchMode: "Batch Mode",
    batchSizeMode: "Batch Size Mode",
    batchTokenMode: "Batch Token Mode",
    batchSize: "Batch Size",
    batchToken: "Batch Token",
    previewCommand: "Preview Command",
    startEvaluation: "Start Evaluation",
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
    valuesAutoFilledPrefix: "Values auto-filled from selected checkpoint.",
    plmCfgFallback: "PLM missing in ckpt config, fallback to current/default value.",
    trainMethodCfgFallback: "training_method missing in ckpt config, fallback to default eval method.",
    poolingCfgFallback: "pooling_method missing in ckpt config, fallback to default pooling.",
    problemTypeCfgFallback: "problem_type missing in ckpt config, fallback to default problem type.",
    numLabelsCfgFallback: "num_labels missing in ckpt config, fallback to default label count.",
    metricsCfgFallback: "metrics missing in ckpt config, fallback to current/default metrics.",
    ckptConfigNotFound: "Checkpoint config not found. Parameters remain editable with current/default values.",
    errStructurePdb: "Structure PLM (ProSST/ProtSSN/SaProt) requires PDB Folder.",
    errSesNeedsCols: "ses-adapter requires selecting foldseek_seq and/or ss8_seq.",
    errSesTestCols: "ses-adapter requires selected structure columns in test file, or provide PDB Folder.",
    errFileUploadFailed: "File upload failed.",
    errDatasetPreviewFailed: "Dataset preview failed.",
    errPreviewFailed: "Preview failed.",
    errEvaluationFailed: "Evaluation failed.",
    errEvalStartFailed: "Evaluation start failed."
  },
  zh: {
    docTitle: "评估 — VenusFactory2",
    docDescription: "在蛋白质数据集上评估已训练的检查点。",
    headerTitle: "自定义模型评估",
    headerSubtitle: "使用可配置的评估指标和透明的命令评估已训练的模型。",
    readonlyBanner: "在线模式：当前部署下自定义模型控件仅供查看。",
    onlineUnavailable: "在线模式：不可用",
    noOptions: "暂无可选项",
    failedLoadMeta: "加载元数据失败。",
    modelFolder: "模型文件夹",
    modelPath: "模型路径",
    selectModel: "请选择模型",
    dataset: "数据集",
    datasetMode: "数据集模式切换",
    custom: "自定义",
    defaultDataset: "默认",
    datasetPath: "数据集路径",
    source: "来源",
    sourceMode: "自定义数据集来源模式",
    hfPath: "HF 路径",
    upload: "上传",
    testFileHfId: "测试文件 / HF id",
    hfPlaceholder: "HF 数据集 id 或本地数据集路径",
    testLabel: "测试",
    chooseFile: "选择文件",
    fromWorkspace: "从工作区",
    useExample: "使用示例",
    previewDatasetBtn: "预览数据集",
    loading: "加载中…",
    datasetPreview: "数据集预览",
    sampleRows: "样本行",
    noSampleRows: "暂无样本行。",
    noPreviewYet: "暂无预览。",
    evaluationParams: "评估参数",
    datasetPresetsLocked: "已从数据集 JSON 加载默认预设并锁定。",
    taskSettings: "任务设置",
    plm: "PLM",
    evalMethod: "评估方法",
    pooling: "池化方式",
    structureSeq: "结构序列",
    pdbFolder: "PDB 文件夹",
    structureRequirePdb: "所选结构模型需要提供 PDB 文件夹。",
    problemType: "任务类型",
    numLabels: "类别数",
    labelColumn: "标签列",
    sequenceColumn: "序列列",
    metrics: "评估指标",
    optimization: "优化设置",
    batchMode: "批处理模式",
    batchSizeMode: "批次大小模式",
    batchTokenMode: "批次 Token 模式",
    batchSize: "批次大小",
    batchToken: "批次 Token 数",
    previewCommand: "预览命令",
    startEvaluation: "开始评估",
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
    valuesAutoFilledPrefix: "已根据所选 checkpoint 自动填充。",
    plmCfgFallback: "ckpt 配置缺少 PLM，回退至当前/默认值。",
    trainMethodCfgFallback: "ckpt 配置缺少 training_method，回退至默认评估方法。",
    poolingCfgFallback: "ckpt 配置缺少 pooling_method，回退至默认池化方式。",
    problemTypeCfgFallback: "ckpt 配置缺少 problem_type，回退至默认任务类型。",
    numLabelsCfgFallback: "ckpt 配置缺少 num_labels，回退至默认类别数。",
    metricsCfgFallback: "ckpt 配置缺少 metrics，回退至当前/默认指标。",
    ckptConfigNotFound: "未找到 checkpoint 配置。参数仍可编辑，使用当前/默认值。",
    errStructurePdb: "结构 PLM（ProSST/ProtSSN/SaProt）需要提供 PDB 文件夹。",
    errSesNeedsCols: "ses-adapter 需要勾选 foldseek_seq 和/或 ss8_seq。",
    errSesTestCols: "ses-adapter 需要测试文件中包含所选结构列，或提供 PDB 文件夹。",
    errFileUploadFailed: "文件上传失败。",
    errDatasetPreviewFailed: "数据集预览失败。",
    errPreviewFailed: "预览失败。",
    errEvaluationFailed: "评估失败。",
    errEvalStartFailed: "评估启动失败。"
  }
};

function isNotFoundLikeError(message: string): boolean {
  const text = String(message || "").toLowerCase();
  return text.includes("404") || text.includes("not found") || text.includes('{"detail":"not found"}');
}

type CustomModelEvaluationPageProps = {
  readonly?: boolean;
  workspaceEnabled?: boolean;
};

export function CustomModelEvaluationPage({ readonly = false, workspaceEnabled = false }: CustomModelEvaluationPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: t.docTitle, description: t.docDescription });
  const [meta, setMeta] = useState<CustomModelMeta | null>(null);
  const [datasetSelection, setDatasetSelection] = useState<"Custom" | "Pre-defined">("Pre-defined");
  const [customDataSourceMode, setCustomDataSourceMode] = useState<"hf_local" | "upload">("hf_local");
  const [datasetConfig, setDatasetConfig] = useState("");
  const [datasetCustom, setDatasetCustom] = useState("");
  const [testFile, setTestFile] = useState("");
  const [columnOptions, setColumnOptions] = useState<string[]>([]);
  const [problemType, setProblemType] = useState("single_label_classification");
  const [numLabels, setNumLabels] = useState(2);
  const [metrics, setMetrics] = useState<string[]>(DEFAULT_METRICS);
  const [sequenceColumn, setSequenceColumn] = useState("aa_seq");
  const [labelColumn, setLabelColumn] = useState("label");
  const [plmModel, setPlmModel] = useState("");
  const [evalMethod, setEvalMethod] = useState("full");
  const [poolingMethod, setPoolingMethod] = useState("mean");
  const [batchMode, setBatchMode] = useState<"Batch Size Mode" | "Batch Token Mode">("Batch Size Mode");
  const [batchSize, setBatchSize] = useState(1);
  const [batchToken, setBatchToken] = useState(2000);
  const [structureSeq, setStructureSeq] = useState<string[]>([]);
  const [pdbDir, setPdbDir] = useState("");
  const [folderOptions, setFolderOptions] = useState<string[]>(["ckpt"]);
  const [selectedFolder, setSelectedFolder] = useState("ckpt");
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [modelPath, setModelPath] = useState("");

  const [running, setRunning] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreviewResult | null>(null);
  const [datasetRowsExpanded, setDatasetRowsExpanded] = useState(false);
  const [commandPreview, setCommandPreview] = useState("");
  const [error, setError] = useState("");
  const [ckptLocked, setCkptLocked] = useState(false);
  const [ckptConfigNote, setCkptConfigNote] = useState("");
  const [datasetDefaults, setDatasetDefaults] = useState<DatasetConfigDefaults | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const emptySelectLabel = readonly ? t.onlineUnavailable : t.noOptions;
  const structureSeqOptions = useMemo(() => meta?.structure_seq_options || [], [meta?.structure_seq_options]);
  const metricOptions = useMemo(
    () => (meta?.metrics_options && meta.metrics_options.length > 0 ? meta.metrics_options : DEFAULT_METRICS),
    [meta?.metrics_options]
  );
  const selectableColumns = useMemo(() => {
    const merged = new Set<string>([...columnOptions, sequenceColumn, labelColumn]);
    return Array.from(merged).filter(Boolean);
  }, [columnOptions, sequenceColumn, labelColumn]);
  const plmModelKeys = useMemo(() => Object.keys(meta?.plm_models || {}), [meta?.plm_models]);
  const datasetConfigKeys = useMemo(() => Object.keys(meta?.dataset_configs || {}), [meta?.dataset_configs]);
  const showStructureInputs = useMemo(() => {
    const method = String(evalMethod || "").toLowerCase();
    if (method === "ses-adapter" || method === "ses_adapter") return true;
    const modelHint = `${plmModel} ${modelPath}`.toLowerCase();
    return STRUCTURE_MODELS.some((key) => modelHint.includes(key));
  }, [evalMethod, plmModel, modelPath]);
  const isSesAdapter = useMemo(() => {
    const method = String(evalMethod || "").toLowerCase();
    return method === "ses-adapter" || method === "ses_adapter";
  }, [evalMethod]);
  const isStructurePlm = useMemo(() => {
    const modelHint = `${plmModel} ${modelPath}`.toLowerCase();
    return STRUCTURE_MODELS.some((key) => modelHint.includes(key));
  }, [plmModel, modelPath]);
  const structureSeqRequired = useMemo(
    () => structureSeq.filter((item) => SES_STRUCTURE_COLUMNS.includes(item)),
    [structureSeq]
  );
  const knownColumns = useMemo(() => {
    const previewCols = datasetPreview?.preview?.columns || [];
    return new Set<string>([...columnOptions, ...previewCols].map((x) => String(x || "").trim()).filter(Boolean));
  }, [columnOptions, datasetPreview]);
  const evaluationRuleError = useMemo(() => {
    if (isStructurePlm && !pdbDir.trim()) {
      return t.errStructurePdb;
    }
    if (isSesAdapter) {
      if (!structureSeqRequired.length) {
        return t.errSesNeedsCols;
      }
      if (!pdbDir.trim() && datasetSelection === "Custom" && structureSeqRequired.some((col) => !knownColumns.has(col))) {
        return t.errSesTestCols;
      }
    }
    return "";
  }, [isStructurePlm, pdbDir, isSesAdapter, structureSeqRequired, datasetSelection, knownColumns, t]);
  const effectiveDatasetCustom = useMemo(() => {
    if (datasetSelection !== "Custom" || customDataSourceMode !== "hf_local") return "";
    return datasetCustom;
  }, [datasetSelection, customDataSourceMode, datasetCustom]);
  const effectiveTestFile = useMemo(() => {
    if (datasetSelection !== "Custom" || customDataSourceMode !== "upload") return "";
    return testFile;
  }, [datasetSelection, customDataSourceMode, testFile]);
  const lockByDatasetConfig = datasetSelection === "Pre-defined";

  useEffect(() => {
    if (readonly) return;
    void (async () => {
      try {
        const data = await fetchCustomModelMeta();
        setMeta(data);
        setPlmModel(Object.keys(data.plm_models)[0] || "");
        setDatasetConfig(Object.keys(data.dataset_configs)[0] || "");
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
        const allowCkptDatasetOverwrite = datasetSelection === "Custom";
        const display = Object.entries(meta.plm_models).find(([, v]) => v === cfg.plm_model)?.[0];
        const notes: string[] = [];
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

        if (allowCkptDatasetOverwrite && typeof cfg.problem_type === "string") {
          setProblemType(cfg.problem_type);
        } else if (allowCkptDatasetOverwrite) {
          notes.push(t.problemTypeCfgFallback);
        }

        if (allowCkptDatasetOverwrite && typeof cfg.num_labels === "number") {
          setNumLabels(cfg.num_labels);
        } else if (allowCkptDatasetOverwrite) {
          notes.push(t.numLabelsCfgFallback);
        }

        const metricsCfg = cfg.metrics;
        if (allowCkptDatasetOverwrite && Array.isArray(metricsCfg)) {
          setMetrics(metricsCfg as string[]);
        } else if (allowCkptDatasetOverwrite && typeof metricsCfg === "string") {
          setMetrics(
            metricsCfg
              .split(",")
              .map((x) => x.trim())
              .filter(Boolean)
          );
        } else if (allowCkptDatasetOverwrite) {
          notes.push(t.metricsCfgFallback);
        }

        if (allowCkptDatasetOverwrite && typeof cfg.sequence_column_name === "string") {
          setSequenceColumn(cfg.sequence_column_name);
        }
        if (allowCkptDatasetOverwrite && typeof cfg.label_column_name === "string") {
          setLabelColumn(cfg.label_column_name);
        }

        const structureFromCfg = cfg.structure_seq;
        if (allowCkptDatasetOverwrite && Array.isArray(structureFromCfg)) {
          setStructureSeq(structureFromCfg.filter((x): x is string => typeof x === "string"));
        } else if (allowCkptDatasetOverwrite && typeof structureFromCfg === "string") {
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
            ? `${t.valuesAutoFilledPrefix} ${notes.join(" ")}`
            : ""
        );
      } catch {
        setCkptLocked(false);
        setCkptConfigNote(t.ckptConfigNotFound);
      }
    })();
  }, [modelPath, meta, datasetSelection, readonly]);

  useEffect(() => {
    if (readonly) return;
    if (datasetSelection !== "Pre-defined" || !datasetConfig) {
      setDatasetDefaults(null);
      return;
    }
    void (async () => {
      try {
        const cfg = await fetchDatasetConfigDefaults(datasetConfig);
        setDatasetDefaults(cfg);
        if (typeof cfg.problem_type === "string") setProblemType(cfg.problem_type);
        if (typeof cfg.num_labels === "number") setNumLabels(cfg.num_labels);
        if (Array.isArray(cfg.metrics) && cfg.metrics.length > 0) setMetrics(cfg.metrics);
        if (typeof cfg.sequence_column_name === "string") setSequenceColumn(cfg.sequence_column_name);
        if (typeof cfg.label_column_name === "string") setLabelColumn(cfg.label_column_name);
        if (Array.isArray(cfg.structure_seq)) setStructureSeq(cfg.structure_seq);
        if (typeof cfg.pdb_dir === "string") setPdbDir(cfg.pdb_dir);
      } catch {
        setDatasetDefaults(null);
      }
    })();
  }, [datasetSelection, datasetConfig, readonly]);

  useEffect(() => {
    if (datasetSelection === "Pre-defined") {
      setTestFile("");
      setColumnOptions([]);
      setCustomDataSourceMode("hf_local");
    }
  }, [datasetSelection]);

  useEffect(() => {
    setDatasetRowsExpanded(false);
  }, [datasetPreview]);
  const visibleError = readonly && isNotFoundLikeError(error) ? "" : error;

  const args = useMemo(
    () => ({
      plm_model: plmModel,
      model_path: modelPath,
      eval_method: evalMethod,
      dataset_selection: datasetSelection,
      dataset_config: datasetConfig,
      dataset_custom: effectiveDatasetCustom,
      test_file: effectiveTestFile,
      problem_type: problemType,
      num_labels: numLabels,
      metrics,
      pooling_method: poolingMethod,
      sequence_column_name: sequenceColumn,
      label_column_name: labelColumn,
      batch_mode: batchMode,
      batch_size: batchSize,
      batch_token: batchToken,
      structure_seq: structureSeq,
      pdb_dir: pdbDir
    }),
    [
      plmModel,
      modelPath,
      evalMethod,
      datasetSelection,
      customDataSourceMode,
      datasetConfig,
      datasetCustom,
      testFile,
      problemType,
      numLabels,
      metrics,
      poolingMethod,
      sequenceColumn,
      labelColumn,
      batchMode,
      batchSize,
      batchToken,
      structureSeq,
      pdbDir
    ]
  );

  async function onUploadTestFile(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const result = await uploadCustomModelDatasetFile(file);
      setTestFile(result.file_path);
      if (Array.isArray(result.columns)) {
        setColumnOptions(result.columns.filter(Boolean));
        if (result.columns.includes("aa_seq")) setSequenceColumn("aa_seq");
        if (result.columns.includes("label")) setLabelColumn("label");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errFileUploadFailed);
    }
  }

  async function onUseTestExample() {
    if (readonly) return;
    const content = "aa_seq,label\nMKTAYIAKQRQISFVKSHFSRQ,1\nGAVLILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSR,0\n";
    const file = new File([content], "test_example.csv", { type: "text/csv" });
    await onUploadTestFile(file);
  }

  function displayUploadedName(pathValue: string) {
    const normalized = String(pathValue || "").trim();
    if (!normalized) return t.noFileSelected;
    return normalized.split("/").pop() || normalized;
  }

  async function onPreviewDataset() {
    setError("");
    setPreviewLoading(true);
    try {
      const data = await previewDataset({
        dataset_selection: datasetSelection,
        dataset_config: datasetConfig,
        dataset_custom: effectiveDatasetCustom,
        test_file: effectiveTestFile
      });
      setDatasetPreview(data);
      if (Array.isArray(data.column_options)) {
        setColumnOptions(data.column_options.filter(Boolean));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errDatasetPreviewFailed);
    } finally {
      setPreviewLoading(false);
    }
  }

  async function onPreviewCommand() {
    if (evaluationRuleError) {
      setError(evaluationRuleError);
      return;
    }
    try {
      const result = await previewEvaluation(args);
      setCommandPreview(result.command);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errPreviewFailed);
    }
  }

  async function onStart() {
    if (evaluationRuleError) {
      setError(evaluationRuleError);
      return;
    }
    setRunning(true);
    setStatusText(t.statusStarting);
    setProgress(0);
    setLogs([]);
    try {
      await startEvaluationStream(args, (evt) => {
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
        if (evt.type === "error") setError(evt.data.message || t.errEvaluationFailed);
        if (evt.type === "done") {
          if (typeof evt.data.final_progress === "number") {
            setProgress((prev) => Math.max(prev, evt.data.final_progress));
          }
          setStatusText(evt.data.message || (evt.data.success ? t.statusCompleted : t.statusFailed));
          setProgress((prev) => (evt.data.success ? 1 : prev));
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errEvalStartFailed);
      setStatusText(t.statusFailed);
    } finally {
      setRunning(false);
    }
  }

  async function onAbort() {
    await abortEvaluation();
    setStatusText(t.statusAborted);
    setRunning(false);
  }

  function toggleStructureSeqOption(option: string) {
    setStructureSeq((prev) => (prev.includes(option) ? prev.filter((x) => x !== option) : [...prev, option]));
  }

  function toggleMetric(metric: string) {
    setMetrics((prev) => (prev.includes(metric) ? prev.filter((x) => x !== metric) : [...prev, metric]));
  }

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
          <div className="custom-top-line-grid">
            <section className="custom-section-card custom-section-line custom-section-line-eval-model custom-section-line-top">
              <label className="left-controls custom-line-field">{t.modelFolder}
                <select value={selectedFolder} onChange={(e) => setSelectedFolder(e.target.value)}>
                  {folderOptions.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              </label>
              <label className="left-controls custom-line-field">{t.modelPath}
                <select value={modelPath} onChange={(e) => setModelPath(e.target.value)}>
                  <option value="">{t.selectModel}</option>
                  {modelOptions.map((m) => <option key={m.path} value={m.path}>{m.label}</option>)}
                </select>
              </label>
            </section>

            <section className="custom-section-card custom-section-line custom-section-line-dataset custom-section-line-top">
              <div className="custom-dataset-line-main">
                <label className="left-controls custom-line-field custom-dataset-main-dataset">{t.dataset}
                  <SegmentedSwitch
                    value={datasetSelection}
                    onChange={setDatasetSelection}
                    ariaLabel={t.datasetMode}
                    className="custom-segment-switch-wide"
                    options={[
                      { value: "Custom", label: t.custom },
                      { value: "Pre-defined", label: t.defaultDataset }
                    ]}
                  />
                </label>
                <div className="custom-dataset-line-inputs custom-dataset-main-inputs">
                  {datasetSelection === "Pre-defined" ? (
                    <label className="left-controls custom-line-field">{t.datasetPath}
                      <select value={datasetConfig} onChange={(e) => setDatasetConfig(e.target.value)}>
                        {datasetConfigKeys.length === 0 && <option value="">{emptySelectLabel}</option>}
                        {datasetConfigKeys.map((k) => <option key={k} value={k}>{k}</option>)}
                      </select>
                    </label>
                  ) : (
                    <>
                      <label className="left-controls custom-line-field">
                        {t.source}
                        <SegmentedSwitch
                          value={customDataSourceMode}
                          onChange={(value) => setCustomDataSourceMode(value as "hf_local" | "upload")}
                          ariaLabel={t.sourceMode}
                          className="custom-segment-switch-wide"
                          options={[
                            { value: "hf_local", label: t.hfPath },
                            { value: "upload", label: t.upload }
                          ]}
                        />
                      </label>
                      {customDataSourceMode === "hf_local" ? (
                        <label className="left-controls custom-line-field">{t.testFileHfId}
                          <input
                            value={datasetCustom}
                            onChange={(e) => setDatasetCustom(e.target.value)}
                            placeholder={t.hfPlaceholder}
                          />
                        </label>
                      ) : (
                        <div className="custom-upload-dropzone-wrap">
                          <div className="custom-upload-dropzone-grid custom-upload-dropzone-grid-single">
                            <div className="custom-upload-item upload-source-stack">
                              <span className="custom-upload-item-label">{t.testLabel}</span>
                              <label className="custom-upload-trigger">
                                <input
                                  type="file"
                                  accept=".csv,.tsv,.xlsx,.xls"
                                  onChange={(e) => void onUploadTestFile(e.target.files?.[0] || null)}
                                />
                                {t.chooseFile}
                              </label>
                              <WorkspaceFilePicker
                                workspaceEnabled={workspaceEnabled}
                                disabled={readonly || running}
                                acceptedCategories={["table_or_text"]}
                                buttonLabel={t.fromWorkspace}
                                onPick={(picked) => {
                                  const selected = picked[0];
                                  if (!selected) return;
                                  setTestFile(selected.storage_path);
                                }}
                              />
                              <button
                                type="button"
                                className="custom-btn-secondary"
                                onClick={() => void onUseTestExample()}
                                disabled={readonly || running}
                              >
                                {t.useExample}
                              </button>
                              <span className="custom-upload-file-chip custom-upload-file-name" title={displayUploadedName(testFile)}>
                                {displayUploadedName(testFile)}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
                <div className="custom-dataset-preview-slot">
                  <button type="button" className="custom-btn-secondary custom-line-action" onClick={() => void onPreviewDataset()} disabled={previewLoading}>
                    {previewLoading ? t.loading : t.previewDatasetBtn}
                  </button>
                </div>
              </div>
            </section>
          </div>
          </fieldset>
        </aside>

        <section className="custom-bottom-split">
          <aside className="chat-panel left custom-bottom-left custom-model-controls">
            <fieldset className="readonly-fieldset" disabled={readonly}>
            <section className="custom-section-card custom-section-wide custom-dataset-preview-card">
              <div className="custom-dataset-preview-header">
                <h3 className="custom-panel-title custom-dataset-preview-title">{t.datasetPreview}</h3>
                {datasetPreview && (
                  <div className="custom-dataset-preview-controls">
                    <span className="custom-dataset-preview-stats-compact" aria-label={`Train ${datasetPreview.stats.train}, Validation ${datasetPreview.stats.validation}, Test ${datasetPreview.stats.test}`}>
                      {datasetPreview.stats.train}/{datasetPreview.stats.validation}/{datasetPreview.stats.test}
                    </span>
                    {datasetPreview.preview.columns.length > 0 && datasetPreview.preview.rows.length > 0 && (
                      <button
                        type="button"
                        className="custom-preview-toggle-btn"
                        onClick={() => setDatasetRowsExpanded((prev) => !prev)}
                        aria-expanded={datasetRowsExpanded}
                      >
                        {datasetRowsExpanded ? "▾" : "▸"} {t.sampleRows} ({datasetPreview.preview.rows.length})
                      </button>
                    )}
                  </div>
                )}
              </div>
              {datasetPreview ? (
                <>
                  {datasetPreview.preview.columns.length > 0 && datasetPreview.preview.rows.length > 0 ? (
                    <div className={`custom-preview-body ${datasetRowsExpanded ? "expanded" : "collapsed"}`}>
                      {datasetRowsExpanded ? (
                        <div className="custom-table-wrap">
                          <table className="custom-preview-table">
                            <thead>
                              <tr>
                                {datasetPreview.preview.columns.map((col) => (
                                  <th key={col}>{col}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {datasetPreview.preview.rows.map((row, idx) => (
                                <tr key={idx}>
                                  {datasetPreview.preview.columns.map((col) => (
                                    <td key={`${idx}-${col}`}>{String(row[col] ?? "-")}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <p>{t.noSampleRows}</p>
                  )}
                </>
              ) : (
                <p>{t.noPreviewYet}</p>
              )}
            </section>

            <section className="custom-section-card custom-section-wide">
              <h3 className="custom-panel-title">{t.evaluationParams}</h3>
              <div className="custom-advanced-grid custom-advanced-panel">
                {ckptConfigNote && <div className="custom-readonly-note custom-field-span-full">{ckptConfigNote}</div>}
                {lockByDatasetConfig && (
                  <div className="custom-readonly-note custom-field-span-full">
                    {t.datasetPresetsLocked}
                  </div>
                )}
                <div className="custom-param-group-title custom-field-span-full">{t.taskSettings}</div>
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
                {showStructureInputs && (
                  <>
                    <div className="left-controls custom-field-span-2">
                      <span>{t.structureSeq}</span>
                      <div className="custom-multi-grid">
                        {structureSeqOptions.map((item) => {
                          const checked = structureSeq.includes(item);
                          return (
                            <button
                              key={item}
                              type="button"
                              className={`custom-multi-item ${checked ? "active" : ""}`}
                              aria-pressed={checked}
                              onClick={() => toggleStructureSeqOption(item)}
                              disabled={lockByDatasetConfig}
                            >
                              <span className="custom-multi-item-label">{item}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <label className="left-controls custom-field-short">{t.pdbFolder}<input value={pdbDir} onChange={(e) => setPdbDir(e.target.value)} disabled={lockByDatasetConfig && Boolean(datasetDefaults?.pdb_dir)} /></label>
                  </>
                )}
                {isStructurePlm && (
                  <div className="custom-readonly-note custom-field-span-full">
                    {t.structureRequirePdb}
                  </div>
                )}
                {evaluationRuleError && <div className="custom-readonly-note custom-field-span-full">{evaluationRuleError}</div>}
                <div className="custom-field-span-full" />
                <label className="left-controls custom-field-short">{t.problemType}
                  <select value={problemType} onChange={(e) => setProblemType(e.target.value)} disabled={lockByDatasetConfig}>
                    {(meta?.problem_types || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                    {(meta?.problem_types || []).map((x) => <option key={x} value={x}>{x}</option>)}
                  </select>
                </label>
                <label className="left-controls custom-field-short">{t.numLabels}<input type="number" value={numLabels} onChange={(e) => setNumLabels(Number(e.target.value) || 1)} disabled={lockByDatasetConfig} /></label>
                <label className="left-controls custom-field-short">{t.labelColumn}
                  {selectableColumns.length > 0 && !lockByDatasetConfig ? (
                    <select value={labelColumn} onChange={(e) => setLabelColumn(e.target.value)}>
                      {selectableColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  ) : (
                    <input value={labelColumn} onChange={(e) => setLabelColumn(e.target.value)} disabled={lockByDatasetConfig} />
                  )}
                </label>
                <label className="left-controls custom-field-short">{t.sequenceColumn}
                  {selectableColumns.length > 0 && !lockByDatasetConfig ? (
                    <select value={sequenceColumn} onChange={(e) => setSequenceColumn(e.target.value)}>
                      {selectableColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  ) : (
                    <input value={sequenceColumn} onChange={(e) => setSequenceColumn(e.target.value)} disabled={lockByDatasetConfig} />
                  )}
                </label>
                <div className="left-controls custom-field-span-full">
                  <span>{t.metrics}</span>
                  <div className="custom-multi-grid">
                    {metricOptions.map((item) => {
                      const checked = metrics.includes(item);
                      return (
                        <button
                          key={item}
                          type="button"
                          className={`custom-multi-item ${checked ? "active" : ""}`}
                          aria-pressed={checked}
                          onClick={() => toggleMetric(item)}
                          disabled={lockByDatasetConfig}
                        >
                          <span className="custom-multi-item-label">{item}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="custom-param-group-title custom-field-span-full">{t.optimization}</div>
                <label className="left-controls custom-field-short">{t.batchMode}
                  <select value={batchMode} onChange={(e) => setBatchMode(e.target.value as "Batch Size Mode" | "Batch Token Mode")}>
                    <option value="Batch Size Mode">{t.batchSizeMode}</option>
                    <option value="Batch Token Mode">{t.batchTokenMode}</option>
                  </select>
                </label>
                {batchMode === "Batch Size Mode" ? (
                  <label className="left-controls custom-field-short">{t.batchSize}<input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value) || 1)} /></label>
                ) : (
                  <label className="left-controls custom-field-short">{t.batchToken}<input type="number" value={batchToken} onChange={(e) => setBatchToken(Number(e.target.value) || 1000)} /></label>
                )}
              </div>

              <div className="custom-actions-inline-status">
                <div className="custom-row custom-actions">
                  <button type="button" className="custom-btn-secondary" onClick={() => void onPreviewCommand()}>{t.previewCommand}</button>
                  <button type="button" className="custom-btn-primary" disabled={running} onClick={() => void onStart()}>{t.startEvaluation}</button>
                  <button type="button" className="custom-btn-danger" disabled={!running} onClick={() => void onAbort()}>{t.abort}</button>
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
