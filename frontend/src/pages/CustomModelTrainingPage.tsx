import { useEffect, useMemo, useState } from "react";
import {
  abortTraining,
  fetchCustomModelMeta,
  fetchDatasetConfigDefaults,
  fetchModelConfig,
  fetchModelFolders,
  fetchModelsInFolder,
  previewDataset,
  previewTraining,
  startTrainingStream,
  uploadCustomModelDatasetFile,
  type CustomModelMeta,
  type DatasetConfigDefaults,
  type DatasetPreviewResult,
  type ModelOption
} from "../lib/customModelApi";
import { SegmentedSwitch } from "../components/SegmentedSwitch";
import { PageFooter } from "../components/PageFooter";
import { WorkspaceFilePicker } from "../components/WorkspaceFilePicker";

const DEFAULT_METRICS = ["accuracy", "mcc", "f1", "precision", "recall", "auroc"];
const METRIC_COLORS = ["#4f77b3", "#4e9d76", "#b57a42", "#8b63a8", "#af4f6f", "#7a7a7a", "#2f8fdd", "#b07a1f"];
const STRUCTURE_MODELS = ["protssn", "prosst", "saprot"];
const SES_STRUCTURE_COLUMNS = ["foldseek_seq", "ss8_seq"];

function isNotFoundLikeError(message: string): boolean {
  const text = String(message || "").toLowerCase();
  return text.includes("404") || text.includes("not found") || text.includes('{"detail":"not found"}');
}

type LineSeries = { name: string; values: Array<number | null>; color: string };

function lastNumeric(values: Array<number | null>): number | null {
  for (let i = values.length - 1; i >= 0; i -= 1) {
    if (typeof values[i] === "number" && Number.isFinite(values[i])) return values[i] as number;
  }
  return null;
}

function formatMetricValue(v: number | null) {
  if (v == null || !Number.isFinite(v)) return "-";
  if (Math.abs(v) >= 100) return v.toFixed(2);
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toFixed(6);
}

function remapSeriesByEpoch(
  nextEpochs: number[],
  prevEpochs: number[],
  prevValues: Array<number | null>
): Array<number | null> {
  return nextEpochs.map((ep) => {
    const oldIdx = prevEpochs.indexOf(ep);
    return oldIdx >= 0 ? (prevValues[oldIdx] ?? null) : null;
  });
}

function mergeEpochs(prevEpochs: number[], incomingEpochs: number[]): number[] {
  const merged = new Set<number>([...prevEpochs, ...incomingEpochs]);
  return Array.from(merged).sort((a, b) => a - b);
}

function parseEpochMetricLogLine(line: string): {
  epoch: number;
  trainLoss?: number;
  valLoss?: number;
  valMetricName?: string;
  valMetricValue?: number;
} | null {
  const cleanedLine = line.replace(/\r/g, "").replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, "");
  const trainMatch = cleanedLine.match(/Epoch\s+(\d+)(?:\s*\/\s*\d+)?\s+Train\s+Loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/i);
  if (trainMatch) {
    const epoch = Number(trainMatch[1]);
    const trainLoss = Number(trainMatch[2]);
    if (Number.isFinite(epoch) && Number.isFinite(trainLoss)) {
      return { epoch, trainLoss };
    }
    return null;
  }
  const valLossMatch = cleanedLine.match(/Epoch\s+(\d+)(?:\s*\/\s*\d+)?\s+Val\s+Loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/i);
  if (valLossMatch) {
    const epoch = Number(valLossMatch[1]);
    const valLoss = Number(valLossMatch[2]);
    if (Number.isFinite(epoch) && Number.isFinite(valLoss)) {
      return { epoch, valLoss };
    }
    return null;
  }
  const valMetricMatch = cleanedLine.match(
    /Epoch\s+(\d+)(?:\s*\/\s*\d+)?\s+Val\s+([A-Za-z0-9_\-]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/i
  );
  if (!valMetricMatch) return null;
  const epoch = Number(valMetricMatch[1]);
  const valMetricName = String(valMetricMatch[2] || "").toLowerCase();
  const valMetricValue = Number(valMetricMatch[3]);
  if (!Number.isFinite(epoch) || !valMetricName || !Number.isFinite(valMetricValue)) return null;
  return { epoch, valMetricName, valMetricValue };
}

function isTrainingMetricsPayload(
  data: unknown
): data is { epochs: number[]; train_loss: Array<number | null>; val_loss: Array<number | null>; val_metrics: Record<string, Array<number | null>> } {
  if (!data || typeof data !== "object") return false;
  const d = data as Record<string, unknown>;
  if (!Array.isArray(d.epochs) || !Array.isArray(d.train_loss) || !Array.isArray(d.val_loss)) return false;
  if (!d.val_metrics || typeof d.val_metrics !== "object" || Array.isArray(d.val_metrics)) return false;
  return true;
}

function SimpleLineChart({
  title,
  epochs,
  series
}: {
  title: string;
  epochs: number[];
  series: LineSeries[];
}) {
  const width = 640;
  const height = 236;
  const paddingLeft = 54;
  const paddingRight = 16;
  const paddingTop = 12;
  const paddingBottom = 34;
  const axisTickFontSize = 13;
  const axisLabelFontSize = 13;
  const svgHeightPx = 228;
  const drawableW = width - paddingLeft - paddingRight;
  const drawableH = height - paddingTop - paddingBottom;
  const validValues = series.flatMap((s) => s.values.filter((v): v is number => typeof v === "number"));
  const yMin = validValues.length ? Math.min(...validValues) : 0;
  const yMax = validValues.length ? Math.max(...validValues) : 1;
  const yRange = yMax - yMin <= 1e-8 ? 1 : yMax - yMin;
  const xCount = Math.max(1, epochs.length - 1);
  const yTickCount = 5;
  const yTicks = Array.from({ length: yTickCount }, (_, idx) => {
    const ratio = idx / (yTickCount - 1);
    return yMax - ratio * yRange;
  });
  const xTickMax = 5;
  const xTickCount = Math.min(xTickMax, Math.max(2, epochs.length || 2));
  const xTickIndices = Array.from({ length: xTickCount }, (_, idx) => {
    if (epochs.length <= 1) return 0;
    return Math.round((idx / (xTickCount - 1)) * (epochs.length - 1));
  }).filter((v, i, arr) => arr.indexOf(v) === i);

  const formatAxisValue = (value: number) => {
    if (!Number.isFinite(value)) return "";
    if (Math.abs(value) >= 100) return value.toFixed(1);
    if (Math.abs(value) >= 10) return value.toFixed(2);
    if (Math.abs(value) >= 1) return value.toFixed(3);
    return value.toFixed(4);
  };

  const pointsForSeries = (values: Array<number | null>) =>
    values
      .map((v, idx) => {
        if (typeof v !== "number") return null;
        const x = paddingLeft + (idx / xCount) * drawableW;
        const y = paddingTop + (1 - (v - yMin) / yRange) * drawableH;
        return `${x},${y}`;
      })
      .filter(Boolean)
      .join(" ");

  const circlesForSeries = (values: Array<number | null>) =>
    values
      .map((v, idx) => {
        if (typeof v !== "number") return null;
        const x = paddingLeft + (idx / xCount) * drawableW;
        const y = paddingTop + (1 - (v - yMin) / yRange) * drawableH;
        return { x, y };
      })
      .filter((p): p is { x: number; y: number } => p !== null);

  return (
    <div className="custom-chart-card">
      <div className="custom-chart-title">{title}</div>
      <svg
        className="custom-chart-svg"
        style={{ height: `${svgHeightPx}px` }}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        aria-label={title}
      >
        {yTicks.map((tick) => {
          const y = paddingTop + (1 - (tick - yMin) / yRange) * drawableH;
          return (
            <g key={`y-${tick}`}>
              <line x1={paddingLeft} y1={y} x2={width - paddingRight} y2={y} stroke="rgba(90,72,50,0.12)" />
              <text x={paddingLeft - 8} y={y + 4} textAnchor="end" fontSize={axisTickFontSize} fill="#776551">
                {formatAxisValue(tick)}
              </text>
            </g>
          );
        })}
        {xTickIndices.map((idx) => {
          const x = paddingLeft + (idx / xCount) * drawableW;
          const label = epochs[idx] ?? idx;
          return (
            <g key={`x-${idx}`}>
              <line x1={x} y1={height - paddingBottom} x2={x} y2={height - paddingBottom + 4} stroke="rgba(90,72,50,0.35)" />
              <text x={x} y={height - 10} textAnchor="middle" fontSize={axisTickFontSize} fill="#776551">
                {label}
              </text>
            </g>
          );
        })}
        <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} stroke="rgba(90,72,50,0.35)" />
        <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={height - paddingBottom} stroke="rgba(90,72,50,0.35)" />
        {series.map((s) => {
          const circles = circlesForSeries(s.values);
          return (
            <g key={s.name}>
              <polyline fill="none" stroke={s.color} strokeWidth="2.2" points={pointsForSeries(s.values)} />
              {circles.map((pt, idx) => (
                <circle key={`${s.name}-pt-${idx}`} cx={pt.x} cy={pt.y} r="2.4" fill={s.color} />
              ))}
            </g>
          );
        })}
        <text x={(paddingLeft + (width - paddingRight)) / 2} y={height - 4} textAnchor="middle" fontSize={axisLabelFontSize} fill="#6a5743">
          Epoch
        </text>
      </svg>
      <div className="custom-chart-legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}

type CustomModelTrainingPageProps = {
  readonly?: boolean;
  workspaceEnabled?: boolean;
};

export function CustomModelTrainingPage({ readonly = false, workspaceEnabled = false }: CustomModelTrainingPageProps) {
  const [meta, setMeta] = useState<CustomModelMeta | null>(null);
  const [trainingMode, setTrainingMode] = useState<"From Scratch" | "Continue Training">("From Scratch");
  const [datasetSelection, setDatasetSelection] = useState<"Custom" | "Pre-defined">("Pre-defined");
  const [customDataSourceMode, setCustomDataSourceMode] = useState<"hf_local" | "upload">("hf_local");
  const [datasetConfig, setDatasetConfig] = useState("");
  const [datasetCustom, setDatasetCustom] = useState("");
  const [trainFile, setTrainFile] = useState("");
  const [validFile, setValidFile] = useState("");
  const [testFile, setTestFile] = useState("");
  const [columnOptions, setColumnOptions] = useState<string[]>([]);
  const [problemType, setProblemType] = useState("single_label_classification");
  const [numLabels, setNumLabels] = useState(2);
  const [metrics, setMetrics] = useState<string[]>(DEFAULT_METRICS);
  const [sequenceColumn, setSequenceColumn] = useState("aa_seq");
  const [labelColumn, setLabelColumn] = useState("label");
  const [plmModel, setPlmModel] = useState("");
  const [trainingMethod, setTrainingMethod] = useState("full");
  const [poolingMethod, setPoolingMethod] = useState("mean");
  const [batchMode, setBatchMode] = useState<"Batch Size Mode" | "Batch Token Mode">("Batch Size Mode");
  const [batchSize, setBatchSize] = useState(8);
  const [batchToken, setBatchToken] = useState(4000);
  const [learningRate, setLearningRate] = useState(0.0005);
  const [numEpochs, setNumEpochs] = useState(20);
  const [maxSeqLen, setMaxSeqLen] = useState(1024);
  const [gradAccumulation, setGradAccumulation] = useState(1);
  const [warmupSteps, setWarmupSteps] = useState(0);
  const [scheduler, setScheduler] = useState("linear");
  const [outputModelName, setOutputModelName] = useState("best_model.pt");
  const [outputDir, setOutputDir] = useState("demo");
  const [wandbEnabled, setWandbEnabled] = useState(false);
  const [wandbProject, setWandbProject] = useState("");
  const [wandbEntity, setWandbEntity] = useState("");
  const [patience, setPatience] = useState(10);
  const [numWorkers, setNumWorkers] = useState(4);
  const [maxGradNorm, setMaxGradNorm] = useState(-1);
  const [structureSeq, setStructureSeq] = useState<string[]>([]);
  const [pdbDir, setPdbDir] = useState("");
  const [loraR, setLoraR] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [loraDropout, setLoraDropout] = useState(0.1);
  const [loraTargetModules, setLoraTargetModules] = useState("query,key,value");
  const [monitoredMetrics, setMonitoredMetrics] = useState("accuracy");
  const [monitoredStrategy, setMonitoredStrategy] = useState("max");

  const [folderOptions, setFolderOptions] = useState<string[]>([]);
  const [selectedFolder, setSelectedFolder] = useState("ckpt");
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [selectedModelPath, setSelectedModelPath] = useState("");

  const [datasetPreview, setDatasetPreview] = useState<DatasetPreviewResult | null>(null);
  const [datasetRowsExpanded, setDatasetRowsExpanded] = useState(false);
  const [commandPreview, setCommandPreview] = useState("");
  const [running, setRunning] = useState(false);
  const [statusText, setStatusText] = useState("Idle");
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [curveEpochs, setCurveEpochs] = useState<number[]>([]);
  const [trainLossSeries, setTrainLossSeries] = useState<Array<number | null>>([]);
  const [valLossSeries, setValLossSeries] = useState<Array<number | null>>([]);
  const [valMetricSeries, setValMetricSeries] = useState<Record<string, Array<number | null>>>({});
  const [testMetrics, setTestMetrics] = useState<Record<string, number>>({});
  const [testCsvUrl, setTestCsvUrl] = useState("");
  const [datasetDefaults, setDatasetDefaults] = useState<DatasetConfigDefaults | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const emptySelectLabel = readonly ? "Online mode: unavailable" : "No options available";

  useEffect(() => {
    if (readonly) return;
    void (async () => {
      try {
        const data = await fetchCustomModelMeta();
        setMeta(data);
        const firstPlm = Object.keys(data.plm_models)[0] || "";
        const firstDataset = Object.keys(data.dataset_configs)[0] || "";
        setPlmModel(firstPlm);
        setDatasetConfig(firstDataset);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load metadata.");
      }
    })();
  }, [readonly]);

  useEffect(() => {
    if (readonly) return;
    if (trainingMode !== "Continue Training") return;
    void (async () => {
      try {
        const folders = await fetchModelFolders("ckpt");
        setFolderOptions(folders.folders.length ? folders.folders : ["ckpt"]);
      } catch {
        setFolderOptions(["ckpt"]);
      }
    })();
  }, [trainingMode, readonly]);

  useEffect(() => {
    if (readonly) return;
    if (trainingMode !== "Continue Training" || !selectedFolder) return;
    void (async () => {
      try {
        const models = await fetchModelsInFolder(selectedFolder);
        setModelOptions(models.models);
        setSelectedModelPath(models.models[0]?.path || "");
      } catch {
        setModelOptions([]);
        setSelectedModelPath("");
      }
    })();
  }, [trainingMode, selectedFolder, readonly]);

  useEffect(() => {
    if (readonly) return;
    if (!selectedModelPath || trainingMode !== "Continue Training") return;
    void (async () => {
      try {
        const cfg = (await fetchModelConfig(selectedModelPath)).config;
        if (typeof cfg.plm_model === "string" && meta) {
          const display = Object.entries(meta.plm_models).find(([, v]) => v === cfg.plm_model)?.[0];
          if (display) setPlmModel(display);
        }
        if (typeof cfg.training_method === "string") setTrainingMethod(cfg.training_method);
        if (typeof cfg.pooling_method === "string") setPoolingMethod(cfg.pooling_method);
        if (typeof cfg.problem_type === "string") setProblemType(cfg.problem_type);
        if (typeof cfg.num_labels === "number") setNumLabels(cfg.num_labels);
      } catch {
        // ignore optional config mismatch
      }
    })();
  }, [selectedModelPath, meta, trainingMode, readonly]);

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
      setTrainFile("");
      setValidFile("");
      setTestFile("");
      setColumnOptions([]);
      setCustomDataSourceMode("hf_local");
    }
  }, [datasetSelection]);

  useEffect(() => {
    setDatasetRowsExpanded(Boolean(datasetPreview));
  }, [datasetPreview]);

  const isLora = useMemo(
    () => ["plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"].includes(trainingMethod),
    [trainingMethod]
  );
  const metricOptions = useMemo(
    () => (meta?.metrics_options && meta.metrics_options.length > 0 ? meta.metrics_options : DEFAULT_METRICS),
    [meta?.metrics_options]
  );
  const monitoredMetricOptions = useMemo(() => {
    const merged = new Set<string>([...metricOptions, ...metrics, monitoredMetrics]);
    return Array.from(merged).filter(Boolean);
  }, [metricOptions, metrics, monitoredMetrics]);
  const structureSeqOptions = useMemo(() => meta?.structure_seq_options || [], [meta?.structure_seq_options]);
  const selectableColumns = useMemo(() => {
    const merged = new Set<string>([...columnOptions, sequenceColumn, labelColumn]);
    return Array.from(merged).filter(Boolean);
  }, [columnOptions, sequenceColumn, labelColumn]);
  const schedulerOptions = useMemo(() => {
    const base = ["linear", "cosine", "step"];
    return base.includes(scheduler) ? base : [...base, scheduler].filter(Boolean);
  }, [scheduler]);
  const plmModelKeys = useMemo(() => Object.keys(meta?.plm_models || {}), [meta?.plm_models]);
  const datasetConfigKeys = useMemo(() => Object.keys(meta?.dataset_configs || {}), [meta?.dataset_configs]);
  const isSesAdapter = useMemo(() => {
    const method = String(trainingMethod || "").toLowerCase();
    return method === "ses-adapter" || method === "ses_adapter";
  }, [trainingMethod]);
  const isStructurePlm = useMemo(() => {
    const modelHint = `${plmModel} ${selectedModelPath}`.toLowerCase();
    return STRUCTURE_MODELS.some((key) => modelHint.includes(key));
  }, [plmModel, selectedModelPath]);
  const showStructureSeq = isSesAdapter;
  const showPdbDir = isSesAdapter || isStructurePlm;
  const lockByDatasetConfig = datasetSelection === "Pre-defined";
  const structureSeqRequired = useMemo(
    () => structureSeq.filter((item) => SES_STRUCTURE_COLUMNS.includes(item)),
    [structureSeq]
  );
  const knownColumns = useMemo(() => {
    const previewCols = datasetPreview?.preview?.columns || [];
    return new Set<string>([...columnOptions, ...previewCols].map((x) => String(x || "").trim()).filter(Boolean));
  }, [columnOptions, datasetPreview]);
  const trainingRuleError = useMemo(() => {
    if (isSesAdapter && problemType.toLowerCase().startsWith("residue_")) {
      return "ses-adapter cannot be used for residue-level training tasks.";
    }
    if (isStructurePlm && !pdbDir.trim()) {
      return "Structure PLM (ProSST/ProtSSN/SaProt) requires PDB Folder.";
    }
    if (isSesAdapter) {
      if (!structureSeqRequired.length) {
        return "ses-adapter requires selecting foldseek_seq and/or ss8_seq.";
      }
      if (!pdbDir.trim() && datasetSelection === "Custom" && structureSeqRequired.some((col) => !knownColumns.has(col))) {
        return "ses-adapter requires selected structure columns in dataset files, or provide PDB Folder to generate them.";
      }
    }
    return "";
  }, [isSesAdapter, problemType, isStructurePlm, pdbDir, structureSeqRequired, datasetSelection, knownColumns]);
  const effectiveDatasetCustom = useMemo(() => {
    if (datasetSelection !== "Custom" || customDataSourceMode !== "hf_local") return "";
    return datasetCustom;
  }, [datasetSelection, customDataSourceMode, datasetCustom]);
  const effectiveTrainFile = useMemo(() => {
    if (datasetSelection !== "Custom" || customDataSourceMode !== "upload") return "";
    return trainFile;
  }, [datasetSelection, customDataSourceMode, trainFile]);
  const effectiveValidFile = useMemo(() => {
    if (datasetSelection !== "Custom" || customDataSourceMode !== "upload") return "";
    return validFile;
  }, [datasetSelection, customDataSourceMode, validFile]);
  const effectiveTestFile = useMemo(() => {
    if (datasetSelection !== "Custom" || customDataSourceMode !== "upload") return "";
    return testFile;
  }, [datasetSelection, customDataSourceMode, testFile]);
  const latestEpoch = curveEpochs.length ? curveEpochs[curveEpochs.length - 1] : null;
  const latestTrainLoss = lastNumeric(trainLossSeries);
  const latestValLoss = lastNumeric(valLossSeries);
  const bestValMetric = useMemo(() => {
    const keys = Object.keys(valMetricSeries);
    if (!keys.length) return null;
    const key = keys[0];
    const latest = lastNumeric(valMetricSeries[key] || []);
    if (latest == null) return null;
    return { key, value: latest };
  }, [valMetricSeries]);
  const validationChartSeries = useMemo(() => {
    const selected = metrics.map((m) => String(m || "").trim().toLowerCase()).filter(Boolean);
    const existing = Object.keys(valMetricSeries);
    const ordered = Array.from(new Set([...selected, ...existing]));
    return ordered.map((name, idx) => ({
      name: name.toUpperCase(),
      values: valMetricSeries[name] || new Array(curveEpochs.length).fill(null),
      color: METRIC_COLORS[idx % METRIC_COLORS.length]
    }));
  }, [metrics, valMetricSeries, curveEpochs.length]);
  const stepText = useMemo(() => {
    const statusStep = statusText.match(/(?:step|iter(?:ation)?)\s*[:=]?\s*(\d+)(?:\s*\/\s*(\d+))?/i);
    if (statusStep) {
      return statusStep[2] ? `${statusStep[1]}/${statusStep[2]}` : statusStep[1];
    }
    for (let i = logs.length - 1; i >= 0 && i >= logs.length - 40; i -= 1) {
      const m = logs[i].match(/(?:step|iter(?:ation)?)\s*[:=]?\s*(\d+)(?:\s*\/\s*(\d+))?/i);
      if (m) return m[2] ? `${m[1]}/${m[2]}` : m[1];
    }
    return null;
  }, [logs, statusText]);
  const visibleError = readonly && isNotFoundLikeError(error) ? "" : error;

  function toggleMetric(metric: string) {
    setMetrics((prev) => (prev.includes(metric) ? prev.filter((x) => x !== metric) : [...prev, metric]));
  }

  function toggleStructureSeqOption(option: string) {
    setStructureSeq((prev) => (prev.includes(option) ? prev.filter((x) => x !== option) : [...prev, option]));
  }

  async function onUploadDatasetFile(file: File | null, kind: "train" | "valid" | "test") {
    if (!file) return;
    setError("");
    try {
      const result = await uploadCustomModelDatasetFile(file);
      if (kind === "train") setTrainFile(result.file_path);
      if (kind === "valid") setValidFile(result.file_path);
      if (kind === "test") setTestFile(result.file_path);
      if (Array.isArray(result.columns)) {
        setColumnOptions((prev) => {
          const merged = new Set<string>([...prev, ...result.columns]);
          return Array.from(merged).filter(Boolean);
        });
        if (result.columns.includes("aa_seq")) setSequenceColumn("aa_seq");
        if (result.columns.includes("label")) setLabelColumn("label");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "File upload failed.");
    }
  }

  async function onUseDatasetExample(kind: "train" | "valid" | "test") {
    if (readonly) return;
    const content = "aa_seq,label\nMKTAYIAKQRQISFVKSHFSRQ,1\nGAVLILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSR,0\n";
    const file = new File([content], `${kind}_example.csv`, { type: "text/csv" });
    await onUploadDatasetFile(file, kind);
  }

  function displayUploadedName(pathValue: string) {
    const normalized = String(pathValue || "").trim();
    if (!normalized) return "No file selected";
    return normalized.split("/").pop() || normalized;
  }

  function syncProgressToEpoch(epoch: number) {
    const totalEpochs = Number(numEpochs) || 0;
    if (totalEpochs <= 0) return;
    const p = Math.max(0, Math.min(0.98, (epoch + 1) / totalEpochs));
    setProgress((prev) => Math.max(prev, p));
    setStatusText(`Epoch ${epoch + 1}/${totalEpochs}`);
  }

  function applyCurveUpdateFromParsedLog(parsed: {
    epoch: number;
    trainLoss?: number;
    valLoss?: number;
    valMetricName?: string;
    valMetricValue?: number;
  }) {
    setCurveEpochs((prevEpochs) => {
      const nextEpochs = prevEpochs.includes(parsed.epoch)
        ? [...prevEpochs]
        : [...prevEpochs, parsed.epoch].sort((a, b) => a - b);
      const epochIdx = nextEpochs.indexOf(parsed.epoch);

      if (typeof parsed.trainLoss === "number" && Number.isFinite(parsed.trainLoss)) {
        setTrainLossSeries((prev) => {
          const next = remapSeriesByEpoch(nextEpochs, prevEpochs, prev);
          next[epochIdx] = parsed.trainLoss as number;
          return next;
        });
      }
      if (typeof parsed.valLoss === "number" && Number.isFinite(parsed.valLoss)) {
        setValLossSeries((prev) => {
          const next = remapSeriesByEpoch(nextEpochs, prevEpochs, prev);
          next[epochIdx] = parsed.valLoss as number;
          return next;
        });
      }
      if (parsed.valMetricName && typeof parsed.valMetricValue === "number" && Number.isFinite(parsed.valMetricValue)) {
        setValMetricSeries((prev) => {
          const next: Record<string, Array<number | null>> = {};
          for (const [key, values] of Object.entries(prev)) {
            next[key] = remapSeriesByEpoch(nextEpochs, prevEpochs, values);
          }
          if (!next[parsed.valMetricName as string]) {
            next[parsed.valMetricName as string] = new Array(nextEpochs.length).fill(null);
          }
          next[parsed.valMetricName as string][epochIdx] = parsed.valMetricValue as number;
          return next;
        });
      }
      return nextEpochs;
    });
    syncProgressToEpoch(parsed.epoch);
  }

  function reconcileCurveFromMetrics(metrics: {
    epochs: number[];
    train_loss: Array<number | null>;
    val_loss: Array<number | null>;
    val_metrics: Record<string, Array<number | null>>;
  }) {
    setCurveEpochs((prevEpochs) => {
      const nextEpochs = mergeEpochs(prevEpochs, metrics.epochs);
      const metricEpochs = metrics.epochs;
      const incomingTrain = metrics.train_loss;
      const incomingValLoss = metrics.val_loss;
      const incomingValMetrics = metrics.val_metrics;

      setTrainLossSeries((prev) => {
        const next = remapSeriesByEpoch(nextEpochs, prevEpochs, prev);
        nextEpochs.forEach((ep, idx) => {
          const incomingIdx = metricEpochs.indexOf(ep);
          const incoming = incomingIdx >= 0 ? incomingTrain[incomingIdx] : null;
          if ((next[idx] == null || !Number.isFinite(next[idx] as number)) && typeof incoming === "number" && Number.isFinite(incoming)) {
            next[idx] = incoming;
          }
        });
        return next;
      });

      setValLossSeries((prev) => {
        const next = remapSeriesByEpoch(nextEpochs, prevEpochs, prev);
        nextEpochs.forEach((ep, idx) => {
          const incomingIdx = metricEpochs.indexOf(ep);
          const incoming = incomingIdx >= 0 ? incomingValLoss[incomingIdx] : null;
          if ((next[idx] == null || !Number.isFinite(next[idx] as number)) && typeof incoming === "number" && Number.isFinite(incoming)) {
            next[idx] = incoming;
          }
        });
        return next;
      });

      setValMetricSeries((prev) => {
        const next: Record<string, Array<number | null>> = {};
        const metricKeys = new Set<string>([...Object.keys(prev), ...Object.keys(incomingValMetrics)]);
        for (const key of metricKeys) {
          const prevValues = prev[key] || [];
          const incomingValues = incomingValMetrics[key] || [];
          const remapped = remapSeriesByEpoch(nextEpochs, prevEpochs, prevValues);
          nextEpochs.forEach((ep, idx) => {
            const incomingIdx = metricEpochs.indexOf(ep);
            const incoming = incomingIdx >= 0 ? incomingValues[incomingIdx] : null;
            if ((remapped[idx] == null || !Number.isFinite(remapped[idx] as number)) && typeof incoming === "number" && Number.isFinite(incoming)) {
              remapped[idx] = incoming;
            }
          });
          next[key] = remapped;
        }
        return next;
      });
      return nextEpochs;
    });

    const latestEpoch = metrics.epochs.length ? Number(metrics.epochs[metrics.epochs.length - 1]) : NaN;
    if (Number.isFinite(latestEpoch)) {
      syncProgressToEpoch(latestEpoch);
    }
  }

  const trainingArgs = useMemo(
    () => ({
      plm_model: plmModel,
      training_mode: trainingMode,
      transfer_model_dropdown: selectedModelPath,
      dataset_selection: datasetSelection,
      dataset_config: datasetConfig,
      dataset_custom: effectiveDatasetCustom,
      train_file: effectiveTrainFile,
      valid_file: effectiveValidFile,
      test_file: effectiveTestFile,
      problem_type: problemType,
      num_labels: numLabels,
      metrics,
      training_method: trainingMethod,
      pooling_method: poolingMethod,
      sequence_column_name: sequenceColumn,
      label_column_name: labelColumn,
      batch_mode: batchMode,
      batch_size: batchSize,
      batch_token: batchToken,
      learning_rate: learningRate,
      num_epochs: numEpochs,
      max_seq_len: maxSeqLen,
      gradient_accumulation_steps: gradAccumulation,
      warmup_steps: warmupSteps,
      scheduler,
      output_model_name: outputModelName,
      output_dir: outputDir,
      wandb_enabled: wandbEnabled,
      wandb_project: wandbProject,
      wandb_entity: wandbEntity,
      patience,
      num_workers: numWorkers,
      max_grad_norm: maxGradNorm,
      structure_seq: structureSeq,
      pdb_dir: pdbDir,
      lora_r: loraR,
      lora_alpha: loraAlpha,
      lora_dropout: loraDropout,
      lora_target_modules: loraTargetModules,
      monitored_metrics: monitoredMetrics,
      monitored_strategy: monitoredStrategy
    }),
    [
      plmModel,
      trainingMode,
      selectedModelPath,
      datasetSelection,
      customDataSourceMode,
      datasetConfig,
      datasetCustom,
      trainFile,
      validFile,
      testFile,
      problemType,
      numLabels,
      metrics,
      trainingMethod,
      poolingMethod,
      sequenceColumn,
      labelColumn,
      batchMode,
      batchSize,
      batchToken,
      learningRate,
      numEpochs,
      maxSeqLen,
      gradAccumulation,
      warmupSteps,
      scheduler,
      outputModelName,
      outputDir,
      wandbEnabled,
      wandbProject,
      wandbEntity,
      patience,
      numWorkers,
      maxGradNorm,
      structureSeq,
      pdbDir,
      loraR,
      loraAlpha,
      loraDropout,
      loraTargetModules,
      monitoredMetrics,
      monitoredStrategy
    ]
  );

  async function onPreviewDataset() {
    setError("");
    setPreviewLoading(true);
    try {
      const result = await previewDataset({
        dataset_selection: datasetSelection,
        dataset_config: datasetConfig,
        dataset_custom: effectiveDatasetCustom,
        train_file: effectiveTrainFile,
        valid_file: effectiveValidFile,
        test_file: effectiveTestFile
      });
      setDatasetPreview(result);
      if (Array.isArray(result.column_options)) {
        setColumnOptions(result.column_options.filter(Boolean));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dataset preview failed.");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function onPreviewCommand() {
    setError("");
    if (trainingRuleError) {
      setError(trainingRuleError);
      return;
    }
    try {
      const data = await previewTraining(trainingArgs);
      setCommandPreview(data.command);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Preview command failed.");
    }
  }

  async function onStart() {
    setError("");
    if (trainingRuleError) {
      setError(trainingRuleError);
      return;
    }
    setLogs([]);
    setProgress(0);
    setStatusText("Starting...");
    setCurveEpochs([]);
    setTrainLossSeries([]);
    setValLossSeries([]);
    setValMetricSeries({});
    setTestMetrics({});
    setTestCsvUrl("");
    setRunning(true);
    try {
      await startTrainingStream(trainingArgs, (evt) => {
        if (evt.type === "start") {
          setCommandPreview(evt.data.command || "");
          setStatusText("Running");
          return;
        }
        if (evt.type === "progress") {
          const nextProgress = Number.isFinite(evt.data.progress) ? evt.data.progress : 0;
          const nextMessage = evt.data.message || "Running";
          if (!nextMessage.startsWith("Epoch ")) return;
          setProgress((prev) => Math.max(prev, nextProgress));
          setStatusText(nextMessage);
          return;
        }
        if (evt.type === "log") {
          if (evt.data.line) {
            setLogs((prev) => [...prev, evt.data.line]);
            const parsed = parseEpochMetricLogLine(evt.data.line);
            if (parsed) {
              applyCurveUpdateFromParsedLog(parsed);
            }
          }
          return;
        }
        if (evt.type === "training_metrics") {
          if (!isTrainingMetricsPayload(evt.data)) return;
          reconcileCurveFromMetrics(evt.data);
          return;
        }
        if (evt.type === "test_results") {
          setTestMetrics(evt.data.metrics || {});
          if (evt.data.csv_download_url) setTestCsvUrl(evt.data.csv_download_url);
          return;
        }
        if (evt.type === "error") {
          setError(evt.data.message || "Training stream failed.");
          return;
        }
        if (evt.type === "done") {
          if (isTrainingMetricsPayload(evt.data.training_metrics)) {
            setCurveEpochs(evt.data.training_metrics.epochs);
            setTrainLossSeries(evt.data.training_metrics.train_loss);
            setValLossSeries(evt.data.training_metrics.val_loss);
            setValMetricSeries(evt.data.training_metrics.val_metrics);
          }
          setStatusText(evt.data.message || (evt.data.success ? "Completed" : "Failed"));
          setProgress((prev) => (evt.data.success ? 1 : prev));
          if (evt.data.csv_download_url) setTestCsvUrl(evt.data.csv_download_url);
          if (evt.data.test_metrics) setTestMetrics(evt.data.test_metrics);
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed.");
      setStatusText("Failed");
    } finally {
      setRunning(false);
    }
  }

  async function onAbort() {
    await abortTraining();
    setStatusText("Aborted");
    setRunning(false);
  }

  return (
    <div className={`custom-model-page ${readonly ? "readonly-mode" : ""}`}>
      <header className="chat-header">
        <div>
          <h2>Custom Model Training</h2>
          <p>Train your own protein model with configurable datasets and hyperparameters.</p>
        </div>
        <div className={`run-status-bar ${running ? "running" : "stopped"}`}>
          <span className="run-status-dot" />
          <span className="run-status-text">{statusText}</span>
        </div>
      </header>
      {readonly && (
        <div className="readonly-banner" role="status" aria-live="polite">
          Online mode: custom model controls are view-only in this deployment.
        </div>
      )}

      <section className="custom-model-grid">
        <aside className="chat-panel left custom-model-controls">
          <fieldset className="readonly-fieldset" disabled={readonly}>
          <div className="custom-top-line-grid">
            <section
              className={`custom-section-card custom-section-line ${
                trainingMode === "Continue Training" ? "custom-section-line-continue" : "custom-section-line-basic"
              } custom-section-line-top`}
            >
              <label className="left-controls custom-line-field">
                Training Mode
                <SegmentedSwitch
                  value={trainingMode}
                  onChange={setTrainingMode}
                  ariaLabel="Training mode switch"
                  className="custom-segment-switch-wide"
                  options={[
                    { value: "From Scratch", label: "From Scratch" },
                    { value: "Continue Training", label: "Continue Training" }
                  ]}
                />
              </label>

              {trainingMode === "Continue Training" && (
                <>
                  <label className="left-controls custom-line-field">Model Folder
                    <select value={selectedFolder} onChange={(e) => setSelectedFolder(e.target.value)}>
                      {folderOptions.length === 0 && <option value="">{emptySelectLabel}</option>}
                      {folderOptions.map((f) => <option key={f} value={f}>{f}</option>)}
                    </select>
                  </label>
                  <label className="left-controls custom-line-field">Checkpoint
                    <select value={selectedModelPath} onChange={(e) => setSelectedModelPath(e.target.value)}>
                      <option value="">Select model</option>
                      {modelOptions.map((m) => <option key={m.path} value={m.path}>{m.label}</option>)}
                    </select>
                  </label>
                </>
              )}

              {trainingMode === "From Scratch" && (
                <>
                  <label className="left-controls custom-line-field">PLM Model
                    <select value={plmModel} onChange={(e) => setPlmModel(e.target.value)}>
                      {plmModelKeys.length === 0 && <option value="">{emptySelectLabel}</option>}
                      {plmModelKeys.map((k) => <option key={k} value={k}>{k}</option>)}
                    </select>
                  </label>
                  <label className="left-controls custom-line-field">Output Dir
                    <input value={outputDir} onChange={(e) => setOutputDir(e.target.value)} />
                  </label>
                  <label className="left-controls custom-line-field">Output Name
                    <input value={outputModelName} onChange={(e) => setOutputModelName(e.target.value)} />
                  </label>
                </>
              )}
            </section>

            <section className="custom-section-card custom-section-line custom-section-line-dataset custom-section-line-top">
              <div className="custom-dataset-line-main">
                <label className="left-controls custom-line-field custom-dataset-main-dataset">
                  Dataset
                  <SegmentedSwitch
                    value={datasetSelection}
                    onChange={setDatasetSelection}
                    ariaLabel="Dataset mode switch"
                    className="custom-segment-switch-wide"
                    options={[
                      { value: "Custom", label: "Custom" },
                      { value: "Pre-defined", label: "default" }
                    ]}
                  />
                </label>
                <div className="custom-dataset-line-inputs custom-dataset-main-inputs">
                  {datasetSelection === "Pre-defined" ? (
                    <label className="left-controls custom-line-field">
                      Dataset Path
                      <select value={datasetConfig} onChange={(e) => setDatasetConfig(e.target.value)}>
                        {datasetConfigKeys.length === 0 && <option value="">{emptySelectLabel}</option>}
                        {datasetConfigKeys.map((k) => <option key={k} value={k}>{k}</option>)}
                      </select>
                    </label>
                  ) : (
                    <>
                      <label className="left-controls custom-line-field">
                        Source
                        <SegmentedSwitch
                          value={customDataSourceMode}
                          onChange={(value) => setCustomDataSourceMode(value as "hf_local" | "upload")}
                          ariaLabel="Custom dataset source mode"
                          className="custom-segment-switch-wide"
                          options={[
                            { value: "hf_local", label: "HF Path" },
                            { value: "upload", label: "Upload" }
                          ]}
                        />
                      </label>
                      {customDataSourceMode === "hf_local" ? (
                        <label className="left-controls custom-line-field">
                          Train File / HF id
                          <input
                            value={datasetCustom}
                            onChange={(e) => setDatasetCustom(e.target.value)}
                            placeholder="hf dataset id or local dataset path"
                          />
                        </label>
                      ) : (
                        <div className="custom-upload-dropzone-wrap">
                          <div className="custom-upload-dropzone-grid">
                            <div className="custom-upload-item upload-source-stack">
                              <span className="custom-upload-item-label">Train</span>
                              <label className="custom-upload-trigger">
                                <input
                                  type="file"
                                  accept=".csv,.tsv,.xlsx,.xls"
                                  onChange={(e) => void onUploadDatasetFile(e.target.files?.[0] || null, "train")}
                                />
                                Choose File
                              </label>
                              <WorkspaceFilePicker
                                workspaceEnabled={workspaceEnabled}
                                disabled={readonly || running}
                                acceptedCategories={["table_or_text"]}
                                buttonLabel="From Workspace"
                                onPick={(picked) => {
                                  const selected = picked[0];
                                  if (!selected) return;
                                  setTrainFile(selected.storage_path);
                                }}
                              />
                              <button
                                type="button"
                                className="custom-btn-secondary"
                                onClick={() => void onUseDatasetExample("train")}
                                disabled={readonly || running}
                              >
                                Use Example
                              </button>
                              <span className="custom-upload-file-chip custom-upload-file-name" title={displayUploadedName(trainFile)}>
                                {displayUploadedName(trainFile)}
                              </span>
                            </div>
                            <div className="custom-upload-item upload-source-stack">
                              <span className="custom-upload-item-label">Valid</span>
                              <label className="custom-upload-trigger">
                                <input
                                  type="file"
                                  accept=".csv,.tsv,.xlsx,.xls"
                                  onChange={(e) => void onUploadDatasetFile(e.target.files?.[0] || null, "valid")}
                                />
                                Choose File
                              </label>
                              <WorkspaceFilePicker
                                workspaceEnabled={workspaceEnabled}
                                disabled={readonly || running}
                                acceptedCategories={["table_or_text"]}
                                buttonLabel="From Workspace"
                                onPick={(picked) => {
                                  const selected = picked[0];
                                  if (!selected) return;
                                  setValidFile(selected.storage_path);
                                }}
                              />
                              <button
                                type="button"
                                className="custom-btn-secondary"
                                onClick={() => void onUseDatasetExample("valid")}
                                disabled={readonly || running}
                              >
                                Use Example
                              </button>
                              <span className="custom-upload-file-chip custom-upload-file-name" title={displayUploadedName(validFile)}>
                                {displayUploadedName(validFile)}
                              </span>
                            </div>
                            <div className="custom-upload-item upload-source-stack">
                              <span className="custom-upload-item-label">Test</span>
                              <label className="custom-upload-trigger">
                                <input
                                  type="file"
                                  accept=".csv,.tsv,.xlsx,.xls"
                                  onChange={(e) => void onUploadDatasetFile(e.target.files?.[0] || null, "test")}
                                />
                                Choose File
                              </label>
                              <WorkspaceFilePicker
                                workspaceEnabled={workspaceEnabled}
                                disabled={readonly || running}
                                acceptedCategories={["table_or_text"]}
                                buttonLabel="From Workspace"
                                onPick={(picked) => {
                                  const selected = picked[0];
                                  if (!selected) return;
                                  setTestFile(selected.storage_path);
                                }}
                              />
                              <button
                                type="button"
                                className="custom-btn-secondary"
                                onClick={() => void onUseDatasetExample("test")}
                                disabled={readonly || running}
                              >
                                Use Example
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
                    {previewLoading ? "Loading..." : "Preview Dataset"}
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
              <h3 className="custom-panel-title custom-dataset-preview-title">Dataset Preview</h3>
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
                      {datasetRowsExpanded ? "▾" : "▸"} Sample Rows ({datasetPreview.preview.rows.length})
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
                  <p>No sample rows available.</p>
                )}
              </>
            ) : (
              <p>No dataset preview yet.</p>
            )}
          </section>
          <section className="custom-section-card custom-section-wide">
            <h3 className="custom-panel-title">Hyper Parameters</h3>
            <div className="custom-advanced-grid custom-advanced-panel">
            {lockByDatasetConfig && (
              <div className="custom-readonly-note custom-field-span-full">
                Default dataset presets are loaded from dataset JSON and locked.
              </div>
            )}
            <div className="custom-param-group-title custom-field-span-full">Task Settings</div>
            <label className="left-controls custom-field-short">Training Method
              <select value={trainingMethod} onChange={(e) => setTrainingMethod(e.target.value)}>
                {(meta?.training_methods || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                {(meta?.training_methods || []).map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-short">Pooling
              <select value={poolingMethod} onChange={(e) => setPoolingMethod(e.target.value)}>
                {(meta?.pooling_methods || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                {(meta?.pooling_methods || []).map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            {(showStructureSeq || showPdbDir) && (
              <>
                {showStructureSeq && (
                  <div className="left-controls custom-field-span-2">
                    <span>Structure Seq</span>
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
                )}
                {showPdbDir && (
                  <label className="left-controls custom-field-short">PDB Folder<input value={pdbDir} onChange={(e) => setPdbDir(e.target.value)} disabled={lockByDatasetConfig && Boolean(datasetDefaults?.pdb_dir)} /></label>
                )}
              </>
            )}
            {isStructurePlm && (
              <div className="custom-readonly-note custom-field-span-full">
                Selected structure model requires PDB Folder.
              </div>
            )}
            {trainingRuleError && <div className="custom-readonly-note custom-field-span-full">{trainingRuleError}</div>}
            <div className="custom-field-span-full" />
            <label className="left-controls custom-field-short">Problem Type
              <select value={problemType} onChange={(e) => setProblemType(e.target.value)} disabled={lockByDatasetConfig}>
                {(meta?.problem_types || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                {(meta?.problem_types || []).map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-short">Num Labels<input type="number" value={numLabels} onChange={(e) => setNumLabels(Number(e.target.value) || 1)} disabled={lockByDatasetConfig} /></label>
            <label className="left-controls custom-field-short">Label Column
              {selectableColumns.length > 0 && !lockByDatasetConfig ? (
                <select value={labelColumn} onChange={(e) => setLabelColumn(e.target.value)}>
                  {selectableColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              ) : (
                <input value={labelColumn} onChange={(e) => setLabelColumn(e.target.value)} disabled={lockByDatasetConfig} />
              )}
            </label>
            <label className="left-controls custom-field-short">Sequence Column
              {selectableColumns.length > 0 && !lockByDatasetConfig ? (
                <select value={sequenceColumn} onChange={(e) => setSequenceColumn(e.target.value)}>
                  {selectableColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              ) : (
                <input value={sequenceColumn} onChange={(e) => setSequenceColumn(e.target.value)} disabled={lockByDatasetConfig} />
              )}
            </label>
            <div className="left-controls custom-field-span-full">
              <span>Metrics</span>
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
            <div className="custom-param-group-title custom-field-span-full">Optimization</div>
            <label className="left-controls custom-field-short">Batch Mode
              <select value={batchMode} onChange={(e) => setBatchMode(e.target.value as "Batch Size Mode" | "Batch Token Mode")}>
                <option value="Batch Size Mode">Batch Size Mode</option>
                <option value="Batch Token Mode">Batch Token Mode</option>
              </select>
            </label>
            {batchMode === "Batch Size Mode" ? (
              <label className="left-controls custom-field-short">Batch Size<input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value) || 1)} /></label>
            ) : (
              <label className="left-controls custom-field-short">Batch Token<input type="number" value={batchToken} onChange={(e) => setBatchToken(Number(e.target.value) || 1000)} /></label>
            )}
            <label className="left-controls custom-field-short">Scheduler
              <select value={scheduler} onChange={(e) => setScheduler(e.target.value)}>
                {schedulerOptions.map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
            </label>
            <label className="left-controls custom-field-short">Learning Rate<input type="number" step="0.0001" value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value) || 0)} /></label>
            <label className="left-controls custom-field-short">Epochs<input type="number" value={numEpochs} onChange={(e) => setNumEpochs(Number(e.target.value) || 1)} /></label>
            <label className="left-controls custom-field-short">Max Seq Len<input type="number" value={maxSeqLen} onChange={(e) => setMaxSeqLen(Number(e.target.value) || 1)} /></label>
            <label className="left-controls custom-field-short">Grad Accum<input type="number" value={gradAccumulation} onChange={(e) => setGradAccumulation(Number(e.target.value) || 1)} /></label>
            <label className="left-controls custom-field-short">Max Grad Norm<input type="number" value={maxGradNorm} onChange={(e) => setMaxGradNorm(Number(e.target.value))} /></label>
            <label className="left-controls custom-field-short">Num Workers<input type="number" value={numWorkers} onChange={(e) => setNumWorkers(Number(e.target.value) || 0)} /></label>
            <label className="left-controls custom-field-short">Warmup Steps<input type="number" value={warmupSteps} onChange={(e) => setWarmupSteps(Number(e.target.value) || 0)} /></label>
            <div className="custom-param-group-title custom-field-span-full">Monitoring</div>
            <label className="left-controls custom-field-short">Monitored Metric
              <select value={monitoredMetrics} onChange={(e) => setMonitoredMetrics(e.target.value)}>
                {monitoredMetricOptions.map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
            </label>
            <label className="left-controls custom-field-short">Monitored Strategy
              <select value={monitoredStrategy} onChange={(e) => setMonitoredStrategy(e.target.value)}>
                <option value="max">max</option>
                <option value="min">min</option>
              </select>
            </label>
            <label className="left-controls custom-field-short">Patience<input type="number" value={patience} onChange={(e) => setPatience(Number(e.target.value) || 1)} /></label>
            {isLora && (
              <>
                <label className="left-controls custom-field-short">LoRA r<input type="number" value={loraR} onChange={(e) => setLoraR(Number(e.target.value) || 1)} /></label>
                <label className="left-controls custom-field-short">LoRA alpha<input type="number" value={loraAlpha} onChange={(e) => setLoraAlpha(Number(e.target.value) || 1)} /></label>
                <label className="left-controls custom-field-short">LoRA dropout<input type="number" step="0.01" value={loraDropout} onChange={(e) => setLoraDropout(Number(e.target.value) || 0)} /></label>
                <label className="left-controls custom-field-span-full">LoRA target modules<input value={loraTargetModules} onChange={(e) => setLoraTargetModules(e.target.value)} /></label>
              </>
            )}

            <div className="custom-field-span-full custom-wandb-row">
              <label className="left-controls custom-inline-check custom-check-toggle custom-wandb-toggle">
                <input type="checkbox" checked={wandbEnabled} onChange={(e) => setWandbEnabled(e.target.checked)} />
                <span className="custom-check-slider" aria-hidden="true" />
                <span className="custom-check-label">Enable wandb</span>
              </label>
              {wandbEnabled && (
                <>
                  <label className="left-controls custom-wandb-field">wandb Project<input value={wandbProject} onChange={(e) => setWandbProject(e.target.value)} /></label>
                  <label className="left-controls custom-wandb-field">wandb Entity<input value={wandbEntity} onChange={(e) => setWandbEntity(e.target.value)} /></label>
                </>
              )}
            </div>
            </div>

            <div className="custom-actions-inline-status">
              <div className="custom-row custom-actions">
                <button type="button" className="custom-btn-secondary" onClick={() => void onPreviewCommand()}>Preview Command</button>
                <button type="button" className="custom-btn-primary" disabled={running} onClick={() => void onStart()}>Start Training</button>
                <button type="button" className="custom-btn-danger" disabled={!running} onClick={() => void onAbort()}>Abort</button>
              </div>
            </div>
          </section>
          </fieldset>
          </aside>

        <div className="chat-panel right custom-model-output custom-bottom-right">
          <h3 className="custom-output-panel-title custom-panel-title">Output Panel</h3>
          {visibleError && <div className="error-box">{visibleError}</div>}
          <div className="custom-inline-status-panel custom-inline-status-panel-compact">
            <details className="custom-preview-collapse custom-command-preview-collapse">
              <summary>Command Preview</summary>
              <pre className="custom-command custom-command-plain">
                <code>{commandPreview || "Click Preview Command to generate CLI command."}</code>
              </pre>
            </details>
            <div className="custom-inline-status-title">Progress</div>
            <div className="custom-progress-wrap">
              <div className="custom-progress-meta">
                <span>{statusText}</span>
                <span>{Math.round(progress * 100)}%</span>
              </div>
              <div className="custom-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
                <div className="custom-progress-fill" style={{ width: `${Math.max(0, Math.min(100, progress * 100))}%` }} />
              </div>
              <div className="custom-progress-brief">
                <span>Epoch: {latestEpoch != null ? `${latestEpoch}/${numEpochs}` : `-/${numEpochs}`}</span>
                <span>Step: {stepText || "-"}</span>
                <span>Train Loss: {formatMetricValue(latestTrainLoss)}</span>
                <span>Val Loss: {formatMetricValue(latestValLoss)}</span>
                <span>
                  {bestValMetric ? `${bestValMetric.key.toUpperCase()}: ${formatMetricValue(bestValMetric.value)}` : "Val Metric: -"}
                </span>
              </div>
            </div>
          </div>
          <h3>Training Curves</h3>
          <div className="custom-chart-grid">
            <SimpleLineChart
              title="Loss Curve"
              epochs={curveEpochs}
              series={[
                { name: "Train Loss", values: trainLossSeries, color: "#8a6b45" },
                { name: "Val Loss", values: valLossSeries, color: "#c17d63" }
              ]}
            />
            <SimpleLineChart
              title="Validation Metrics"
              epochs={curveEpochs}
              series={validationChartSeries}
            />
          </div>
          <h3>Test Results</h3>
          <div className="custom-preview-block">
            {Object.keys(testMetrics).length > 0 ? (
              <div className="custom-table-wrap">
                <table className="custom-preview-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(testMetrics).map(([k, v]) => (
                      <tr key={k}>
                        <td>{k.toUpperCase()}</td>
                        <td>{Number(v).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p>No test metrics yet.</p>
            )}
            {testCsvUrl && (
              <div className="custom-row custom-actions">
                <a className="custom-btn-secondary custom-link-btn" href={testCsvUrl}>
                  Download CSV
                </a>
              </div>
            )}
          </div>
          <div className="custom-log-box">
            {logs.length ? logs.map((line, idx) => <div key={`${idx}-${line.slice(0, 20)}`}>{line}</div>) : "No logs yet."}
          </div>
        </div>
        </section>
      </section>
      <PageFooter />
    </div>
  );
}
