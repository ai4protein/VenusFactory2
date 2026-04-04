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

const DEFAULT_METRICS = ["accuracy", "mcc", "f1", "precision", "recall", "auroc"];
const STRUCTURE_MODELS = ["protssn", "prosst", "saprot"];
const SES_STRUCTURE_COLUMNS = ["foldseek_seq", "ss8_seq"];

function isNotFoundLikeError(message: string): boolean {
  const text = String(message || "").toLowerCase();
  return text.includes("404") || text.includes("not found") || text.includes('{"detail":"not found"}');
}

type CustomModelEvaluationPageProps = {
  readonly?: boolean;
  workspaceEnabled?: boolean;
};

export function CustomModelEvaluationPage({ readonly = false, workspaceEnabled = false }: CustomModelEvaluationPageProps) {
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
  const [statusText, setStatusText] = useState("Idle");
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
  const emptySelectLabel = readonly ? "Online mode: unavailable" : "No options available";
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
      return "Structure PLM (ProSST/ProtSSN/SaProt) requires PDB Folder.";
    }
    if (isSesAdapter) {
      if (!structureSeqRequired.length) {
        return "ses-adapter requires selecting foldseek_seq and/or ss8_seq.";
      }
      if (!pdbDir.trim() && datasetSelection === "Custom" && structureSeqRequired.some((col) => !knownColumns.has(col))) {
        return "ses-adapter requires selected structure columns in test file, or provide PDB Folder.";
      }
    }
    return "";
  }, [isStructurePlm, pdbDir, isSesAdapter, structureSeqRequired, datasetSelection, knownColumns]);
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
        setError(err instanceof Error ? err.message : "Failed to load metadata.");
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
          notes.push("PLM missing in ckpt config, fallback to current/default value.");
        }

        if (typeof cfg.training_method === "string") {
          setEvalMethod(cfg.training_method);
        } else {
          notes.push("training_method missing in ckpt config, fallback to default eval method.");
        }

        if (typeof cfg.pooling_method === "string") {
          setPoolingMethod(cfg.pooling_method);
        } else {
          notes.push("pooling_method missing in ckpt config, fallback to default pooling.");
        }

        if (allowCkptDatasetOverwrite && typeof cfg.problem_type === "string") {
          setProblemType(cfg.problem_type);
        } else if (allowCkptDatasetOverwrite) {
          notes.push("problem_type missing in ckpt config, fallback to default problem type.");
        }

        if (allowCkptDatasetOverwrite && typeof cfg.num_labels === "number") {
          setNumLabels(cfg.num_labels);
        } else if (allowCkptDatasetOverwrite) {
          notes.push("num_labels missing in ckpt config, fallback to default label count.");
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
          notes.push("metrics missing in ckpt config, fallback to current/default metrics.");
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
            ? `Values auto-filled from selected checkpoint. ${notes.join(" ")}`
            : ""
        );
      } catch {
        setCkptLocked(false);
        setCkptConfigNote("Checkpoint config not found. Parameters remain editable with current/default values.");
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
      setError(err instanceof Error ? err.message : "File upload failed.");
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
    if (!normalized) return "No file selected";
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
      setError(err instanceof Error ? err.message : "Dataset preview failed.");
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
      setError(err instanceof Error ? err.message : "Preview failed.");
    }
  }

  async function onStart() {
    if (evaluationRuleError) {
      setError(evaluationRuleError);
      return;
    }
    setRunning(true);
    setStatusText("Starting...");
    setProgress(0);
    setLogs([]);
    try {
      await startEvaluationStream(args, (evt) => {
        if (evt.type === "start") {
          setCommandPreview(evt.data.command || "");
          setStatusText("Running");
          return;
        }
        if (evt.type === "progress") {
          const nextProgress = Number.isFinite(evt.data.progress) ? evt.data.progress : 0;
          const nextMessage = evt.data.message || "Running";
          setProgress((prev) => Math.max(prev, nextProgress));
          setStatusText((prev) => {
            if (nextMessage.startsWith("Epoch ")) return nextMessage;
            if (prev.startsWith("Epoch ") && nextProgress < 0.999) return prev;
            return nextMessage;
          });
        }
        if (evt.type === "log" && evt.data.line) setLogs((prev) => [...prev, evt.data.line]);
        if (evt.type === "error") setError(evt.data.message || "Evaluation failed.");
        if (evt.type === "done") {
          if (typeof evt.data.final_progress === "number") {
            setProgress((prev) => Math.max(prev, evt.data.final_progress));
          }
          setStatusText(evt.data.message || (evt.data.success ? "Completed" : "Failed"));
          setProgress((prev) => (evt.data.success ? 1 : prev));
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation start failed.");
      setStatusText("Failed");
    } finally {
      setRunning(false);
    }
  }

  async function onAbort() {
    await abortEvaluation();
    setStatusText("Aborted");
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
          <h2>Custom Model Evaluation</h2>
          <p>Evaluate trained models with configurable metrics and transparent commands.</p>
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
            <section className="custom-section-card custom-section-line custom-section-line-eval-model custom-section-line-top">
              <label className="left-controls custom-line-field">Molde Folder
                <select value={selectedFolder} onChange={(e) => setSelectedFolder(e.target.value)}>
                  {folderOptions.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              </label>
              <label className="left-controls custom-line-field">Model Path
                <select value={modelPath} onChange={(e) => setModelPath(e.target.value)}>
                  <option value="">Select model</option>
                  {modelOptions.map((m) => <option key={m.path} value={m.path}>{m.label}</option>)}
                </select>
              </label>
            </section>

            <section className="custom-section-card custom-section-line custom-section-line-dataset custom-section-line-top">
              <div className="custom-dataset-line-main">
                <label className="left-controls custom-line-field custom-dataset-main-dataset">Dataset
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
                    <label className="left-controls custom-line-field">Dataset Path
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
                        <label className="left-controls custom-line-field">Test File / HF id
                          <input
                            value={datasetCustom}
                            onChange={(e) => setDatasetCustom(e.target.value)}
                            placeholder="hf dataset id or local dataset path"
                          />
                        </label>
                      ) : (
                        <div className="custom-upload-dropzone-wrap">
                          <div className="custom-upload-dropzone-grid custom-upload-dropzone-grid-single">
                            <div className="custom-upload-item upload-source-stack">
                              <span className="custom-upload-item-label">Test</span>
                              <label className="custom-upload-trigger">
                                <input
                                  type="file"
                                  accept=".csv,.tsv,.xlsx,.xls"
                                  onChange={(e) => void onUploadTestFile(e.target.files?.[0] || null)}
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
                                onClick={() => void onUseTestExample()}
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
                <p>No preview yet.</p>
              )}
            </section>

            <section className="custom-section-card custom-section-wide">
              <h3 className="custom-panel-title">Evaluation Params</h3>
              <div className="custom-advanced-grid custom-advanced-panel">
                {ckptConfigNote && <div className="custom-readonly-note custom-field-span-full">{ckptConfigNote}</div>}
                {lockByDatasetConfig && (
                  <div className="custom-readonly-note custom-field-span-full">
                    Default dataset presets are loaded from dataset JSON and locked.
                  </div>
                )}
                <div className="custom-param-group-title custom-field-span-full">Task Settings</div>
                <label className="left-controls custom-field-short">PLM
                  <select value={plmModel} onChange={(e) => setPlmModel(e.target.value)} disabled={ckptLocked}>
                    {plmModelKeys.length === 0 && <option value="">{emptySelectLabel}</option>}
                    {plmModelKeys.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                </label>
                <label className="left-controls custom-field-short">Eval Method
                  <select value={evalMethod} onChange={(e) => setEvalMethod(e.target.value)} disabled={ckptLocked}>
                    {(meta?.training_methods || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                    {(meta?.training_methods || []).map((x) => <option key={x} value={x}>{x}</option>)}
                  </select>
                </label>
                <label className="left-controls custom-field-short">Pooling
                  <select value={poolingMethod} onChange={(e) => setPoolingMethod(e.target.value)} disabled={ckptLocked}>
                    {(meta?.pooling_methods || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                    {(meta?.pooling_methods || []).map((x) => <option key={x} value={x}>{x}</option>)}
                  </select>
                </label>
                {showStructureInputs && (
                  <>
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
                    <label className="left-controls custom-field-short">PDB Folder<input value={pdbDir} onChange={(e) => setPdbDir(e.target.value)} disabled={lockByDatasetConfig && Boolean(datasetDefaults?.pdb_dir)} /></label>
                  </>
                )}
                {isStructurePlm && (
                  <div className="custom-readonly-note custom-field-span-full">
                    Selected structure model requires PDB Folder.
                  </div>
                )}
                {evaluationRuleError && <div className="custom-readonly-note custom-field-span-full">{evaluationRuleError}</div>}
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
              </div>

              <div className="custom-actions-inline-status">
                <div className="custom-row custom-actions">
                  <button type="button" className="custom-btn-secondary" onClick={() => void onPreviewCommand()}>Preview Command</button>
                  <button type="button" className="custom-btn-primary" disabled={running} onClick={() => void onStart()}>Start Evaluation</button>
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
              </div>
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
