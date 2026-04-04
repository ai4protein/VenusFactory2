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

const STRUCTURE_MODELS = ["protssn", "prosst", "saprot"];
const SES_STRUCTURE_COLUMNS = ["foldseek_seq", "ss8_seq"];

function isNotFoundLikeError(message: string): boolean {
  const text = String(message || "").toLowerCase();
  return text.includes("404") || text.includes("not found") || text.includes('{"detail":"not found"}');
}

type CustomModelPredictPageProps = {
  readonly?: boolean;
  workspaceEnabled?: boolean;
};

export function CustomModelPredictPage({ readonly = false, workspaceEnabled = false }: CustomModelPredictPageProps) {
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
  const [statusText, setStatusText] = useState("Idle");
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [commandPreview, setCommandPreview] = useState("");
  const [error, setError] = useState("");
  const [ckptLocked, setCkptLocked] = useState(false);
  const [ckptConfigNote, setCkptConfigNote] = useState("");
  const emptySelectLabel = readonly ? "Online mode: unavailable" : "No options available";

  useEffect(() => {
    if (readonly) return;
    void (async () => {
      try {
        const data = await fetchCustomModelMeta();
        setMeta(data);
        setPlmModel(Object.keys(data.plm_models)[0] || "");
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
        const notes: string[] = [];
        const display = Object.entries(meta.plm_models).find(([, v]) => v === cfg.plm_model)?.[0];
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

        if (typeof cfg.problem_type === "string") {
          setProblemType(cfg.problem_type);
        } else {
          notes.push("problem_type missing in ckpt config, fallback to default problem type.");
        }

        if (typeof cfg.num_labels === "number") {
          setNumLabels(cfg.num_labels);
        } else {
          notes.push("num_labels missing in ckpt config, fallback to default label count.");
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
            ? `Values auto-filled from selected model. ${notes.join(" ")}`
            : "Values auto-filled from selected model and locked for consistency."
        );
      } catch {
        setCkptLocked(false);
        setCkptConfigNote("Model config not found. Parameters remain editable with current/default values.");
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
      return "Structure PLM (ProSST/ProtSSN/SaProt) requires PDB Folder.";
    }
    if (predictionMode === "batch") {
      if (batchInputSource === "paste" && !batchFastaText.trim()) {
        return "Batch FASTA text cannot be empty.";
      }
      if (batchInputSource !== "paste" && !inputFile.trim()) {
        return "Batch prediction requires an input file.";
      }
    }
    if (isSesAdapter) {
      if (!structureSeqRequired.length) {
        return "ses-adapter requires selecting foldseek_seq and/or ss8_seq.";
      }
      if (predictionMode === "single" && !pdbDir.trim()) {
        if (structureSeqRequired.includes("foldseek_seq") && !foldseekSeq.trim()) {
          return "Single predict with ses-adapter requires Foldseek Sequence or PDB Folder.";
        }
        if (structureSeqRequired.includes("ss8_seq") && !ss8Seq.trim()) {
          return "Single predict with ses-adapter requires SS8 Sequence or PDB Folder.";
        }
      }
      if (predictionMode === "batch" && !pdbDir.trim() && structureSeqRequired.some((col) => !knownBatchColumns.has(col))) {
        return "Batch ses-adapter requires selected structure columns in input file, or provide PDB Folder.";
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
    knownBatchColumns
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
      setError(err instanceof Error ? err.message : "Preview failed.");
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
    setStatusText("Starting...");
    try {
      const resolvedInputFile = await resolveBatchInputFile();
      await startPredictStream({ ...predictArgs, input_file: resolvedInputFile }, (evt) => {
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
        if (evt.type === "error") setError(evt.data.message || "Predict failed.");
        if (evt.type === "done") {
          const finalProgress = evt.data.final_progress;
          if (typeof finalProgress === "number") {
            setProgress((prev) => Math.max(prev, finalProgress));
          }
          setStatusText(evt.data.message || (evt.data.success ? "Completed" : "Failed"));
          setProgress((prev) => (evt.data.success ? 1 : prev));
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Predict start failed.");
      setStatusText("Failed");
    } finally {
      setRunning(false);
    }
  }

  async function onAbort() {
    if (readonly) return;
    await abortPredict();
    setStatusText("Aborted");
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
      setError(err instanceof Error ? err.message : "File upload failed.");
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
    if (!normalized) return "No file selected";
    return normalized.split("/").pop() || normalized;
  }
  const visibleError = readonly && isNotFoundLikeError(error) ? "" : error;

  return (
    <div className={`custom-model-page ${readonly ? "readonly-mode" : ""}`}>
      <header className="chat-header">
        <div>
          <h2>Custom Model Predict</h2>
          <p>Run single or batch inference with your selected custom model.</p>
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
          <section className="custom-section-card custom-section-wide">
            <label className="left-controls custom-field-short">Model Folder
              <select value={selectedFolder} onChange={(e) => setSelectedFolder(e.target.value)}>
                {folderOptions.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-medium">Model Path
              <select value={modelPath} onChange={(e) => setModelPath(e.target.value)}>
                <option value="">Select model</option>
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
              <h3 className="custom-panel-title">Predict Parameters</h3>
              <div className="custom-advanced-grid custom-advanced-panel custom-predict-params">
                {ckptConfigNote && <div className="custom-readonly-note custom-field-span-full">{ckptConfigNote}</div>}
                <div className="custom-param-group-title custom-field-span-full">Model Settings</div>
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
            <label className="left-controls custom-field-short">Problem Type
              <select value={problemType} onChange={(e) => setProblemType(e.target.value)} disabled={ckptLocked}>
                {(meta?.problem_types || []).length === 0 && <option value="">{emptySelectLabel}</option>}
                {(meta?.problem_types || []).map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            <label className="left-controls custom-field-short">Num Labels<input type="number" value={numLabels} onChange={(e) => setNumLabels(Number(e.target.value) || 1)} disabled={ckptLocked} /></label>
                <div className="custom-field-span-full predict-mode-row">
                  <div className="custom-mode-inline" role="group" aria-label="Prediction Mode">
                    <span className="custom-mode-inline-label">Prediction Mode</span>
                    <SegmentedSwitch
                      value={predictionMode}
                      onChange={setPredictionMode}
                      ariaLabel="Prediction mode switch"
                      className="custom-segment-switch-compact"
                      options={[
                        { value: "single", label: "Single" },
                        { value: "batch", label: "Batch" }
                      ]}
                    />
                  </div>
                  {predictionMode === "batch" && (
                    <label className="left-controls predict-batch-size-field">
                      Batch Size
                      <input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value) || 1)} />
                    </label>
                  )}
                </div>
                <div className="custom-param-group-title custom-field-span-full">Input Settings</div>
                {predictionMode === "single" ? (
                  isStructurePlm ? (
                    <p className="custom-readonly-note custom-field-span-full">Selected structure model uses PDB Folder based input.</p>
                  ) : (
                    <label className="left-controls custom-field-span-full">AA Sequence
                      <textarea rows={6} value={aaSeq} onChange={(e) => setAaSeq(e.target.value)} placeholder="Paste protein sequence..." />
                    </label>
                  )
                ) : (
                  <>
                    <div className="custom-batch-source-row custom-field-span-full predict-batch-source-row">
                      <span className="custom-mode-inline-label">Batch Source</span>
                      <SegmentedSwitch
                        value={batchInputSource}
                        onChange={(value) => setBatchInputSource(value as "upload" | "paste" | "path")}
                        ariaLabel="Batch input source switch"
                        className="custom-segment-switch-compact"
                        options={[
                          { value: "upload", label: "Upload" },
                          { value: "paste", label: "Paste FASTA" },
                          { value: "path", label: "Path" }
                        ]}
                      />
                    </div>
                    {batchInputSource === "upload" ? (
                      <div className="custom-upload-dropzone-wrap custom-field-span-2 predict-batch-input-row">
                        <div className="custom-upload-dropzone-grid">
                          <div className="custom-upload-item upload-source-stack">
                            <span className="custom-upload-item-label">File</span>
                            <label className="custom-upload-trigger">
                              <input
                                type="file"
                                accept=".csv,.tsv,.xlsx,.xls,.fasta,.fa,.txt"
                                onChange={(e) => void onUploadBatchFile(e.target.files?.[0] || null)}
                              />
                              Choose File
                            </label>
                            <WorkspaceFilePicker
                              workspaceEnabled={workspaceEnabled}
                              disabled={readonly || running}
                              acceptedCategories={["table_or_text", "sequence"]}
                              buttonLabel="From Workspace"
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
                              Use Example
                            </button>
                            <span className="custom-upload-file-chip">{displayUploadedName(inputFile)}</span>
                            <button type="button" className="custom-upload-clear-btn" onClick={clearBatchInputFile}>
                              Clear
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : batchInputSource === "paste" ? (
                      <label className="left-controls custom-field-span-2 predict-batch-input-row">Paste FASTA
                        <textarea
                          rows={5}
                          value={batchFastaText}
                          onChange={(e) => setBatchFastaText(e.target.value)}
                          placeholder=">seq1&#10;MKT...&#10;>seq2&#10;GAV..."
                        />
                      </label>
                    ) : (
                      <label className="left-controls custom-field-span-2 predict-batch-input-row">Input File Path
                        <input value={inputFile} onChange={(e) => setInputFile(e.target.value)} placeholder="e.g. data/test.csv or data/test.fasta" />
                      </label>
                    )}
                  </>
                )}

                {showStructureInputs && (
                  <>
                    <div className="custom-param-group-title custom-field-span-full">Structure Inputs</div>
                    {showStructureSeq && (
                      <div className="left-controls custom-field-span-2">
                        <span>Structure Seq</span>
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
                      <label className="left-controls custom-field-span-2">Foldseek Sequence
                        <textarea rows={3} value={foldseekSeq} onChange={(e) => setFoldseekSeq(e.target.value)} />
                      </label>
                    )}
                    {showSs8Input && (
                      <label className="left-controls custom-field-span-2">SS8 Sequence
                        <textarea rows={3} value={ss8Seq} onChange={(e) => setSs8Seq(e.target.value)} />
                      </label>
                    )}
                    {showPdbDir && <label className="left-controls custom-field-short">PDB Dir<input value={pdbDir} onChange={(e) => setPdbDir(e.target.value)} /></label>}
                  </>
                )}
                {isStructurePlm && (
                  <div className="custom-readonly-note custom-field-span-full">
                    Selected structure model requires PDB Folder.
                  </div>
                )}
                {predictRuleError && <div className="custom-readonly-note custom-field-span-full">{predictRuleError}</div>}
              </div>

              <div className="custom-actions-inline-status">
                <div className="custom-row custom-actions">
                  <button type="button" className="custom-btn-secondary" disabled={readonly} onClick={() => void onPreviewCommand()}>Preview Command</button>
                  <button type="button" className="custom-btn-primary" disabled={readonly || running} onClick={() => void onStart()}>Start Predict</button>
                  <button type="button" className="custom-btn-danger" disabled={readonly || !running} onClick={() => void onAbort()}>Abort</button>
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
