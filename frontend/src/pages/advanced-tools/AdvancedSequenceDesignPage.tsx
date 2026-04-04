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
  const [progressMessage, setProgressMessage] = useState("Idle");

  useEffect(() => {
    setError("");
  }, [uploadedPath]);

  useEffect(() => {
    void (async () => {
      try {
        const meta = await fetchAdvancedToolsMeta();
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
        throw new Error("ProteinMPNN Sequence Design requires .pdb input.");
      }
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadAdvancedDefaultExample("pdb");
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load example.");
    }
  }

  async function onRun() {
    setError("");
    setRunning(true);
    setProgress(0);
    setProgressMessage("Preparing ProteinMPNN task...");
    try {
      if (!uploadedPath) throw new Error("Please upload a PDB file first.");
      if (parsedTemperatures.length === 0) throw new Error("Temperatures must include at least one numeric value.");

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
      setProgressMessage("ProteinMPNN design completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <AdvancedToolsLayout
      title="Sequence Design (ProteinMPNN)"
      subtitle="Configure full ProteinMPNN inference options for structure-conditioned sequence design."
      running={running}
      progress={progress}
      progressMessage={progressMessage}
      left={
        <>
          <section className="custom-section-card">
            <h3>ProteinMPNN Core Design</h3>
            <p className="advanced-ai-note">
              Choose model family by data type first: Soluble for discovery/design, Vanilla for membrane proteins, CA only when
              you only have C-alpha coordinates.
            </p>
            <label className="left-controls">
              Model Family
              <select value={modelFamily} onChange={(e) => setModelFamily(e.target.value as ModelFamily)}>
                <option value="soluble">Soluble (recommended for protein discovery and design)</option>
                <option value="vanilla">Vanilla (recommended for membrane proteins)</option>
                <option value="ca">CA (only for C-alpha coarse-grained coordinates)</option>
              </select>
            </label>
            <label className="left-controls">
              Designed Chains (optional)
              <input
                value={designedChainsText}
                onChange={(e) => setDesignedChainsText(e.target.value)}
                placeholder="A,B (empty means all chains)"
              />
            </label>
            <label className="left-controls">
              Fixed Chains
              <input value={fixedChainsText} onChange={(e) => setFixedChainsText(e.target.value)} placeholder="e.g. A" />
            </label>
            <label className="left-controls">
              Temperatures
              <input
                value={temperaturesText}
                onChange={(e) => setTemperaturesText(e.target.value)}
                placeholder="0.1 or 0.1,0.2"
              />
            </label>
            <label className="left-controls">
              Number of sequences
              <input
                type="number"
                min={1}
                max={512}
                value={numSequences}
                onChange={(e) => setNumSequences(Number(e.target.value))}
              />
            </label>
            <label className="left-controls">
              Fixed Residues (optional)
              <textarea
                rows={2}
                className="advanced-two-line-text"
                value={fixedResiduesText}
                onChange={(e) => setFixedResiduesText(e.target.value)}
                placeholder="A12,C13 or A:12,13;B:5-8"
              />
            </label>
            <label className="quick-ai-toggle advanced-homomer-row">
              <input type="checkbox" checked={homomer} onChange={(e) => setHomomer(e.target.checked)} />
              <span className="quick-ai-toggle-box" />
              <span className="quick-ai-toggle-text">Enable homomer tying</span>
            </label>
          </section>

          <section className="custom-section-card">
            <h3>Model and Runtime</h3>
            <label className="left-controls">
              Model Name
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {modelOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label className="left-controls">
              Omit AAs
              <input value={omitAas} onChange={(e) => setOmitAas(e.target.value)} placeholder="X" />
            </label>
            <p className="advanced-ai-note">
              `v_48_020` (0.20A) is the default for most structures (AI-generated backbones, AlphaFold, routine redesign). Use
              `v_48_002` (0.02A) only for very high-resolution native structures.
            </p>
            <label className="left-controls">
              Seed
              <input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="0" />
            </label>
            <label className="left-controls">
              Batch Size
              <input value={batchSize} onChange={(e) => setBatchSize(e.target.value)} placeholder="1" />
            </label>
            <label className="left-controls">
              Max Length
              <input value={maxLength} onChange={(e) => setMaxLength(e.target.value)} placeholder="200000" />
            </label>
          </section>

          <section className="custom-section-card">
            <h3>Optional Advanced Rules</h3>
            <p className="advanced-ai-note">Use readable text rules. Backend converts them to JSON/JSONL automatically.</p>
            <label className="left-controls">
              Tied Positions
              <input
                value={tiedPositionsText}
                onChange={(e) => setTiedPositionsText(e.target.value)}
                placeholder="A12=B12;A13=B13"
              />
            </label>
            <label className="left-controls">
              Omit AA Rules
              <input
                value={omitAaRulesText}
                onChange={(e) => setOmitAaRulesText(e.target.value)}
                placeholder="A12:WY;B5:AP"
              />
            </label>
            <label className="left-controls">
              AA Bias
              <input
                value={aaBiasText}
                onChange={(e) => setAaBiasText(e.target.value)}
                placeholder="A:-1.1,F:0.7"
              />
            </label>
            <label className="left-controls">
              Bias By Residue
              <input
                value={biasByResidueText}
                onChange={(e) => setBiasByResidueText(e.target.value)}
                placeholder="A12:F=1.0|W=-0.3;B5:A=0.2"
              />
            </label>
            <label className="left-controls">
              PSSM Rules
              <input
                value={pssmRulesText}
                onChange={(e) => setPssmRulesText(e.target.value)}
                placeholder="Optional JSON or compact rules"
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
            <h3>Structure Input</h3>
            <div className="custom-file-example-row upload-source-stack">
              <div className="file-source-inline">
                <label className="left-controls custom-file-picker-field">
                  Select PDB File
                  <input type="file" accept=".pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                </label>
                <WorkspaceFilePicker
                  workspaceEnabled={workspaceEnabled}
                  disabled={running}
                  acceptedCategories={["structure"]}
                  buttonLabel="From Workspace"
                  onPick={(picked) => {
                    const selected = picked[0];
                    if (!selected) return;
                    setUploadedPath(selected.storage_path);
                  }}
                />
              </div>
              <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                Use Example PDB
              </button>
            </div>
            {uploadedPath && <div className="report-preview">Selected: {uploadedPath}</div>}
          </section>

          <button type="button" className="custom-btn-primary" onClick={() => void onRun()} disabled={running}>
            {running ? "Running..." : "Start Sequence Design"}
          </button>
          <button type="button" className="custom-btn-secondary" onClick={resetToDefaults} disabled={running}>
            Reset to Defaults
          </button>
        </>
      }
      right={
        <AdvancedResultPanel
          title="ProteinMPNN Sequence Design Result"
          resultPayload={resultPayload}
          aiSummary={(resultPayload?.ai_summary as string) || ""}
          error={error}
          showSummaryTab
        />
      }
    />
  );
}
