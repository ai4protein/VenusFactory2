import { useEffect, useMemo, useState } from "react";
import { CopyableTextBlock } from "../components/CommandPreviewCard";
import { SegmentedSwitch } from "../components/SegmentedSwitch";

type MetaResponse = {
  dataset_mapping_zero_shot: string[];
  model_mapping_zero_shot: string[];
  dataset_mapping_function: string[];
  residue_mapping_function: string[];
  protein_properties_function: string[];
};

type ToolTab = "mutation" | "function" | "residue" | "properties";

const DEFAULT_META: MetaResponse = {
  dataset_mapping_zero_shot: ["Activity", "Binding", "Expression", "Organismal Fitness", "Stability"],
  model_mapping_zero_shot: ["ESM2-650M", "ESM-IF1"],
  dataset_mapping_function: ["Solubility", "Subcellular Localization", "Membrane Protein", "Metal Ion Binding"],
  residue_mapping_function: ["Activity Site", "Binding Site", "Conserved Site", "Motif"],
  protein_properties_function: [
    "Physical and chemical properties",
    "Relative solvent accessible surface area (PDB only)",
    "SASA value (PDB only)",
    "Secondary structure (PDB only)"
  ]
};

export function QuickToolsPage() {
  const [tab, setTab] = useState<ToolTab>("mutation");
  const [meta, setMeta] = useState<MetaResponse>(DEFAULT_META);
  const [metaLoaded, setMetaLoaded] = useState(false);
  const [sequence, setSequence] = useState("");
  const [chainId, setChainId] = useState("A");
  const [uploadedPath, setUploadedPath] = useState("");
  const [uploadedSuffix, setUploadedSuffix] = useState("");
  const [selectedMutationModel, setSelectedMutationModel] = useState("ESM2-650M");
  const [selectedFunctionTask, setSelectedFunctionTask] = useState(DEFAULT_META.dataset_mapping_function[0]);
  const [selectedResidueTask, setSelectedResidueTask] = useState(DEFAULT_META.residue_mapping_function[0]);
  const [selectedPropertyTask, setSelectedPropertyTask] = useState(
    DEFAULT_META.protein_properties_function[0]
  );
  const [loading, setLoading] = useState(false);
  const [resultText, setResultText] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    void ensureMeta();
  }, []);

  async function ensureMeta() {
    if (metaLoaded) return;
    try {
      const res = await fetch("/api/quick-tools/meta");
      if (!res.ok) return;
      const data = (await res.json()) as MetaResponse;
      setMeta(data);
      if (data.dataset_mapping_function.length) setSelectedFunctionTask(data.dataset_mapping_function[0]);
      if (data.residue_mapping_function.length) setSelectedResidueTask(data.residue_mapping_function[0]);
      if (data.protein_properties_function.length)
        setSelectedPropertyTask(data.protein_properties_function[0]);
      if (data.model_mapping_zero_shot.length)
        setSelectedMutationModel(data.model_mapping_zero_shot[0]);
      setMetaLoaded(true);
    } catch {
      // Keep default meta as fallback
      setMetaLoaded(true);
    }
  }

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      const data = await uploadFileAndGetPath(file);
      setUploadedPath(data.file_path);
      setUploadedSuffix(data.suffix);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setLoading(false);
    }
  }

  async function uploadFileAndGetPath(file: File) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/quick-tools/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    return (await res.json()) as { file_path: string; suffix: string };
  }

  async function runMutation() {
    const body =
      uploadedSuffix === ".pdb"
        ? {
            structure_file: uploadedPath,
            model_name: selectedMutationModel.includes("IF1") ? selectedMutationModel : "ESM-IF1"
          }
        : {
            sequence: sequence.trim() || undefined,
            fasta_file: uploadedPath || undefined,
            model_name: selectedMutationModel
          };
    const url =
      uploadedSuffix === ".pdb"
        ? "/api/v1/mutation/zero-shot/structure"
        : "/api/v1/mutation/zero-shot/sequence";
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  }

  async function runFunction() {
    if (!uploadedPath && !sequence.trim()) {
      throw new Error("Please upload a FASTA file or input sequence.");
    }
    let fastaPath = uploadedPath;
    if (!fastaPath && sequence.trim()) {
      // Mutation endpoint accepts raw sequence, function API requires fasta_file.
      // We convert by uploading a temporary in-browser file.
      const blob = new Blob([`>input\n${sequence.trim()}\n`], { type: "text/plain" });
      const file = new File([blob], "input.fasta");
      const uploaded = await uploadFileAndGetPath(file);
      fastaPath = uploaded.file_path;
      setUploadedPath(uploaded.file_path);
      setUploadedSuffix(uploaded.suffix);
    }
    return fetch("/api/v1/predict/finetuned/protein-function", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fasta_file: fastaPath || uploadedPath,
        task: selectedFunctionTask,
        model_name: "ESM2-650M"
      })
    });
  }

  async function runResidue() {
    if (!uploadedPath && !sequence.trim()) {
      throw new Error("Please upload a FASTA file or input sequence.");
    }
    let fastaPath = uploadedPath;
    if (!fastaPath && sequence.trim()) {
      const blob = new Blob([`>input\n${sequence.trim()}\n`], { type: "text/plain" });
      const file = new File([blob], "input.fasta");
      const uploaded = await uploadFileAndGetPath(file);
      fastaPath = uploaded.file_path;
      setUploadedPath(uploaded.file_path);
      setUploadedSuffix(uploaded.suffix);
    }
    return fetch("/api/v1/predict/finetuned/residue-function", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fasta_file: fastaPath || uploadedPath,
        task: selectedResidueTask,
        model_name: "ESM2-650M"
      })
    });
  }

  async function runProperties() {
    if (!uploadedPath) {
      throw new Error("Properties requires uploaded FASTA/PDB file.");
    }
    if (selectedPropertyTask === "Physical and chemical properties") {
      return fetch("/api/v1/predict/features/physchem", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fasta_file: uploadedPath })
      });
    }
    if (selectedPropertyTask.includes("Relative solvent")) {
      return fetch("/api/v1/predict/features/rsa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pdb_file: uploadedPath, chain_id: chainId })
      });
    }
    if (selectedPropertyTask.includes("SASA")) {
      return fetch("/api/v1/predict/features/sasa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pdb_file: uploadedPath })
      });
    }
    return fetch("/api/v1/predict/features/secondary-structure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdb_file: uploadedPath, chain_id: chainId })
    });
  }

  async function runCurrentTool() {
    setError("");
    setLoading(true);
    try {
      await ensureMeta();
      let res: Response;
      if (tab === "mutation") res = await runMutation();
      else if (tab === "function") res = await runFunction();
      else if (tab === "residue") res = await runResidue();
      else res = await runProperties();

      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`);
      }
      const data = await res.json();
      setResultText(JSON.stringify(data, null, 2));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Execution failed.");
    } finally {
      setLoading(false);
    }
  }

  const tabTitle = useMemo(() => {
    if (tab === "mutation") return "Directed Evolution";
    if (tab === "function") return "Protein Function";
    if (tab === "residue") return "Functional Residue";
    return "Physicochemical Property";
  }, [tab]);

  return (
    <div className="quick-tools-page">
      <header className="chat-header">
        <div>
          <h2>Quick Tools</h2>
          <p>Fast entry for directed evolution, function prediction, residue analysis and properties.</p>
        </div>
      </header>

      <section className="quick-tools-grid">
        <aside className="chat-panel left">
          <h3>Tool Selection</h3>
          <div className="quick-tool-tabs">
            <SegmentedSwitch
              value={tab}
              onChange={setTab}
              ariaLabel="Quick tools tab switch"
              className="quick-tool-segment-switch"
              options={[
                { value: "mutation", label: "Directed Evolution" },
                { value: "function", label: "Function" },
                { value: "residue", label: "Residue" },
                { value: "properties", label: "Properties" }
              ]}
            />
          </div>

          <h3>Input</h3>
          <textarea
            rows={8}
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Paste sequence here (optional for mutation/function/residue)..."
          />
          <input type="file" accept=".fasta,.fa,.pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
          {uploadedPath && <div className="report-preview">Uploaded: {uploadedPath}</div>}

          {tab === "mutation" && (
            <label className="left-controls">
              Model
              <select value={selectedMutationModel} onChange={(e) => setSelectedMutationModel(e.target.value)}>
                {meta.model_mapping_zero_shot.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
          )}

          {tab === "function" && (
            <label className="left-controls">
              Task
              <select value={selectedFunctionTask} onChange={(e) => setSelectedFunctionTask(e.target.value)}>
                {meta.dataset_mapping_function.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
          )}

          {tab === "residue" && (
            <label className="left-controls">
              Task
              <select value={selectedResidueTask} onChange={(e) => setSelectedResidueTask(e.target.value)}>
                {meta.residue_mapping_function.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
          )}

          {tab === "properties" && (
            <>
              <label className="left-controls">
                Property
                <select value={selectedPropertyTask} onChange={(e) => setSelectedPropertyTask(e.target.value)}>
                  {meta.protein_properties_function.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>
              <label className="left-controls">
                PDB Chain
                <input value={chainId} onChange={(e) => setChainId(e.target.value)} />
              </label>
            </>
          )}

          <button type="button" onClick={() => void runCurrentTool()} disabled={loading}>
            {loading ? "Running..." : `Run ${tabTitle}`}
          </button>
          {error && <div className="error-box">{error}</div>}
        </aside>

        <section className="chat-panel center report-output">
          <div className="report-result-header">
            <h3>{tabTitle} Result</h3>
          </div>
          <CopyableTextBlock
            text={resultText}
            emptyText="Result JSON will appear here..."
            wrapperClassName="report-copy-wrap"
            preClassName="report-text"
            ariaLabel="Copy quick tool result"
          />
        </section>
      </section>
    </div>
  );
}
