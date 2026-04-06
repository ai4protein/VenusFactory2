import { streamSSEFromPost } from "./sse";

export type AdvancedToolsMeta = {
  dataset_mapping_zero_shot: string[];
  sequence_model_options: string[];
  structure_model_options: string[];
  model_mapping_function: string[];
  residue_model_mapping_function: string[];
  dataset_mapping_function: Record<string, string[]>;
  residue_mapping_function: Record<string, string[]>;
  llm_models: string[];
  proteinmpnn_model_options?: {
    vanilla: string[];
    soluble: string[];
    ca: string[];
  };
  mode?: "local" | "online";
  online_fasta_limit?: number;
  online_sequence_design_limit?: number;
  online_limit_enabled?: boolean;
};

export type AdvancedToolProgressEvent = {
  progress: number;
  message: string;
};

type UploadResponse = {
  file_path: string;
  name: string;
  suffix: string;
  content?: string;
  kind?: "fasta" | "pdb";
};

type DirectedEvolutionRequest = {
  input_mode: "sequence" | "structure";
  function_selection?: string;
  file_path?: string;
  sequence?: string;
  model_name: string;
  enable_ai: boolean;
  llm_provider: string;
  user_api_key: string;
};

type ProteinFunctionRequest = {
  task: string;
  file_path?: string;
  sequence?: string;
  model_name: string;
  datasets: string[];
  enable_ai: boolean;
  llm_provider: string;
  user_api_key: string;
};

type FunctionalResidueRequest = {
  task: string;
  file_path?: string;
  sequence?: string;
  model_name: string;
  enable_ai: boolean;
  llm_provider: string;
  user_api_key: string;
};

type AdvancedAiSummaryRequest = {
  tool: string;
  task: string;
  llm_provider: string;
  user_api_key: string;
  result_payload: Record<string, unknown>;
};

type ProteinDiscoveryRequest = {
  pdb_file: string;
  protect_start: number;
  protect_end: number;
  mmseqs_threads: number;
  mmseqs_iterations: number;
  mmseqs_max_seqs: number;
  cluster_min_seq_id: number;
  cluster_threads: number;
  top_n_threshold: number;
  evalue_threshold: number;
};

export type AdvancedSequenceDesignRequest = {
  structure_file: string;
  model_family?: "soluble" | "vanilla" | "ca";
  designed_chains?: string[];
  fixed_chains?: string[];
  fixed_residues_text?: string;
  homomer?: boolean;
  num_sequences?: number;
  temperatures?: number[];
  omit_aas?: string;
  model_name?: string;
  backbone_noise?: number;
  ca_only?: boolean;
  use_soluble_model?: boolean;
  seed?: number;
  batch_size?: number;
  max_length?: number;
  tied_positions_text?: string;
  omit_aa_rules_text?: string;
  aa_bias_text?: string;
  bias_by_residue_text?: string;
  pssm_rules_text?: string;
  pssm_multi?: number;
  pssm_threshold?: number;
  pssm_log_odds_flag?: number;
  pssm_bias_flag?: number;
};

const DEFAULT_META: AdvancedToolsMeta = {
  dataset_mapping_zero_shot: ["Activity", "Binding", "Expression", "Organismal Fitness", "Stability"],
  sequence_model_options: ["VenusPLM", "ESM2-650M", "ESM-1v", "ESM-1b"],
  structure_model_options: ["VenusREM (foldseek-based)", "ProSST-2048", "ProtSSN", "ESM-IF1", "SaProt", "MIF-ST"],
  model_mapping_function: ["ESM2-650M"],
  residue_model_mapping_function: ["ESM2-650M"],
  dataset_mapping_function: { Solubility: ["DeepSol"] },
  residue_mapping_function: { "Activity Site": ["Protein_Mutation"] },
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"],
  proteinmpnn_model_options: {
    vanilla: ["v_48_020", "v_48_002"],
    soluble: ["v_48_020", "v_48_002"],
    ca: ["v_48_020", "v_48_002"]
  },
  mode: "local",
  online_fasta_limit: 50,
  online_sequence_design_limit: 50,
  online_limit_enabled: false
};

async function parseError(res: Response): Promise<string> {
  const fallback = `Request failed (${res.status})`;
  try {
    const data = (await res.json()) as { detail?: string; error?: string; message?: string };
    return data.detail || data.error || data.message || fallback;
  } catch {
    return fallback;
  }
}

async function requestJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as T;
}

type StreamRunOptions = {
  url: string;
  body: Record<string, unknown>;
  onProgress?: (evt: AdvancedToolProgressEvent) => void;
};

async function runAdvancedToolStream(options: StreamRunOptions): Promise<Record<string, unknown>> {
  let finalPayload: Record<string, unknown> | null = null;
  let streamError = "";

  await streamSSEFromPost(options.url, options.body, ({ event, data }) => {
    let parsed: Record<string, unknown> = {};
    try {
      parsed = data ? (JSON.parse(data) as Record<string, unknown>) : {};
    } catch {
      parsed = {};
    }

    if (event === "progress") {
      const rawProgress = typeof parsed.progress === "number" ? parsed.progress : 0;
      options.onProgress?.({
        progress: Math.max(0, Math.min(1, rawProgress)),
        message: String(parsed.message || "Running...")
      });
      return;
    }

    if (event === "error") {
      streamError = String(parsed.message || "Run failed.");
      return;
    }

    if (event === "done") {
      if (parsed.success === false) {
        streamError = String(parsed.message || streamError || "Run failed.");
      }
      const payload = parsed.result_payload;
      if (payload && typeof payload === "object") {
        finalPayload = payload as Record<string, unknown>;
      }
      if (typeof parsed.final_progress === "number") {
        options.onProgress?.({
          progress: Math.max(0, Math.min(1, parsed.final_progress)),
          message: String(parsed.message || "Completed")
        });
      }
    }
  });

  if (streamError) throw new Error(streamError);
  if (!finalPayload) throw new Error("No result payload returned from stream.");
  return finalPayload;
}

export async function fetchAdvancedToolsMeta(): Promise<AdvancedToolsMeta> {
  try {
    return await requestJSON<AdvancedToolsMeta>("/api/advanced-tools/meta");
  } catch {
    return DEFAULT_META;
  }
}

export async function uploadAdvancedToolFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return requestJSON<UploadResponse>("/api/advanced-tools/upload", {
    method: "POST",
    body: form
  });
}

export async function loadAdvancedDefaultExample(kind: "fasta" | "pdb" = "fasta"): Promise<UploadResponse> {
  return requestJSON<UploadResponse>(`/api/advanced-tools/default-example?kind=${kind}`);
}

export async function runAdvancedDirectedEvolution(body: DirectedEvolutionRequest) {
  return requestJSON<Record<string, unknown>>("/api/advanced-tools/directed-evolution/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function runAdvancedDirectedEvolutionStream(
  body: DirectedEvolutionRequest,
  onProgress?: (evt: AdvancedToolProgressEvent) => void
) {
  return runAdvancedToolStream({
    url: "/api/advanced-tools/directed-evolution/run/stream",
    body: body as unknown as Record<string, unknown>,
    onProgress
  });
}

export async function runAdvancedProteinFunction(body: ProteinFunctionRequest) {
  return requestJSON<Record<string, unknown>>("/api/advanced-tools/protein-function/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function runAdvancedProteinFunctionStream(
  body: ProteinFunctionRequest,
  onProgress?: (evt: AdvancedToolProgressEvent) => void
) {
  return runAdvancedToolStream({
    url: "/api/advanced-tools/protein-function/run/stream",
    body: body as unknown as Record<string, unknown>,
    onProgress
  });
}

export async function runAdvancedFunctionalResidue(body: FunctionalResidueRequest) {
  return requestJSON<Record<string, unknown>>("/api/advanced-tools/functional-residue/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function runAdvancedFunctionalResidueStream(
  body: FunctionalResidueRequest,
  onProgress?: (evt: AdvancedToolProgressEvent) => void
) {
  return runAdvancedToolStream({
    url: "/api/advanced-tools/functional-residue/run/stream",
    body: body as unknown as Record<string, unknown>,
    onProgress
  });
}

export async function runAdvancedProteinDiscovery(body: ProteinDiscoveryRequest) {
  return requestJSON<Record<string, unknown>>("/api/advanced-tools/protein-discovery/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function runAdvancedProteinDiscoveryStream(
  body: ProteinDiscoveryRequest,
  onProgress?: (evt: AdvancedToolProgressEvent) => void
) {
  return runAdvancedToolStream({
    url: "/api/advanced-tools/protein-discovery/run/stream",
    body: body as unknown as Record<string, unknown>,
    onProgress
  });
}

export async function runAdvancedSequenceDesignStream(
  body: AdvancedSequenceDesignRequest,
  onProgress?: (evt: AdvancedToolProgressEvent) => void
) {
  return runAdvancedToolStream({
    url: "/api/advanced-tools/sequence-design/run/stream",
    body: body as unknown as Record<string, unknown>,
    onProgress
  });
}

export async function requestAdvancedAiSummary(body: AdvancedAiSummaryRequest) {
  return requestJSON<{ summary: string; provider: string; model: string }>("/api/advanced-tools/ai-summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export function validateFastaWithHeader(content: string): string {
  const trimmed = content.trim();
  if (!trimmed) throw new Error("FASTA content is empty.");

  const lines = trimmed.split(/\r?\n/).map((line) => line.trim()).filter((line) => line.length > 0);
  if (lines.length === 0) throw new Error("FASTA content is empty.");
  if (!lines[0].startsWith(">")) throw new Error("FASTA must include header line starting with >");

  const normalized: string[] = [];
  let currentHeader = "";
  let currentSequence = "";

  for (const line of lines) {
    if (line.startsWith(">")) {
      if (line.length <= 1) throw new Error("FASTA header cannot be empty.");
      if (currentHeader) {
        if (!currentSequence) throw new Error(`Sequence under header '${currentHeader.slice(1)}' is empty.`);
        normalized.push(currentHeader, currentSequence);
      }
      currentHeader = line;
      currentSequence = "";
      continue;
    }
    if (!currentHeader) throw new Error("FASTA must include header line starting with >");
    currentSequence += line.toUpperCase().replace(/[^A-Z]/g, "");
  }

  if (!currentHeader) throw new Error("FASTA must include header line starting with >");
  if (!currentSequence) throw new Error(`Sequence under header '${currentHeader.slice(1)}' is empty.`);
  normalized.push(currentHeader, currentSequence);
  return `${normalized.join("\n")}\n`;
}

export function splitNormalizedFastaRecords(normalized: string): Array<{ header: string; sequence: string; fasta: string }> {
  const records: Array<{ header: string; sequence: string; fasta: string }> = [];
  let currentHeader = "";
  let currentSequence = "";
  for (const line of normalized.split(/\r?\n/)) {
    const text = line.trim();
    if (!text) continue;
    if (text.startsWith(">")) {
      if (currentHeader && currentSequence) {
        records.push({ header: currentHeader.slice(1).trim(), sequence: currentSequence, fasta: `${currentHeader}\n${currentSequence}\n` });
      }
      currentHeader = text;
      currentSequence = "";
      continue;
    }
    currentSequence += text;
  }
  if (currentHeader && currentSequence) {
    records.push({ header: currentHeader.slice(1).trim(), sequence: currentSequence, fasta: `${currentHeader}\n${currentSequence}\n` });
  }
  return records;
}

export function getAdvancedDownloadUrl(filePath: string, inline = false): string {
  const inlinePart = inline ? "&inline=1" : "";
  return `/api/advanced-tools/download?file_path=${encodeURIComponent(filePath)}${inlinePart}`;
}

export function extractAdvancedDownloadPath(payload: Record<string, unknown> | null): string {
  if (!payload) return "";
  if (typeof payload.download_path === "string") return payload.download_path;
  const result = payload.result as Record<string, unknown> | undefined;
  if (typeof result?.download_path === "string") return result.download_path;
  if (typeof result?.download_archive === "string") return result.download_archive;
  if (typeof result?.download_csv === "string") return result.download_csv;
  if (typeof result?.zip_download === "string") return result.zip_download;
  return "";
}
