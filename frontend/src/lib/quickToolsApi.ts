import { streamSSEFromPost } from "./sse";

export type QuickToolsMeta = {
  dataset_mapping_zero_shot: string[];
  model_mapping_zero_shot: string[];
  dataset_mapping_function: string[];
  residue_mapping_function: string[];
  protein_properties_function: string[];
  llm_models: string[];
  mode?: "local" | "online";
  online_fasta_limit?: number;
  online_limit_enabled?: boolean;
};

export type QuickToolProgressEvent = {
  progress: number;
  message: string;
};

export type QuickSequenceDesignRequest = {
  structureFile: string;
  modelFamily?: "soluble" | "vanilla" | "ca";
  designedChains?: string[];
  fixedResiduesText?: string;
  numSequences: number;
  modelName?: string;
  backboneNoise?: number;
  useSolubleModel?: boolean;
  caOnly?: boolean;
  temperatures?: number[];
};

type UploadResponse = {
  file_path: string;
  name: string;
  suffix: string;
  content?: string;
};

const DEFAULT_META: QuickToolsMeta = {
  dataset_mapping_zero_shot: ["Activity", "Binding", "Expression", "Organismal Fitness", "Stability"],
  model_mapping_zero_shot: ["ESM2-650M", "ESM-IF1"],
  dataset_mapping_function: ["Solubility", "Subcellular Localization", "Membrane Protein", "Metal Ion Binding"],
  residue_mapping_function: ["Activity Site", "Binding Site", "Conserved Site", "Motif"],
  protein_properties_function: [
    "Physical and chemical properties",
    "Relative solvent accessible surface area (PDB only)",
    "SASA value (PDB only)",
    "Secondary structure (PDB only)"
  ],
  llm_models: ["DeepSeek", "ChatGPT", "Gemini"],
  mode: "local",
  online_fasta_limit: 50,
  online_limit_enabled: false
};

async function parseError(res: Response): Promise<string> {
  const fallback = `Request failed (${res.status})`;
  try {
    const data = (await res.json()) as { detail?: string };
    return data.detail || fallback;
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
  onProgress?: (evt: QuickToolProgressEvent) => void;
};

async function runQuickToolStream(options: StreamRunOptions): Promise<Record<string, unknown>> {
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

export async function fetchQuickToolsMeta(): Promise<QuickToolsMeta> {
  try {
    return await requestJSON<QuickToolsMeta>("/api/quick-tools/meta");
  } catch {
    return DEFAULT_META;
  }
}

export async function uploadQuickToolFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return requestJSON<UploadResponse>("/api/quick-tools/upload", {
    method: "POST",
    body: form
  });
}

export async function uploadSequenceAsFasta(sequence: string): Promise<UploadResponse> {
  const normalized = validateFastaWithHeader(sequence);
  const blob = new Blob([normalized], { type: "text/plain" });
  const file = new File([blob], "input.fasta");
  return uploadQuickToolFile(file);
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

export async function loadQuickToolDefaultExample(kind: "fasta" | "pdb" = "fasta"): Promise<UploadResponse> {
  return requestJSON<UploadResponse>(`/api/quick-tools/default-example?kind=${kind}`);
}

export function normalizePastedFastaForDisplay(content: string): string {
  const text = (content || "").trim();
  if (!text) return "";
  if (text.includes(">")) {
    return text.endsWith("\n") ? text : `${text}\n`;
  }
  return text.replace(/\s+/g, "");
}

export async function runMutationTool(args: {
  uploadedPath: string;
  uploadedSuffix: string;
  sequence: string;
  modelName: string;
}) {
  const sequenceForRequest =
    !args.uploadedPath && args.uploadedSuffix !== ".pdb" && args.sequence.trim()
      ? validateFastaWithHeader(args.sequence)
      : args.sequence.trim();
  const body = args.uploadedSuffix === ".pdb"
    ? {
        structure_file: args.uploadedPath,
        model_name: args.modelName.includes("IF1") ? args.modelName : "ESM-IF1"
      }
    : {
        sequence: sequenceForRequest || "",
        fasta_file: args.uploadedPath || "",
        model_name: args.modelName
      };
  return requestJSON<Record<string, unknown>>("/api/quick-tools/run/mutation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function runMutationToolStream(
  args: {
    uploadedPath: string;
    uploadedSuffix: string;
    sequence: string;
    modelName: string;
  },
  onProgress?: (evt: QuickToolProgressEvent) => void
) {
  const sequenceForRequest =
    !args.uploadedPath && args.uploadedSuffix !== ".pdb" && args.sequence.trim()
      ? validateFastaWithHeader(args.sequence)
      : args.sequence.trim();
  const body: Record<string, unknown> = args.uploadedSuffix === ".pdb"
    ? {
        structure_file: args.uploadedPath,
        model_name: args.modelName.includes("IF1") ? args.modelName : "ESM-IF1"
      }
    : {
        sequence: sequenceForRequest || "",
        fasta_file: args.uploadedPath || "",
        model_name: args.modelName
      };
  return runQuickToolStream({
    url: "/api/quick-tools/run/mutation/stream",
    body,
    onProgress
  });
}

export async function runProteinFunctionTool(args: {
  fastaFile: string;
  task: string;
}) {
  return requestJSON<Record<string, unknown>>("/api/quick-tools/run/protein-function", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      fasta_file: args.fastaFile,
      task: args.task,
      model_name: "ESM2-650M"
    })
  });
}

export async function runProteinFunctionToolStream(
  args: {
    fastaFile: string;
    task: string;
  },
  onProgress?: (evt: QuickToolProgressEvent) => void
) {
  return runQuickToolStream({
    url: "/api/quick-tools/run/protein-function/stream",
    body: {
      fasta_file: args.fastaFile,
      task: args.task,
      model_name: "ESM2-650M"
    },
    onProgress
  });
}

export async function runFunctionalResidueTool(args: {
  fastaFile: string;
  task: string;
}) {
  return requestJSON<Record<string, unknown>>("/api/quick-tools/run/residue-function", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      fasta_file: args.fastaFile,
      task: args.task,
      model_name: "ESM2-650M"
    })
  });
}

export async function runFunctionalResidueToolStream(
  args: {
    fastaFile: string;
    task: string;
  },
  onProgress?: (evt: QuickToolProgressEvent) => void
) {
  return runQuickToolStream({
    url: "/api/quick-tools/run/residue-function/stream",
    body: {
      fasta_file: args.fastaFile,
      task: args.task,
      model_name: "ESM2-650M"
    },
    onProgress
  });
}

export async function runPropertiesTool(args: {
  task: string;
  uploadedPath: string;
  chainId: string;
}) {
  return requestJSON<Record<string, unknown>>("/api/quick-tools/run/properties", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task: args.task,
      file_path: args.uploadedPath,
      chain_id: args.chainId
    })
  });
}

export async function runPropertiesToolStream(
  args: {
    task: string;
    uploadedPath: string;
    chainId: string;
  },
  onProgress?: (evt: QuickToolProgressEvent) => void
) {
  return runQuickToolStream({
    url: "/api/quick-tools/run/properties/stream",
    body: {
      task: args.task,
      file_path: args.uploadedPath,
      chain_id: args.chainId
    },
    onProgress
  });
}

export async function runSequenceDesignToolStream(
  args: QuickSequenceDesignRequest,
  onProgress?: (evt: QuickToolProgressEvent) => void
) {
  return runQuickToolStream({
    url: "/api/quick-tools/run/sequence-design/stream",
    body: {
      structure_file: args.structureFile,
      model_family: args.modelFamily ?? "soluble",
      designed_chains: args.designedChains ?? [],
      fixed_residues_text: args.fixedResiduesText ?? "",
      num_sequences: args.numSequences,
      model_name: args.modelName ?? "v_48_020",
      backbone_noise: args.backboneNoise ?? 0.2,
      use_soluble_model: args.useSolubleModel ?? true,
      ca_only: args.caOnly ?? false,
      temperatures: args.temperatures ?? [0.1]
    },
    onProgress
  });
}

export function getDownloadUrl(filePath: string, inline = false): string {
  const inlinePart = inline ? "&inline=1" : "";
  return `/api/quick-tools/download?file_path=${encodeURIComponent(filePath)}${inlinePart}`;
}

export function extractDownloadPath(payload: Record<string, unknown> | null): string {
  if (!payload) return "";
  const data = payload.data as Record<string, unknown> | undefined;
  const fileInfo = payload.file_info as Record<string, unknown> | undefined;
  if (typeof data?.csv_path === "string") return data.csv_path;
  if (typeof data?.heatmap_path === "string") return data.heatmap_path;
  if (typeof fileInfo?.file_path === "string") return fileInfo.file_path;
  return "";
}

export async function requestQuickToolAiSummary(args: {
  tool: string;
  task: string;
  provider: string;
  userApiKey: string;
  resultPayload: Record<string, unknown>;
}) {
  return requestJSON<{ summary: string; provider: string; model: string }>("/api/quick-tools/ai-summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tool: args.tool,
      task: args.task,
      llm_provider: args.provider,
      user_api_key: args.userApiKey,
      result_payload: args.resultPayload
    })
  });
}
