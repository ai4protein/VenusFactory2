import { streamSSEFromPost } from "./sse";

export type CustomModelMeta = {
  plm_models: Record<string, string>;
  dataset_configs: Record<string, string>;
  training_methods: string[];
  pooling_methods: string[];
  problem_types: string[];
  metrics_options: string[];
  structure_seq_options: string[];
};

export type DatasetPreviewResult = {
  dataset_name: string;
  stats: { train: number; validation: number; test: number };
  preview: { columns: string[]; rows: Record<string, string>[] };
  column_options?: string[];
};

export type ModelOption = { label: string; path: string };
export type DatasetConfigDefaults = {
  problem_type?: string;
  num_labels?: number;
  metrics?: string[];
  sequence_column_name?: string;
  label_column_name?: string;
  structure_seq?: string[];
  pdb_dir?: string;
};
export type UploadedDatasetFile = {
  file_path: string;
  name: string;
  suffix: string;
  columns: string[];
  run_id?: string;
};

export type StreamEventPayload =
  | { type: "start"; data: { run_id: string; command: string; task: string } }
  | { type: "progress"; data: { progress: number; message: string } }
  | { type: "log"; data: { line: string } }
  | { type: "training_metrics"; data: { epochs: number[]; train_loss: Array<number | null>; val_loss: Array<number | null>; val_metrics: Record<string, Array<number | null>> } }
  | { type: "test_results"; data: { metrics: Record<string, number>; csv_download_url?: string } }
  | {
      type: "done";
      data: {
        success: boolean;
        aborted: boolean;
        message: string;
        run_id: string;
        final_progress?: number;
        csv_download_url?: string;
        test_metrics?: Record<string, number>;
        training_metrics?: {
          epochs: number[];
          train_loss: Array<number | null>;
          val_loss: Array<number | null>;
          val_metrics: Record<string, Array<number | null>>;
        };
      };
    }
  | { type: "error"; data: { message: string } };

async function requestJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export async function fetchCustomModelMeta() {
  return requestJSON<CustomModelMeta>("/api/custom-model/meta");
}

export async function fetchModelFolders(root = "ckpt") {
  return requestJSON<{ folders: string[] }>(`/api/custom-model/models/folders?root=${encodeURIComponent(root)}`);
}

export async function fetchModelsInFolder(folder: string) {
  return requestJSON<{ models: ModelOption[] }>(`/api/custom-model/models?folder=${encodeURIComponent(folder)}`);
}

export async function fetchModelConfig(modelPath: string) {
  return requestJSON<{ config: Record<string, unknown> }>(
    `/api/custom-model/models/config?model_path=${encodeURIComponent(modelPath)}`
  );
}

export async function previewDataset(payload: {
  dataset_selection: "Custom" | "Pre-defined";
  dataset_config?: string;
  dataset_custom?: string;
  train_file?: string;
  valid_file?: string;
  test_file?: string;
}) {
  return requestJSON<DatasetPreviewResult>("/api/custom-model/dataset/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function uploadCustomModelDatasetFile(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/custom-model/upload", {
    method: "POST",
    body: form
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Upload failed (${res.status})`);
  }
  return (await res.json()) as UploadedDatasetFile;
}

export async function uploadCustomModelPredictBatchFile(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/custom-model/predict/upload", {
    method: "POST",
    body: form
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Upload failed (${res.status})`);
  }
  return (await res.json()) as UploadedDatasetFile;
}

export async function uploadCustomModelPredictBatchText(text: string, filename = "batch_input.fasta") {
  return requestJSON<UploadedDatasetFile>("/api/custom-model/predict/text-upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, filename })
  });
}

export async function fetchDatasetConfigDefaults(datasetConfig: string) {
  return requestJSON<DatasetConfigDefaults>(
    `/api/custom-model/dataset/config-values?dataset_config=${encodeURIComponent(datasetConfig)}`
  );
}

export async function previewTraining(args: Record<string, unknown>) {
  return requestJSON<{ command: string; args: Record<string, unknown> }>("/api/custom-model/training/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ args })
  });
}

export async function previewEvaluation(args: Record<string, unknown>) {
  return requestJSON<{ command: string; args: Record<string, unknown> }>("/api/custom-model/evaluation/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ args })
  });
}

export async function previewPredict(args: Record<string, unknown>) {
  return requestJSON<{ command: string; args: Record<string, unknown> }>("/api/custom-model/predict/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ args })
  });
}

export async function abortTraining() {
  return requestJSON<{ aborted: boolean }>("/api/custom-model/training/abort", { method: "POST" });
}

export async function abortEvaluation() {
  return requestJSON<{ aborted: boolean }>("/api/custom-model/evaluation/abort", { method: "POST" });
}

export async function abortPredict() {
  return requestJSON<{ aborted: boolean }>("/api/custom-model/predict/abort", { method: "POST" });
}

function parseEventData(data: string) {
  if (!data) return {};
  try {
    return JSON.parse(data) as Record<string, unknown>;
  } catch {
    return { raw: data };
  }
}

export async function startTrainingStream(
  args: Record<string, unknown>,
  onEvent: (payload: StreamEventPayload) => void,
  signal?: AbortSignal
) {
  await streamSSEFromPost(
    "/api/custom-model/training/start",
    { args },
    ({ event, data }) => {
      const payload = parseEventData(data);
      if (event === "start") onEvent({ type: "start", data: payload as StreamEventPayload["data"] & { run_id: string } });
      else if (event === "progress") onEvent({ type: "progress", data: payload as StreamEventPayload["data"] & { progress: number } });
      else if (event === "log") onEvent({ type: "log", data: payload as StreamEventPayload["data"] & { line: string } });
      else if (event === "training_metrics") onEvent({ type: "training_metrics", data: payload as StreamEventPayload["data"] & { epochs: number[] } });
      else if (event === "test_results") onEvent({ type: "test_results", data: payload as StreamEventPayload["data"] & { metrics: Record<string, number> } });
      else if (event === "error") onEvent({ type: "error", data: payload as StreamEventPayload["data"] & { message: string } });
      else if (event === "done") onEvent({ type: "done", data: payload as StreamEventPayload["data"] & { success: boolean } });
    },
    signal
  );
}

export async function startEvaluationStream(
  args: Record<string, unknown>,
  onEvent: (payload: StreamEventPayload) => void,
  signal?: AbortSignal
) {
  await streamSSEFromPost(
    "/api/custom-model/evaluation/start",
    { args },
    ({ event, data }) => {
      const payload = parseEventData(data);
      if (event === "start") onEvent({ type: "start", data: payload as StreamEventPayload["data"] & { run_id: string } });
      else if (event === "progress") onEvent({ type: "progress", data: payload as StreamEventPayload["data"] & { progress: number } });
      else if (event === "log") onEvent({ type: "log", data: payload as StreamEventPayload["data"] & { line: string } });
      else if (event === "error") onEvent({ type: "error", data: payload as StreamEventPayload["data"] & { message: string } });
      else if (event === "done") onEvent({ type: "done", data: payload as StreamEventPayload["data"] & { success: boolean } });
    },
    signal
  );
}

export async function startPredictStream(
  args: Record<string, unknown>,
  onEvent: (payload: StreamEventPayload) => void,
  signal?: AbortSignal
) {
  await streamSSEFromPost(
    "/api/custom-model/predict/start",
    { args },
    ({ event, data }) => {
      const payload = parseEventData(data);
      if (event === "start") onEvent({ type: "start", data: payload as StreamEventPayload["data"] & { run_id: string } });
      else if (event === "progress") onEvent({ type: "progress", data: payload as StreamEventPayload["data"] & { progress: number } });
      else if (event === "log") onEvent({ type: "log", data: payload as StreamEventPayload["data"] & { line: string } });
      else if (event === "error") onEvent({ type: "error", data: payload as StreamEventPayload["data"] & { message: string } });
      else if (event === "done") onEvent({ type: "done", data: payload as StreamEventPayload["data"] & { success: boolean } });
    },
    signal
  );
}

export function buildArtifactDownloadUrl(path: string) {
  return `/api/custom-model/artifacts/download?path=${encodeURIComponent(path)}`;
}
