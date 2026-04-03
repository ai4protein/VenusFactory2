export type DownloadMethod = "Single ID" | "From File";

export type DownloadTaskResponse = {
  success: boolean;
  message: string;
  preview: string;
  archive_relative_path: string;
  task_folder: string;
  details: Record<string, unknown>;
};

type DownloadTaskBody = {
  method: DownloadMethod;
  id_value: string;
  file_content: string;
  save_error_file: boolean;
  merge?: boolean;
  file_type?: "pdb" | "cif";
  unzip?: boolean;
};

async function parseError(res: Response): Promise<string> {
  const fallback = `Request failed (${res.status})`;
  try {
    const data = (await res.json()) as { detail?: string; error?: string };
    return data.detail || data.error || fallback;
  } catch {
    return fallback;
  }
}

export async function runDownloadTask(endpoint: string, body: DownloadTaskBody): Promise<DownloadTaskResponse> {
  const res = await fetch(`/api/download/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as DownloadTaskResponse;
}

export function buildArchiveDownloadUrl(relativePath: string): string {
  const safePath = relativePath
    .split("/")
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join("/");
  return `/api/download/${safePath}`;
}
