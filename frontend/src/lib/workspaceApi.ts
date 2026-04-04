export type WorkspaceFile = {
  id: string;
  display_name: string;
  storage_path: string;
  source: string;
  mime: string;
  suffix: string;
  size: number;
  created_at: string;
  tags: string[];
  category: string;
  bucket: "user_upload" | "chat_session" | "tool_upload" | "other";
  ttl_policy: string;
};

export type ListWorkspaceFilesResponse = {
  items: WorkspaceFile[];
  total: number;
  page: number;
  page_size: number;
  mode: "local" | "online";
  enabled: boolean;
};

export type WorkspaceTextFileResponse = {
  storage_path: string;
  content: string;
  line_count: number;
  truncated: boolean;
};

type ListParams = {
  q?: string;
  source?: string;
  fileType?: string;
  page?: number;
  pageSize?: number;
  sort?: "created_desc" | "name_asc" | "size_desc";
  includeSessions?: boolean;
};

function withQuery(path: string, params: Record<string, string | number | boolean | undefined>): string {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === "") return;
    qs.set(key, String(value));
  });
  const query = qs.toString();
  return query ? `${path}?${query}` : path;
}

async function parseError(res: Response): Promise<string> {
  const fallback = `Request failed (${res.status})`;
  try {
    const data = (await res.json()) as { detail?: string };
    return data.detail || fallback;
  } catch {
    return fallback;
  }
}

export async function listWorkspaceFiles(params: ListParams = {}): Promise<ListWorkspaceFilesResponse> {
  const url = withQuery("/api/workspace/files", {
    q: params.q,
    source: params.source,
    file_type: params.fileType,
    page: params.page ?? 1,
    page_size: params.pageSize ?? 50,
    sort: params.sort ?? "created_desc",
    include_sessions: Boolean(params.includeSessions)
  });
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as ListWorkspaceFilesResponse;
}

export async function uploadWorkspaceFile(file: File): Promise<WorkspaceFile> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/workspace/upload", {
    method: "POST",
    body: form
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  const data = (await res.json()) as { file: WorkspaceFile };
  return data.file;
}

export async function registerWorkspacePath(storagePath: string): Promise<WorkspaceFile> {
  const url = withQuery("/api/workspace/register", { storage_path: storagePath });
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  const data = (await res.json()) as { file: WorkspaceFile };
  return data.file;
}

export async function readWorkspaceTextFile(
  storagePath: string,
  opts: { maxLines?: number; maxChars?: number } = {}
): Promise<WorkspaceTextFileResponse> {
  const url = withQuery("/api/workspace/read-text", {
    storage_path: storagePath,
    max_lines: opts.maxLines ?? 5000,
    max_chars: opts.maxChars ?? 200000
  });
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as WorkspaceTextFileResponse;
}

export async function replaceWorkspaceFile(storagePath: string, file: File): Promise<WorkspaceFile> {
  const url = withQuery("/api/workspace/file", { storage_path: storagePath });
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(url, {
    method: "PATCH",
    body: form
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  const data = (await res.json()) as { file: WorkspaceFile };
  return data.file;
}

export async function deleteWorkspaceFile(storagePath: string): Promise<{ ok: boolean; deleted_path: string }> {
  const url = withQuery("/api/workspace/file", { storage_path: storagePath });
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as { ok: boolean; deleted_path: string };
}
