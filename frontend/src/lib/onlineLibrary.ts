import { getChatSessionAuthHeaders, listChatSessions } from "./api";

export type LibraryFile = {
  id: string; source: "chat" | "report"; identity: string; key: string;
  name: string; size: number; modified: number; mode: string; title: string; extension: string;
};

async function request(path: string, payload: unknown, signal?: AbortSignal) {
  const response = await fetch(`/api/library/${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload), cache: "no-store", signal
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : `Unable to load file (${response.status})`);
  }
  return response;
}

export async function loadOnlineLibrary(signal: AbortSignal) {
  const { sessions } = await listChatSessions();
  signal.throwIfAborted();
  const refs = sessions.flatMap(session => {
    const token = getChatSessionAuthHeaders(session.session_id)["x-session-access-token"];
    return token ? [{ identity: session.session_id, token }] : [];
  });
  const titles = new Map(sessions.map(s => [s.session_id, s.title]));
  const items: LibraryFile[] = [];
  let truncated = false;
  // Each batch is authenticated independently; no session credentials appear in URLs.
  for (let offset = 0; offset < Math.max(1, refs.length); offset += 100) {
    const response = await request("files", { sessions: refs.slice(offset, offset + 100), include_reports: offset === 0 }, signal);
    const data = await response.json() as { items: LibraryFile[]; truncated: boolean };
    items.push(...data.items.map(item => ({ ...item, title: item.source === "chat" ? titles.get(item.identity) || item.title : item.title })));
    truncated ||= data.truncated;
  }
  return { items, truncated };
}

function reference(item: LibraryFile) {
  return { source: item.source, identity: item.identity, key: item.key,
    token: item.source === "chat" ? getChatSessionAuthHeaders(item.identity)["x-session-access-token"] || "" : "" };
}

export async function libraryBlob(item: LibraryFile, signal?: AbortSignal) {
  return (await request("content", reference(item), signal)).blob();
}

export function saveLibraryBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function libraryArchive(items: LibraryFile[]) {
  const response = await request("archive", { files: items.map(reference) });
  saveLibraryBlob(await response.blob(), "library-files.zip");
}
