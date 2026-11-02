import { quotaError } from "./onlineQuota";
export type SseEvent = { event: string; data: string; id?: string };
type EventHandler = (event: SseEvent) => void;

function redactAbsolutePath(detail: unknown): string {
  const text = typeof detail === "string" ? detail : String(detail || "");
  if (!text) return "";
  return text.replace(/\/(?:home|Users|tmp|var|opt|mnt|data)(?:\/[^\s'"]+)+/g, "<redacted-path>");
}

function normalizeErrorText(raw: unknown): string {
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return "";
    try {
      const parsed = JSON.parse(trimmed) as { detail?: string; error?: string; message?: string };
      return (
        String(parsed?.detail || "").trim() ||
        String(parsed?.error || "").trim() ||
        String(parsed?.message || "").trim() ||
        trimmed
      );
    } catch {
      return trimmed;
    }
  }
  return String(raw || "").trim();
}

function toFriendlySseError(status: number, detail: string): string {
  const safeDetail = redactAbsolutePath(detail);
  const text = normalizeErrorText(safeDetail).toLowerCase();
  if (status === 429) {
    return safeDetail || "Daily chat quota reached in online mode. Please try again tomorrow.";
  }
  if (status === 404 || text.includes("not found")) {
    return "This feature is not available in online mode.";
  }
  if (status >= 500) {
    return "Server is temporarily unavailable. Please try again in a moment.";
  }
  if (safeDetail) return safeDetail;
  return `Request failed (${status}).`;
}

/** Parse one SSE frame without dropping whitespace or multiline data. */
export function parseSseFrame(frame: string): SseEvent | null {
  let event = "message", id: string | undefined;
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const key = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (key === "event") event = value;
    else if (key === "data") data.push(value);
    else if (key === "id" && !value.includes("\0")) id = value;
  }
  return data.length ? { event, data: data.join("\n"), id } : null;
}

class ResponseError extends Error {}

async function checkedResponse(response: Response) {
  if (response.ok && response.body) return response;
  let detail = "";
  try {
    const raw = await response.text();
    try {
      const parsed = JSON.parse(raw); quotaError(parsed);
      detail = typeof parsed.detail === "string" ? parsed.detail : parsed.detail?.message || parsed.error || parsed.message || "";
    } catch { detail = raw; }
  } catch { /* Network errors are handled by the reconnect loop. */ }
  throw new ResponseError(toFriendlySseError(response.status, detail));
}

function reconnectDelay(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); };
    const timer = setTimeout(() => { signal?.removeEventListener("abort", abort); resolve(); }, ms);
    if (signal?.aborted) abort(); else signal?.addEventListener("abort", abort, { once: true });
  });
}

async function streamSSE(url: string, body: unknown, onEvent: EventHandler, signal?: AbortSignal,
                         headers?: Record<string, string>, initialCursor = "", method: "GET" | "POST" = "POST") {
  const requestKey = crypto.randomUUID();
  let cursor = initialCursor, endpoint = url, runId = initialCursor.split(":")[0] || "", attempts = 0;
  let verb = method;
  const managedSubmission = Boolean(body && typeof body === "object" && "execution" in body && body.execution);
  while (true) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    try {
      const response = await checkedResponse(await fetch(endpoint, {
        method: verb, signal,
        headers: { ...(verb === "POST" ? { "Content-Type": "application/json", "Idempotency-Key": requestKey } : {}),
                   ...(headers || {}), ...(cursor ? { "Last-Event-ID": cursor } : {}) },
        ...(verb === "POST" ? { body: JSON.stringify(body) } : {})
      }));
      const returnedRun = response.headers.get("X-Run-ID");
      const resume = response.headers.get("X-Event-Stream");
      if (returnedRun) {
        if (runId && returnedRun !== runId) throw new ResponseError("The run changed while reconnecting.");
        runId = returnedRun;
        if (resume?.startsWith("/api/chat/sessions/")) { endpoint = resume; verb = "GET"; }
      }
      const reader = response.body!.getReader(), decoder = new TextDecoder();
      let buffer = "";
      try {
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });
          buffer = buffer.replace(/\r\n/g, "\n");
          const parts = buffer.split("\n\n"); buffer = parts.pop() || "";
          for (const part of parts) {
            const event = parseSseFrame(part);
            if (!event) continue;
            if (event.id) {
              const [identity, ordinal] = event.id.split(":");
              if (runId && identity !== runId) throw new ResponseError("Event belongs to another run.");
              if (cursor && identity === cursor.split(":")[0] && Number(ordinal) <= Number(cursor.split(":")[1])) continue;
              cursor = event.id;
            }
            onEvent(event); attempts = 0;
            if (event.event === "done") return;
          }
          if (done) break;
        }
      } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
      if (!runId) return; // Legacy endpoints retain their original EOF contract.
      throw new TypeError("Event stream disconnected");
    } catch (error) {
      if (signal?.aborted || error instanceof ResponseError || (!runId && !managedSubmission) || attempts >= 7) throw error;
      await reconnectDelay(Math.min(500 * 2 ** attempts++, 5000), signal);
    }
  }
}

export async function streamSSEFromPost(url: string, body: unknown, onEvent: EventHandler, signal?: AbortSignal, headers?: Record<string, string>) {
  return streamSSE(url, body, onEvent, signal, headers);
}

export async function streamSSEFromGet(url: string, onEvent: EventHandler, signal?: AbortSignal, headers?: Record<string, string>, after = "") {
  return streamSSE(url, undefined, onEvent, signal, headers, after, "GET");
}
