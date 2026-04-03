type EventHandler = (event: { event: string; data: string }) => void;

function redactAbsolutePath(detail: string): string {
  if (!detail) return "";
  return detail.replace(/\/(?:home|Users|tmp|var|opt|mnt|data)(?:\/[^\s'"]+)+/g, "<redacted-path>");
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

export async function streamSSEFromPost(
  url: string,
  body: unknown,
  onEvent: EventHandler,
  signal?: AbortSignal
) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal
  });

  if (!response.ok) {
    let detail = "";
    try {
      const maybeJson = (await response.json()) as { detail?: string; error?: string; message?: string };
      detail = maybeJson?.detail || maybeJson?.error || maybeJson?.message || "";
    } catch {
      try {
        detail = await response.text();
      } catch {
        detail = "";
      }
    }
    throw new Error(toFriendlySseError(response.status, detail));
  }

  if (!response.body) {
    throw new Error(`SSE response body is empty: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      const lines = part.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) {
          event = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          data += line.slice(5).trim();
        }
      }
      onEvent({ event, data });
      if (event === "done") {
        return;
      }
    }
  }
}
