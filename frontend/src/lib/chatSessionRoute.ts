import type { ChatMode } from "./useModelRegistry";

type WorkspaceMode = ChatMode | "researcher";

export function chatRouteKey(sessionId: string, mode: WorkspaceMode) {
  return `${mode}:${sessionId}`;
}

export function chatSessionLocation(lang: string, sessionId: string, mode: WorkspaceMode, current = new URLSearchParams()) {
  const params = new URLSearchParams(current);
  params.delete("session_id");
  params.delete("new");
  if (mode === "science_agent") params.delete("mode");
  else params.set("mode", mode === "science_expert" ? "scientist" : "researcher");
  const suffix = sessionId ? `/${encodeURIComponent(sessionId)}` : "";
  return `/${lang === "zh" ? "zh" : "en"}/agent/chat${suffix}${params.size ? `?${params}` : ""}`;
}
