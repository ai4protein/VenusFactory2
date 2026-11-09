import type { ChatHistoryItem, ChatSnapshot } from "./api";

export type ChatStreamState = { snapshot: ChatSnapshot | null; streamingIdx: number };
export type SnapshotUpdate = ChatSnapshot | null | ((previous: ChatSnapshot | null) => ChatSnapshot | null);
export type ChatDelta = { content: string; role_id?: string; turn_id?: string };
export type StreamStart = { role_id?: string; turn_id?: string; kind?: string };
export type ChatStreamAction =
  | { type: "replace"; value: SnapshotUpdate }
  | { type: "state"; snapshot: ChatSnapshot }
  | { type: "start"; info: StreamStart }
  | { type: "tokens"; deltas: ChatDelta[] }
  | { type: "thinking"; delta: ChatDelta }
  | { type: "stop" };

const terminalStates = new Set([
  "completed", "error", "failed", "planning_failed", "execution_failed", "stopped",
  "cancelled", "canceled", "aborted", "interrupted", "paused", "done",
]);

/** These checkpoints are authoritative; a local unfinished answer cannot replace them. */
export function isChatStreamBoundary(status: string | undefined): boolean {
  const value = (status || "").toLowerCase();
  return !value || value.startsWith("waiting_") || terminalStates.has(value);
}

function isThinkingPlaceholder(item: ChatHistoryItem): boolean {
  return item.role === "assistant" && (item.kind === "thinking" || item.phase === "thinking"
    || /preparing clarification|准备澄清/i.test(item.content || ""));
}

export function sanitizeChatSnapshot(snapshot: ChatSnapshot): ChatSnapshot {
  const history = Array.isArray(snapshot.history) ? snapshot.history : [];
  if (!isChatStreamBoundary(snapshot.status)) return snapshot;
  const cleaned = history.filter(item => !isThinkingPlaceholder(item));
  return cleaned.length === history.length && history === snapshot.history
    ? snapshot : { ...snapshot, history: cleaned };
}

/** Legacy graph status messages predate structured phase markers. */
function isTextPlaceholder(item: ChatHistoryItem): boolean {
  if (item.role !== "assistant" || item.kind === "thinking"
      || item.kind === "checkpoint" || item.kind === "compaction" || item.kind === "model_result") return false;
  return Boolean(item.phase) || /思考中|Thinking|正在设计|designing the pipeline|准备澄清|preparing clarification|撰写|Summarizing|正在总结|汇总|writing sub-report|writing the draft report/i.test(item.content || "");
}

type MessageIdentity = Pick<ChatHistoryItem, "turn_id" | "role_id">;
function sameIdentity(left: MessageIdentity, right: MessageIdentity): boolean {
  return !(left.turn_id && right.turn_id && left.turn_id !== right.turn_id)
    && !(left.role_id && right.role_id && left.role_id !== right.role_id);
}

function sameHistoryTurn(previous: ChatSnapshot, incoming: ChatSnapshot): boolean {
  if (previous.session_id !== incoming.session_id || previous.history.length !== incoming.history.length) return false;
  if (previous.active_run?.id && incoming.active_run?.id && previous.active_run.id !== incoming.active_run.id) return false;
  return previous.history.every((item, index) => {
    const next = incoming.history[index];
    return item.role !== "user" && next.role !== "user"
      || item.role === next.role && item.content === next.content && sameIdentity(item, next);
  });
}

/** Preserve an in-flight tail only within the same session, run and user turn. */
export function mergeChatSnapshot(previous: ChatSnapshot | null, incoming: ChatSnapshot): ChatSnapshot {
  if (isChatStreamBoundary(incoming.status)) return sanitizeChatSnapshot(incoming);
  if (!previous?.history?.length || !incoming.history?.length || !sameHistoryTurn(previous, incoming)) return incoming;
  const last = previous.history[previous.history.length - 1];
  const next = incoming.history[incoming.history.length - 1];
  const text = (last.content || "").trim(), nextText = (next.content || "").trim();
  if (last.role !== "assistant" || next.role !== "assistant" || last.kind === "thinking"
      || next.kind === "checkpoint" || next.kind === "compaction" || next.kind === "model_result"
      || last.kind === "model_result" || !sameIdentity(last, next)
      || text.length <= nextText.length
      || !(isTextPlaceholder(next) || !nextText || text.startsWith(nextText))) return incoming;
  const history = [...incoming.history];
  history[history.length - 1] = {
    ...next, content: last.content, role_id: last.role_id || next.role_id,
    turn_id: last.turn_id || next.turn_id, kind: last.kind || "text", phase: undefined,
  };
  return { ...incoming, history };
}

function startStream(snapshot: ChatSnapshot, info: StreamStart): ChatSnapshot {
  const history = [...snapshot.history], last = history[history.length - 1];
  const identity = { turn_id: info.turn_id, role_id: info.role_id };
  if (info.kind === "thinking") {
    if (last?.role === "assistant" && last.kind === "thinking" && sameIdentity(last, identity)) {
      history[history.length - 1] = { ...last, turn_id: info.turn_id || last.turn_id, role_id: info.role_id || last.role_id };
    } else {
      history.push({ role: "assistant", content: "", ...identity, kind: "thinking", phase: "thinking" });
    }
  } else if (last && sameIdentity(last, identity)
      && (isTextPlaceholder(last) || last.role === "assistant" && last.kind !== "thinking" && last.kind !== "model_result" && !(last.content || "").trim())) {
    history[history.length - 1] = {
      role: "assistant", content: "", role_id: info.role_id || last.role_id,
      turn_id: info.turn_id || last.turn_id, kind: "text",
    };
  } else {
    history.push({ role: "assistant", content: "", ...identity, kind: "text" });
  }
  return { ...snapshot, history };
}

function appendDeltas(snapshot: ChatSnapshot, deltas: ChatDelta[], thinking = false): ChatSnapshot {
  const history = [...snapshot.history];
  for (const delta of deltas) {
    if (!delta.content) continue;
    const last = history[history.length - 1];
    const kind = thinking ? "thinking" : "text";
    if (last?.role === "assistant" && (last.kind === "thinking") === thinking
        && !last.phase?.startsWith("research_") && last.kind !== "checkpoint" && last.kind !== "compaction" && last.kind !== "model_result"
        && sameIdentity(last, delta)) {
      history[history.length - 1] = { ...last, content: (last.content || "") + delta.content };
    } else {
      history.push({ role: "assistant", ...delta, kind });
    }
  }
  return { ...snapshot, history };
}

/** Pure display transitions. Transport, credentials and server state live elsewhere. */
export function chatStreamReducer(state: ChatStreamState, action: ChatStreamAction): ChatStreamState {
  if (action.type === "replace") {
    const snapshot = typeof action.value === "function" ? action.value(state.snapshot) : action.value;
    return { snapshot: snapshot ? sanitizeChatSnapshot(snapshot) : null, streamingIdx: -1 };
  }
  if (action.type === "stop") return { ...state, streamingIdx: -1 };
  if (action.type === "state") {
    const previous = state.snapshot, incoming = action.snapshot;
    const sameTail = previous?.history?.length && incoming.history?.length
      && sameHistoryTurn(previous, incoming)
      && previous.history[previous.history.length - 1].kind === incoming.history[incoming.history.length - 1].kind
      && sameIdentity(previous.history[previous.history.length - 1], incoming.history[incoming.history.length - 1]);
    return {
      snapshot: mergeChatSnapshot(previous, incoming),
      streamingIdx: !sameTail || isChatStreamBoundary(incoming.status) ? -1 : state.streamingIdx,
    };
  }
  if (!state.snapshot) return state;
  if (action.type === "tokens" && !action.deltas.some(delta => delta.content)
      || action.type === "thinking" && !action.delta.content) return state;
  const snapshot = action.type === "start" ? startStream(state.snapshot, action.info)
    : action.type === "thinking" ? appendDeltas(state.snapshot, [action.delta], true)
    : appendDeltas(state.snapshot, action.deltas);
  return { snapshot, streamingIdx: snapshot.history.length - 1 };
}
