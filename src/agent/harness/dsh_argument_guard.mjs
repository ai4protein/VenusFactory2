// Validate the original model payload before native MCP dispatch.
export const name = "venus-tool-arguments";
export const inject = ["tools", "sessions"];

export function argumentError(raw) {
  try {
    const value = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      return "Tool arguments must be a JSON object.";
    }
    const pending = [value];
    while (pending.length) {
      const item = pending.pop();
      if (typeof item === "number" && !Number.isFinite(item)) {
        return "Tool arguments must contain only finite numbers.";
      }
      if (item !== null && typeof item === "object") pending.push(...Object.values(item));
    }
  } catch {
    return "Invalid tool arguments: send one complete JSON object.";
  }
}

export function apply(ctx) {
  const calls = new Map();
  ctx.on("session/event", (session, event) => {
    if (event.type === "tool/call") {
      const data = event.data;
      let entries = calls.get(session.id);
      if (!entries) calls.set(session.id, entries = new Map());
      entries.set(data.callId, {name: data.name, error: argumentError(data.arguments)});
    } else if (event.type === "turn/end") {
      calls.delete(session.id);
    }
  });
  ctx.on("session/disposed", session => calls.delete(session.id));
  ctx.on("tools/pre-execute", (exec, next) => {
    if (!exec.name.startsWith("mcp__venusfactory__")) return next();
    const entries = calls.get(exec.agent?.id);
    const original = entries?.get(exec.callId);
    entries?.delete(exec.callId);
    const reason = !original || original.name !== exec.name
      ? "Cannot validate original tool arguments. Retry with a complete JSON object."
      : original.error;
    if (reason) return {kind: "deny", reason};
    return next();
  });
}
