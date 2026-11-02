const activeStates = new Set([
  "started", "running", "active", "preparing", "pending", "queued", "stopping",
  "thinking", "planning", "executing", "searching", "collecting", "chapters", "writing", "reviewing", "revising",
]);
export function isRunActive(status?: string | null): boolean {
  return activeStates.has((status || "").toLowerCase());
}
export function planProgress(steps: Array<{ id: string; status: string }>, checkpoint?: { completed_steps?: string[] }) {
  const complete = steps.filter(s => s.status === "completed").length;
  const reviewing = new Set((checkpoint?.completed_steps || []).filter(id => steps.some(s => s.id === id && s.status !== "completed")));
  return { complete, total: steps.length, reviewing };
}
