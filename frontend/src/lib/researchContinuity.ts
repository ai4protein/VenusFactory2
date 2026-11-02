/** Only the newest view request may replace the selected research. */
export class ResearchSelection {
  id = "";
  revision = 0;
  select(id: string) { this.id = id; return ++this.revision; }
  current(id: string, revision: number) { return this.id === id && this.revision === revision; }
}

type DraftStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;
const prefix = "vf2_research_draft:";
export function readResearchDraft<T>(key: string, fallback: T, storage: DraftStorage = sessionStorage): T {
  try { const saved = JSON.parse(storage.getItem(prefix + key) || "null"); return saved && typeof saved === "object" ? { ...fallback, ...saved } : fallback; }
  catch { return fallback; }
}
export function writeResearchDraft<T>(key: string, value: T, storage: DraftStorage = sessionStorage) {
  try { storage.setItem(prefix + key, JSON.stringify(value)); } catch { /* In-memory editing remains available. */ }
}
export function moveResearchDraft(from: string, to: string, storage: DraftStorage = sessionStorage) {
  try { const saved = storage.getItem(prefix + from); if (saved) { storage.setItem(prefix + to, saved); storage.removeItem(prefix + from); } } catch { /* Storage may be unavailable. */ }
}
