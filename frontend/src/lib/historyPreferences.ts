export type HistoryPins = Record<string, string[]>;

export function parseHistoryPins(value: string | null): HistoryPins {
  try {
    const parsed: unknown = JSON.parse(value || "{}");
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") return {};
    return Object.fromEntries(Object.entries(parsed).filter(([, ids]) => Array.isArray(ids))
      .map(([scope, ids]) => [scope, [...new Set((ids as unknown[]).filter((id): id is string => typeof id === "string" && !!id))]]));
  } catch { return {}; }
}

export function updateHistoryPin(pins: HistoryPins, scope: string, id: string, pinned: boolean): HistoryPins {
  const remaining = (pins[scope] || []).filter(value => value !== id);
  return { ...pins, [scope]: pinned ? [id, ...remaining] : remaining };
}

export function pinnedFirst<T>(items: T[], ids: string[], getId: (item: T) => string): T[] {
  const rank = new Map(ids.map((id, index) => [id, index]));
  return [...items].sort((a, b) => (rank.get(getId(a)) ?? ids.length) - (rank.get(getId(b)) ?? ids.length));
}

export function historyTime(value: string | number, lang: string): string {
  const date = new Date(value);
  if (!value || Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(lang === "zh" ? "zh-CN" : "en-GB", {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", timeZoneName: "short",
  }).format(date);
}
