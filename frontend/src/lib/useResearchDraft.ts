import { useLayoutEffect, useRef, useState, type SetStateAction } from "react";
import { readResearchDraft, writeResearchDraft } from "./researchContinuity";

/** Persist edits at the time of input, never from a stale hydration effect. */
export function useResearchDraft<T>(key: string, fallback: T) {
  const [entry, setEntry] = useState(() => ({ key, value: readResearchDraft(key, fallback) }));
  const current = useRef(entry);
  const value = entry.key === key ? entry.value : readResearchDraft(key, fallback);
  useLayoutEffect(() => { current.current = { key, value }; }, [key, value]);
  function setValue(update: SetStateAction<T>) {
    const next = typeof update === "function" ? (update as (v: T) => T)(current.current.value) : update;
    const saved = { key: current.current.key, value: next };
    current.current = saved; writeResearchDraft(saved.key, next); setEntry(saved);
  }
  return [value, setValue] as const;
}
