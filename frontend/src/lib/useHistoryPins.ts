import { useEffect, useState } from "react";
import { parseHistoryPins, updateHistoryPin } from "./historyPreferences";

const KEY = "vf2_history_pins_v1";
const EVENT = "vf2:history-pins";
function readPins() {
  try { return parseHistoryPins(localStorage.getItem(KEY)); } catch { return {}; }
}

/** A browser preference, scoped separately to each workspace's history. */
export function useHistoryPins(scope: string) {
  const [pins, setPins] = useState(readPins);
  useEffect(() => {
    const sync = () => setPins(readPins());
    const storage = (event: StorageEvent) => { if (!event.key || event.key === KEY) sync(); };
    window.addEventListener("storage", storage);
    window.addEventListener(EVENT, sync);
    return () => { window.removeEventListener("storage", storage); window.removeEventListener(EVENT, sync); };
  }, []);
  function setPinned(id: string, pinned: boolean) {
    let latest = pins;
    try { latest = parseHistoryPins(localStorage.getItem(KEY)); } catch { /* Keep in-memory preferences. */ }
    const next = updateHistoryPin(latest, scope, id, pinned);
    setPins(next);
    try {
      localStorage.setItem(KEY, JSON.stringify(next));
      window.dispatchEvent(new Event(EVENT));
    } catch { /* The current tab still supports pinning when storage is unavailable. */ }
  }
  return { ids: pins[scope] || [], setPinned };
}
