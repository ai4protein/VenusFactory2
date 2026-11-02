import type { ChatMode } from "./useModelRegistry";

export type ModeSession = { session_id: string; chat_mode?: ChatMode };

export function activeSessionStorageKey(mode: ChatMode) {
  return `vf2_active_session_id:${mode}`;
}

export function sessionsForMode<T extends ModeSession>(sessions: T[], mode: ChatMode): T[] {
  return sessions.filter(session => (session.chat_mode || "science_agent") === mode);
}

export function sessionCandidates(sessions: ModeSession[], mode: ChatMode, remembered: string | null, legacy: string | null = null) {
  if (remembered === "") return []; // An explicitly opened draft stays a draft.
  const ids = sessionsForMode(sessions, mode).map(session => session.session_id);
  const preferred = remembered ?? legacy;
  return preferred && ids.includes(preferred) ? [preferred, ...ids.filter(id => id !== preferred)] : ids;
}
