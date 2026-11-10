import { useCallback, useEffect, useReducer, useRef } from "react";
import { listChatSessions, type ChatSnapshot } from "./api";
import { hasSessionContent, readSessionCache, reduceSessionHistory, sessionHistoryState,
  titleFromHistory, writeSessionCache, type SessionMeta } from "./chatSessions";
import type { HistoryActivity } from "./optimisticHistory";
import type { ChatMode } from "./useModelRegistry";

/** Own sidebar metadata; navigation, messages and running jobs stay with their owners. */
export function useChatSessionHistory(sessionId: string, snapshot: ChatSnapshot | null, mode: ChatMode) {
  const [state, dispatch] = useReducer(reduceSessionHistory, undefined, () => {
    try { return sessionHistoryState(readSessionCache(localStorage)); }
    catch { return sessionHistoryState(); }
  });
  const requestId = useRef(0);
  const mounted = useRef(true);
  const setSessions = useCallback((update: SessionMeta[] | ((rows: SessionMeta[]) => SessionMeta[])) => {
    dispatch({ type: "update", update });
  }, []);
  const onHistoryActivity = useCallback((key: string, row: HistoryActivity | null) => {
    dispatch({ type: "activity", key, row });
  }, []);
  const fetchSessions = useCallback(async () => {
    const id = ++requestId.current;
    dispatch({ type: "request", id });
    const data = await listChatSessions();
    const sessions = (data.sessions || []).filter(hasSessionContent);
    if (mounted.current) dispatch({ type: "received", id, sessions });
    return sessions;
  }, []);

  useEffect(() => {
    mounted.current = true;
    let pending = false;
    const timer = window.setInterval(() => {
      if (pending) return;
      pending = true;
      void fetchSessions().catch(() => {}).finally(() => { pending = false; });
    }, 3000);
    return () => { mounted.current = false; window.clearInterval(timer); };
  }, [fetchSessions]);

  useEffect(() => {
    try { writeSessionCache(localStorage, state.sessions); }
    catch { /* Browser storage can be unavailable. */ }
  }, [state.sessions]);

  const activeSessionTitle = snapshot?.session_id === sessionId ? titleFromHistory(snapshot.history) : "";
  useEffect(() => {
    if (!sessionId || snapshot?.session_id !== sessionId) return;
    const historySize = snapshot.history?.length ?? 0;
    if (!activeSessionTitle && !historySize) return;
    setSessions(previous => {
      const prior = previous.find(row => row.session_id === sessionId);
      const title = activeSessionTitle || prior?.title || "";
      const status = snapshot.status || "";
      if (prior && prior.title === title && prior.history_size === historySize && prior.status === status) return previous;
      if (prior) return previous.map(row => row.session_id === sessionId
        ? { ...row, title, history_size: historySize, status } : row);
      return [{ session_id: sessionId, created_at: snapshot.created_at || new Date().toISOString(),
        model_name: snapshot.model_name || "", history_size: historySize, status,
        title, has_content: true, chat_mode: mode }, ...previous];
    });
  }, [sessionId, snapshot?.session_id, activeSessionTitle, snapshot?.history?.length, snapshot?.status, mode, setSessions]);

  return { sessions: state.sessions, historyActivity: state.activity, activeSessionTitle,
    setSessions, onHistoryActivity, fetchSessions };
}
