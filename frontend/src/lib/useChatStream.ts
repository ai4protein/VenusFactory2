import { useCallback, useEffect, useReducer, useState } from "react";
import type { ChatSnapshot } from "./api";
import type { SseEvent } from "./sse";
import { chatStreamReducer, type ChatDelta, type SnapshotUpdate, type StreamStart } from "./chatStreamState";
import { ChatTokenQueue } from "./chatTokenQueue";

/** Owns the temporary rendered stream, independently of the persisted session. */
export function useChatStream() {
  const [state, dispatch] = useReducer(chatStreamReducer, { snapshot: null, streamingIdx: -1 });
  const [queue] = useState(() => new ChatTokenQueue(
    deltas => dispatch({ type: "tokens", deltas }),
    callback => window.requestAnimationFrame(callback),
    frame => window.cancelAnimationFrame(frame),
  ));
  useEffect(() => () => queue.reset(), [queue]);

  const setSnapshot = useCallback((value: SnapshotUpdate) => {
    queue.reset();
    dispatch({ type: "replace", value });
  }, [queue]);
  const stopStream = useCallback(() => {
    queue.flush();
    dispatch({ type: "stop" });
  }, [queue]);
  const discardStream = useCallback(() => {
    queue.reset();
    dispatch({ type: "stop" });
  }, [queue]);
  const updateStream = useCallback(({ event, data }: SseEvent): ChatSnapshot | undefined => {
    if (!data) return;
    if (event === "state") {
      const snapshot = JSON.parse(data) as ChatSnapshot;
      queue.flush();
      dispatch({ type: "state", snapshot });
      return snapshot;
    }
    if (event === "stream_start") {
      const info = JSON.parse(data) as StreamStart;
      queue.flush();
      dispatch({ type: "start", info });
    } else if (event === "token") {
      queue.push(JSON.parse(data) as ChatDelta);
    } else if (event === "thinking") {
      const delta = JSON.parse(data) as ChatDelta;
      queue.flush();
      dispatch({ type: "thinking", delta });
    }
  }, [queue]);

  return { ...state, setSnapshot, stopStream, discardStream, updateStream };
}
