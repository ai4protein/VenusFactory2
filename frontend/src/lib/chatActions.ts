import type { SseEvent } from "./sse";

type StreamSlot = { current: AbortController | null };

type StreamAction = {
  stream: StreamSlot;
  current: () => boolean;
  request: (signal: AbortSignal, onEvent: (event: SseEvent) => void) => Promise<void>;
  onStart: () => void;
  onEvent: (event: SseEvent) => void;
  onError: (error: unknown) => void | Promise<void>;
  onFinish: () => void;
  onSuccess?: () => void;
};

/** One attached action owns the view; detached actions may finish on the server. */
export async function runChatStreamAction(action: StreamAction): Promise<boolean> {
  if (action.stream.current || !action.current()) return false;
  const controller = new AbortController();
  action.stream.current = controller;
  const ownsView = () => action.stream.current === controller && action.current();
  let failedFrame = false;
  try {
    action.onStart();
    await action.request(controller.signal, event => {
      if (!ownsView() || controller.signal.aborted) return;
      if (event.event === "error") failedFrame = true;
      action.onEvent(event);
    });
    if (!ownsView() || controller.signal.aborted || failedFrame) return false;
    action.onSuccess?.();
    return true;
  } catch (error) {
    if (ownsView() && !controller.signal.aborted) await action.onError(error);
    return false;
  } finally {
    if (action.stream.current === controller) {
      const current = action.current();
      action.stream.current = null;
      if (current) action.onFinish();
    }
  }
}

type UpdateAction<T> = {
  current: () => boolean;
  request: () => Promise<T>;
  onSuccess: (result: T) => void | Promise<void>;
  onError: (error: unknown) => void;
};

/** Apply an acknowledged update only while its original session is still visible. */
export async function runChatUpdateAction<T>(action: UpdateAction<T>): Promise<boolean> {
  if (!action.current()) return false;
  try {
    const result = await action.request();
    if (!action.current()) return false;
    await action.onSuccess(result);
    return true;
  } catch (error) {
    if (action.current()) action.onError(error);
    return false;
  }
}

type CancelAction = Omit<UpdateAction<unknown>, "onSuccess"> & {
  stream: StreamSlot;
  onStart: () => void;
  onSuccess: () => void;
};

export async function cancelChatStreamAction(action: CancelAction): Promise<boolean> {
  if (!action.current()) return false;
  const controller = action.stream.current;
  const ownsView = () => action.current() && action.stream.current === controller;
  action.onStart();
  try {
    await action.request();
    // Never dereference the current slot after awaiting a different session's stop.
    controller?.abort();
    if (ownsView()) action.onSuccess();
    return true;
  } catch (error) {
    if (ownsView()) action.onError(error);
    return false;
  }
}
