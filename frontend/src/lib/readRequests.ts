// Coalesce GETs only. Never share mutations or retain session snapshots.
const pending = new Map<string, Promise<unknown>>();
export function shareRead<T>(key: string, load: () => Promise<T>): Promise<T> {
  const existing = pending.get(key);
  if (existing) return existing as Promise<T>;
  const promise = Promise.resolve().then(load).finally(() => {
    if (pending.get(key) === promise) pending.delete(key);
  });
  pending.set(key, promise);
  return promise;
}

/** End a read generation after a successful mutation without cancelling its readers. */
export function invalidateRead(key: string) {
  pending.delete(key);
}
