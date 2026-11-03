/** Tab-local previews only. Fresh authenticated state remains authoritative. */
export class SessionPreviewCache<T extends { session_id: string }> {
  private entries = new Map<string, { value: T; size: number }>();
  private size = 0;
  private maxEntries: number;
  private maxCharacters: number;
  constructor(maxEntries = 8, maxCharacters = 2_000_000) {
    this.maxEntries = maxEntries;
    this.maxCharacters = maxCharacters;
  }

  remember(value: T | null) {
    if (!value?.session_id) return;
    this.forget(value.session_id);
    const size = JSON.stringify(value).length;
    if (size > this.maxCharacters) return;
    this.entries.set(value.session_id, { value, size });
    this.size += size;
    while (this.entries.size > this.maxEntries || this.size > this.maxCharacters) {
      this.forget(this.entries.keys().next().value!);
    }
  }

  peek(id: string): T | null {
    const entry = this.entries.get(id);
    if (!entry) return null;
    this.entries.delete(id);
    this.entries.set(id, entry);
    return entry.value;
  }

  forget(id: string) {
    const entry = this.entries.get(id);
    if (entry) this.size -= entry.size;
    this.entries.delete(id);
  }
}
