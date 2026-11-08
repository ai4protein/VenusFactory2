import type { ChatDelta } from "./chatStreamState";

/** One render batch per frame; cancelled callbacks cannot drain a later session. */
export class ChatTokenQueue {
  private pending: ChatDelta[] = [];
  private frame: number | null = null;
  private revision = 0;
  private readonly deliver: (deltas: ChatDelta[]) => void;
  private readonly requestFrame: (callback: () => void) => number;
  private readonly cancelFrame: (frame: number) => void;

  constructor(deliver: (deltas: ChatDelta[]) => void,
              requestFrame: (callback: () => void) => number,
              cancelFrame: (frame: number) => void) {
    this.deliver = deliver;
    this.requestFrame = requestFrame;
    this.cancelFrame = cancelFrame;
  }

  push(delta: ChatDelta): void {
    if (!delta.content) return;
    this.pending.push(delta);
    if (this.frame !== null) return;
    const revision = this.revision;
    this.frame = this.requestFrame(() => {
      if (revision !== this.revision) return;
      this.frame = null;
      const pending = this.pending;
      this.pending = [];
      if (pending.length) this.deliver(pending);
    });
  }

  /** Flush before a new message or authoritative server snapshot. */
  flush(): void {
    const pending = this.pending;
    this.reset();
    if (pending.length) this.deliver(pending);
  }

  /** Discard buffered text when leaving a conversation or unmounting. */
  reset(): void {
    this.revision += 1;
    if (this.frame !== null) this.cancelFrame(this.frame);
    this.frame = null;
    this.pending = [];
  }
}
