export function submitsComposer(event: { key: string; shiftKey?: boolean; altKey?: boolean; isComposing?: boolean; keyCode?: number }): boolean {
  return event.key === "Enter" && !event.shiftKey && !event.altKey && !event.isComposing && event.keyCode !== 229;
}
